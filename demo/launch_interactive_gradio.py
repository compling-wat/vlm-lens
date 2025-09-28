"""Gradio demo for visualizing VLM first token probability distributions."""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          Qwen2VLForConditionalGeneration)

from demo.lookup import get_model_info
from src.models.config import ModelSelection  # noqa: E402

models_cache: Dict[str, Any] = {}
processors_cache: Dict[str, Any] = {}
current_model_selection: Optional[ModelSelection] = None


def read_layer_spec(spec_file_path: str) -> List[str]:
    """Read available layers from the model spec file.

    Args:
        spec_file_path: Path to the model specification file.

    Returns:
        List of available layer names, skipping blank lines.
    """
    try:
        with open(spec_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Filter out blank lines and strip whitespace
        layers = [line.strip() for line in lines if line.strip()]
        return layers

    except FileNotFoundError:
        print(f'Spec file not found: {spec_file_path}')
        return ['Default layer (spec file not found)']
    except Exception as e:
        print(f'Error reading spec file: {str(e)}')
        return ['Default layer (error reading spec)']


def update_layer_choices(model_choice: str) -> Tuple[gr.Dropdown, gr.Button]:
    """Update layer dropdown choices based on selected model.

    Args:
        model_choice: Selected model name.

    Returns:
        Updated dropdown component and button visibility.
    """
    if not model_choice:
        return gr.Dropdown(choices=[], visible=False), gr.Button(visible=False)

    try:
        # Convert string choice to ModelSelection enum
        model_selection = ModelSelection(model_choice.lower())

        # Get model info and read layer spec
        model_path, model_spec_path = get_model_info(model_selection)
        layers = read_layer_spec(model_spec_path)

        # Return updated dropdown with layer choices and make button visible
        return (
            gr.Dropdown(
                choices=layers,
                label=f'Select Layer for {model_choice}',
                value=layers[0] if layers else None,
                visible=True,
                interactive=True
            ),
            gr.Button('Analyze', variant='primary', visible=True)
        )

    except ValueError:
        return (
            gr.Dropdown(
                choices=['Model not implemented'],
                label='Select Layer',
                visible=True,
                interactive=False
            ),
            gr.Button('Analyze', variant='primary', visible=False)
        )
    except Exception as e:
        return (
            gr.Dropdown(
                choices=[f'Error: {str(e)}'],
                label='Select Layer',
                visible=True,
                interactive=False
            ),
            gr.Button('Analyze', variant='primary', visible=False)
        )


def load_model(model_selection: ModelSelection) -> Tuple[Any, Any]:
    """Load the specified VLM model and processor.

    Args:
        model_selection: The model to load from ModelSelection enum.

    Returns:
        Tuple containing the loaded model and processor.

    Raises:
        NotImplementedError: If the model is not implemented.
    """
    global models_cache, processors_cache, current_model_selection

    model_key = model_selection.value

    # Check if model is already loaded
    if model_key in models_cache and model_key in processors_cache:
        current_model_selection = model_selection
        return models_cache[model_key], processors_cache[model_key]

    print(f'Loading {model_selection.value} model...')

    try:
        model_path, model_spec = get_model_info(model_selection)
        print(f'Model path: {model_path}')
        print(f'Model spec: {model_spec}')

        if model_selection == ModelSelection.LLAVA:
            # Load LLaVA model
            processor = AutoProcessor.from_pretrained(model_path)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map='auto'
            )
        elif model_selection == ModelSelection.QWEN:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map='auto'
            )
        else:
            # For other models, raise NotImplementedError for now
            raise NotImplementedError(f'Model {model_selection.value} is not yet implemented')

        # Cache the loaded model and processor
        models_cache[model_key] = model
        processors_cache[model_key] = processor
        current_model_selection = model_selection

        print(f'{model_selection.value} model loaded successfully!')
        return model, processor

    except Exception as e:
        print(f'Error loading model {model_selection.value}: {str(e)}')
        raise


def get_first_token_probabilities(
    instruction: str,
    image: Image.Image,
    model_selection: ModelSelection,
    selected_layer: str
) -> Tuple[List[str], np.ndarray]:
    """Process the instruction and image through the selected VLM and return first token probabilities.

    Args:
        instruction: Text instruction for the model.
        image: PIL Image to process.
        model_selection: The VLM to use.
        selected_layer: The selected layer for analysis.

    Returns:
        Tuple containing list of top tokens and their probabilities.

    Raises:
        NotImplementedError: If the model is not implemented.
    """
    try:
        # Load model if not already loaded or if different model selected
        model, processor = load_model(model_selection)

        print(f'Using layer: {selected_layer}')

        if model_selection in [ModelSelection.LLAVA, ModelSelection.QWEN]:

            messages = [{
                'role': 'user',
                'content': [{'type': 'image'}, {'type': 'text', 'text': instruction}],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors='pt',
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,  # Only generate first token
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False
                )
        else:
            raise NotImplementedError(f'Model {model_selection.value} processing not yet implemented')

        # Get the logits for the first generated token
        first_token_logits = outputs.scores[0][0]  # Shape: [vocab_size]

        # Convert logits to probabilities
        probabilities = torch.softmax(first_token_logits, dim=-1)

        # Get top-k probabilities for visualization
        top_k = 20
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Convert tokens back to text
        top_tokens = [processor.tokenizer.decode([idx.item()]) for idx in top_indices]

        return top_tokens, top_probs.cpu().numpy()

    except Exception as e:
        print(f'Error in processing: {str(e)}')
        return [], np.array([])


def create_probability_plot(
    tokens: List[str],
    probabilities: np.ndarray,
    model_name: str,
    layer_name: str
) -> Figure:
    """Create a matplotlib plot of token probabilities.

    Args:
        tokens: List of token strings.
        probabilities: Array of probability values.
        model_name: Name of the model for the plot title.
        layer_name: Name of the selected layer.

    Returns:
        Matplotlib Figure object.
    """
    if len(tokens) == 0:
        # Return empty plot if no data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data to display',
                horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bar plot
    bars = ax.bar(range(len(tokens)), probabilities, color='skyblue',
                  edgecolor='navy', alpha=0.7)

    # Customize the plot
    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'First Token Probability Distribution\n{model_name.upper()} - Layer: {layer_name}',
                 fontsize=14, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')

    # Add probability values on top of bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


def process_inputs(
    model_choice: str,
    selected_layer: str,
    instruction: str,
    image: Optional[Image.Image]
) -> Tuple[Optional[Figure], str]:
    """Main function to process inputs and return plot.

    Args:
        model_choice: String name of the selected model.
        selected_layer: String name of the selected layer.
        instruction: Text instruction for the model.
        image: PIL Image to process, can be None.

    Returns:
        Tuple containing the plot figure and info text.
    """
    if image is None:
        return None, 'Please upload an image.'

    if not instruction.strip():
        return None, 'Please provide an instruction.'

    if not model_choice:
        return None, 'Please select a model.'

    if not selected_layer:
        return None, 'Please select a layer.'

    try:
        # Convert string choice to ModelSelection enum
        model_selection = ModelSelection(model_choice.lower())

        # Get token probabilities
        tokens, probabilities = get_first_token_probabilities(
            instruction, image, model_selection, selected_layer
        )

        if len(tokens) == 0:
            return None, 'Error: Could not process the inputs. Please check the model loading.'

        # Create plot
        plot = create_probability_plot(tokens, probabilities, model_choice, selected_layer)

        # Create info text
        info_text = f'Model: {model_choice.upper()}\n'
        info_text += f'Layer: {selected_layer}\n'
        info_text += f"Instruction: \'{instruction}\'\n"
        info_text += f"Top token: \'{tokens[0]}\' (probability: {probabilities[0]:.4f})"

        return plot, info_text

    except ValueError as e:
        return None, f'Invalid model selection: {str(e)}'
    except Exception as e:
        return None, f'Error: {str(e)}'


def get_available_models() -> List[str]:
    """Get list of available model names for the dropdown.

    Returns:
        List of model names as strings.
    """
    # For now, only return implemented models
    implemented_models = [
        ModelSelection.LLAVA.value,
        ModelSelection.QWEN.value
    ]
    return [model.capitalize() for model in implemented_models]


def create_demo() -> gr.Blocks:
    """Create and configure the Gradio demo interface.

    Returns:
        Configured Gradio Blocks interface.
    """
    with gr.Blocks(title='VLM Token Probability Visualizer') as demo:
        gr.Markdown("""
        # Vision-Language Model First Token Probability Distribution

        This demo processes an instruction and image through various Vision-Language Models (VLMs)
        and visualizes the probability distribution of the first token in the response.

        **Instructions:**
        1. Select a VLM model from the dropdown
        2. Select a layer from the available embedding layers
        3. Upload an image
        4. Enter your instruction/question about the image
        5. Click "Analyze" to see the first token probability distribution

        **Note:** Currently only LLaVA is implemented. Other models will be added soon.
        """)

        with gr.Row():
            with gr.Column():
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label='Select VLM',
                    value=None,
                    interactive=True
                )

                layer_dropdown = gr.Dropdown(
                    choices=[],
                    label='Select Layer',
                    visible=False,
                    interactive=True
                )

                instruction_input = gr.Textbox(
                    label='Instruction',
                    placeholder='Describe what you see in this image...',
                    lines=3
                )
                image_input = gr.Image(
                    label='Upload Image',
                    type='pil'
                )
                analyze_btn = gr.Button('Analyze', variant='primary', visible=False)

            with gr.Column():
                plot_output = gr.Plot(label='First Token Probability Distribution')
                info_output = gr.Textbox(
                    label='Analysis Info',
                    lines=5,
                    interactive=False
                )

        # Set up event handlers
        model_dropdown.change(
            fn=update_layer_choices,
            inputs=[model_dropdown],
            outputs=[layer_dropdown, analyze_btn]
        )

        analyze_btn.click(
            fn=process_inputs,
            inputs=[model_dropdown, layer_dropdown, instruction_input, image_input],
            outputs=[plot_output, info_output]
        )

        # Add examples
        gr.Examples(
            examples=[
                ['What is in this image? Describe in one word.', None],
                ['Describe the main object in the picture in one word.', None],
                ['What color is the dominant object? Describe in one word.', None],
            ],
            inputs=[instruction_input, image_input]
        )

    return demo


if __name__ == '__main__':
    # Create and launch the demo
    demo = create_demo()
    demo.launch(
        share=True,
        server_name='0.0.0.0',
        server_port=7860
    )
