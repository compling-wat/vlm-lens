"""Gradio demo for visualizing VLM first token probability distributions with two images."""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from demo.lookup import ModelVariants, get_model_info  # noqa: E402
from src.main import get_model  # noqa: E402
from src.models.base import ModelBase  # noqa: E402
from src.models.config import Config, ModelSelection  # noqa: E402

models_cache: Dict[str, Any] = {}
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
        # Convert string choice to ModelVariants enum
        model_var = ModelVariants(model_choice.lower())

        # Get model info and read layer spec
        _, _, model_spec_path = get_model_info(model_var)
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


def load_model(model_var: ModelVariants, config: Config) -> ModelBase:
    """Load the specified VLM and processor.

    Args:
        model_var: The model to load from ModelVariants enum.
        config: The configuration object with model parameters.

    Returns:
        The loaded model instance.

    Raises:
        Exception: If model loading fails.
    """
    global models_cache, current_model_selection

    model_key = model_var.value

    # Check if model is already loaded
    if model_key in models_cache:
        current_model_selection = model_var
        return models_cache[model_key]

    print(f'Loading {model_var.value} model...')

    try:
        model_selection = config.architecture
        model = get_model(config.architecture, config)

        # Cache the loaded model and processor
        models_cache[model_key] = model
        current_model_selection = model_selection

        print(f'{model_selection.value} model loaded successfully!')
        return model

    except Exception as e:
        print(f'Error loading model {model_selection.value}: {str(e)}')
        raise


def get_single_image_probabilities(
    instruction: str,
    image: Image.Image,
    vlm: ModelBase,
    model_selection: ModelSelection,
    top_k: int = 8
) -> Tuple[List[str], np.ndarray]:
    """Process a single image and return first token probabilities.

    Args:
        instruction: Text instruction for the model.
        image: PIL Image to process.
        vlm: Loaded model.
        model_selection: The VLM being used.
        top_k: Number of top tokens to return.

    Returns:
        Tuple containing list of top tokens and their probabilities.

    Raises:
        NotImplementedError: If the model processing is not implemented.
    """
    if model_selection in [ModelSelection.LLAVA, ModelSelection.QWEN]:

        # Generate prompt and process inputs
        text = vlm._generate_prompt(instruction, has_images=True)
        inputs = vlm._generate_processor_output(text, image)

        with torch.no_grad():
            outputs = vlm.model.generate(
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
    top_probs, top_indices = torch.topk(probabilities, top_k)

    # Convert tokens back to text
    top_tokens = [vlm.processor.tokenizer.decode([idx.item()]) for idx in top_indices]

    return top_tokens, top_probs.cpu().numpy()


def scale_figure_fonts(fig: Figure, factor: float = 1.5) -> None:
    """Multiply all text sizes in a Matplotlib Figure by `factor`.

    Args:
        fig: The Matplotlib Figure to scale.
        factor: The scaling factor (e.g., 1.5 to increase by 50%).
    """
    for ax in fig.get_axes():
        # titles & axis labels
        ax.title.set_fontsize(ax.title.get_fontsize() * factor)
        ax.xaxis.label.set_size(ax.xaxis.label.get_size() * factor)
        ax.yaxis.label.set_size(ax.yaxis.label.get_size() * factor)
        # tick labels
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(lbl.get_fontsize() * factor)
        # texts placed via ax.text(...) (e.g., numbers above bars / "No data" notes)
        for t in ax.texts:
            t.set_fontsize(t.get_fontsize() * factor)
    # any stray Text artists attached to the figure (rare, but safe)
    for t in fig.findobj(match=Text):
        if t.figure is fig:
            t.set_fontsize(t.get_fontsize() * factor)


def create_dual_probability_plot(
    tokens1: List[str], probabilities1: np.ndarray,
    tokens2: List[str], probabilities2: np.ndarray
) -> Figure:
    """Create a matplotlib plot comparing token probabilities from two images.

    Args:
        tokens1: List of token strings from first image.
        probabilities1: Array of probability values from first image.
        tokens2: List of token strings from second image.
        probabilities2: Array of probability values from second image.

    Returns:
        Matplotlib Figure object.
    """
    if len(tokens1) == 0 and len(tokens2) == 0:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.text(0.5, 0.5, 'No data to display',
                horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    # Unify y-range with padding (cap at 1.0)
    max1 = float(np.max(probabilities1)) if len(tokens1) else 0.0
    max2 = float(np.max(probabilities2)) if len(tokens2) else 0.0
    y_upper = min(1.0, max(max1, max2) * 1.15 + 1e-6)  # ~15% headroom

    # Create subplots side by side with shared y
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), sharey=True)
    ax1.set_ylim(0, y_upper)
    ax2.set_ylim(0, y_upper)

    # Plot first image results
    if len(tokens1) > 0:
        bars1 = ax1.bar(range(len(tokens1)), probabilities1, color='lightcoral',
                        edgecolor='darkred', alpha=0.7)
        ax1.set_xlabel('Tokens', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('Image 1 - First Token Probabilities',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(tokens1)))
        ax1.set_xticklabels(tokens1, rotation=45, ha='right')

        # Clamp label position so it stays inside the axes
        for bar, prob in zip(bars1, probabilities1):
            h = bar.get_height()
            y = min(h + 0.02 * y_upper, y_upper * 0.98)
            ax1.text(bar.get_x() + bar.get_width()/2., y, f'{prob:.3f}',
                     ha='center', va='bottom', fontsize=9)

        ax1.grid(axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No data for Image 1',
                 horizontalalignment='center', verticalalignment='center')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

    # Plot second image results
    if len(tokens2) > 0:
        bars2 = ax2.bar(range(len(tokens2)), probabilities2, color='skyblue',
                        edgecolor='navy', alpha=0.7)
        ax2.set_xlabel('Tokens', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Image 2 - First Token Probabilities',
                      fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(tokens2)))
        ax2.set_xticklabels(tokens2, rotation=45, ha='right')

        for bar, prob in zip(bars2, probabilities2):
            h = bar.get_height()
            y = min(h + 0.02 * y_upper, y_upper * 0.98)
            ax2.text(bar.get_x() + bar.get_width()/2., y, f'{prob:.3f}',
                     ha='center', va='bottom', fontsize=9)

        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data for Image 2',
                 horizontalalignment='center', verticalalignment='center')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    # Give extra space for rotated tick labels
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    return fig


def process_dual_inputs(
    model_choice: str,
    selected_layer: str,
    instruction: str,
    image1: Optional[Image.Image],
    image2: Optional[Image.Image],
    top_k: int = 8
) -> Tuple[Optional[Figure], str]:
    """Main function to process dual inputs and return comparison plot.

    Args:
        model_choice: String name of the selected model.
        selected_layer: String name of the selected layer.
        instruction: Text instruction for the model.
        image1: First PIL Image to process, can be None.
        image2: Second PIL Image to process, can be None.
        top_k: Number of top tokens to display.

    Returns:
        Tuple containing the plot figure and info text.
    """
    if image1 is None and image2 is None:
        return None, 'Please upload at least one image.'

    if not instruction.strip():
        return None, 'Please provide an instruction.'

    if not model_choice:
        return None, 'Please select a model.'

    if not selected_layer:
        return None, 'Please select a layer.'

    try:
        # Initialize a config
        model_var = ModelVariants(model_choice.lower())
        model_selection, model_path, _ = get_model_info(model_var)
        config = Config(model_selection, model_path, selected_layer, instruction)
        config.model = {
            'torch_dtype': torch.float16,
            'low_cpu_mem_usage': True,
            'device_map': 'auto'
        }

        # Load the model
        model = load_model(model_var, config)

        # Handle cases where only one image is provided
        if image1 is None:
            image1 = image2
            tokens1, probs1 = [], np.array([])
            tokens2, probs2 = get_single_image_probabilities(
                instruction, image2, model, model_selection, top_k
            )
        elif image2 is None:
            image2 = image1
            tokens1, probs1 = get_single_image_probabilities(
                instruction, image1, model, model_selection, top_k
            )
            tokens2, probs2 = [], np.array([])
        else:
            tokens1, probs1 = get_single_image_probabilities(
                instruction, image1, model, model_selection, top_k
            )
            tokens2, probs2 = get_single_image_probabilities(
                instruction, image2, model, model_selection, top_k
            )

        if len(tokens1) == 0 and len(tokens2) == 0:
            return None, 'Error: Could not process the inputs. Please check the model loading.'

        # Create comparison plot
        plot = create_dual_probability_plot(
            tokens1, probs1, tokens2, probs2
        )
        scale_figure_fonts(plot, factor=1.25)

        # Create info text
        info_text = f'Model: {model_choice.upper()}\n'
        info_text += f'Layer: {selected_layer}\n'
        info_text += f'Top-K: {top_k}\n'
        info_text += f"Instruction: '{instruction}'\n\n"

        if len(tokens1) > 0:
            info_text += f"Image 1 - Top token: '{tokens1[0]}' (probability: {probs1[0]:.4f})\n"
        else:
            info_text += 'Image 1 - No data\n'

        if len(tokens2) > 0:
            info_text += f"Image 2 - Top token: '{tokens2[0]}' (probability: {probs2[0]:.4f})\n"
        else:
            info_text += 'Image 2 - No data\n'

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
        ModelVariants.LLAVA_15_7B.value,
        ModelVariants.QWENVL_20_2B.value
    ]
    return [model.capitalize() for model in implemented_models]


def create_demo() -> gr.Blocks:
    """Create and configure the Gradio demo interface for dual image comparison.

    Returns:
        Configured Gradio Blocks interface.
    """
    with gr.Blocks(title='VLM-Lens Visualizer') as demo:
        gr.Markdown("""
        # VLM-Lens Demo

        This VLM-Lens demo processes an instruction with up to two images through various Vision-Language Models (VLMs)
        and visualizes the probability distribution of the first token in the response for each image.

        **Instructions:**
        1. Select a VLM from the dropdown
        2. Select a layer from the available embedding layers
        3. Upload two images for comparison
        4. Enter your instruction/question about the images
        5. Adjust the number of top tokens to display (1-20)
        6. Click "Analyze" to see the first token probability distributions side by side

        **Note:** You can upload just one image if you prefer single image analysis.
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

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label='Number of Top Tokens to Display',
                    info='Select how many top probability tokens to show in the visualization'
                )

                with gr.Row():
                    image1_input = gr.Image(
                        label='Upload Image 1',
                        type='pil'
                    )
                    image2_input = gr.Image(
                        label='Upload Image 2',
                        type='pil'
                    )

                analyze_btn = gr.Button('Analyze', variant='primary', visible=False)

            with gr.Column():
                plot_output = gr.Plot(label='First Token Probability Distribution Comparison')
                info_output = gr.Textbox(
                    label='Analysis Info',
                    lines=8,
                    interactive=False
                )

        # Set up event handlers
        model_dropdown.change(
            fn=update_layer_choices,
            inputs=[model_dropdown],
            outputs=[layer_dropdown, analyze_btn]
        )

        analyze_btn.click(
            fn=process_dual_inputs,
            inputs=[model_dropdown, layer_dropdown, instruction_input, image1_input, image2_input, top_k_slider],
            outputs=[plot_output, info_output]
        )

        # Add examples
        gr.Examples(
            examples=[
                ['What is in this image? Describe in one word.', None, None],
                ['Describe the main object in the picture in one word.', None, None],
                ['What color is the dominant object? Describe in one word.', None, None],
            ],
            inputs=[instruction_input, image1_input, image2_input]
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
