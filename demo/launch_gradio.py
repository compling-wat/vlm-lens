"""Gradio demo for visualizing VLM first token probability distributions with two images."""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from demo.attn_utils import extract_attention_weights  # noqa: E402
from demo.attn_utils import (visualize_attention_head_grid,
                             visualize_image_to_text_attention)
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
                label=f'Select Module for {model_choice}',
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
                label='Select Module',
                visible=True,
                interactive=False
            ),
            gr.Button('Analyze', variant='primary', visible=False)
        )
    except Exception as e:
        return (
            gr.Dropdown(
                choices=[f'Error: {str(e)}'],
                label='Select Module',
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
    """
    # Generate prompt and process inputs
    vlm.model.eval()
    text = vlm._generate_prompt(instruction, has_images=True)
    inputs = vlm._generate_processor_output(text, image)
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(vlm.config.device)

    with torch.no_grad():
        outputs = vlm.model.generate(
            **inputs,
            max_new_tokens=1,  # Only generate first token
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False
        )

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


def get_module_similarity_pooled(
        vlm: ModelBase,
        module_name: str,
        image1: Image.Image,
        image2: Image.Image,
        instruction: str,
        pooling: str = 'mean'
) -> float:
    """Compute cosine similarity with optional pooling strategies.

    Args:
        vlm: The loaded VLM (ModelBase instance).
        module_name: The layer/module name to extract features from.
        image1: First PIL Image.
        image2: Second PIL Image.
        instruction: Text instruction for the model.
        pooling: Pooling strategy - 'mean', 'max', 'cls', or 'none'.

    Returns:
        Cosine similarity value between the two embeddings.

    Raises:
        ValueError: If feature extraction fails or module not found.
    """
    embeddings = {}
    target_module = None

    def hook_fn(
        module: torch.nn.Module,
        input: Any,
        output: Any
    ) -> None:
        """Forward hook to capture module output.

        Args:
            module: The module being hooked.
            input: The input to the module.
            output: The output from the module.
        """
        if isinstance(output, tuple):
            embeddings['activation'] = output[0].detach()
        else:
            embeddings['activation'] = output.detach()

    # Find and register hook
    for name, module in vlm.model.named_modules():
        if name == module_name:
            target_module = module
            hook_handle = module.register_forward_hook(hook_fn)
            break

    if target_module is None:
        raise ValueError(f"Module '{module_name}' not found in model")

    try:
        vlm.model.eval()
        # Extract embedding for image1
        text = vlm._generate_prompt(instruction, has_images=True)
        inputs1 = vlm._generate_processor_output(text, image1)
        for key in inputs1:
            if isinstance(inputs1[key], torch.Tensor):
                inputs1[key] = inputs1[key].to(vlm.config.device)

        embeddings.clear()
        with torch.no_grad():
            _ = vlm.model(**inputs1)

        if 'activation' not in embeddings:
            raise ValueError('Failed to extract features for image1')

        embedding1 = embeddings['activation']

        # Extract embedding for image2
        inputs2 = vlm._generate_processor_output(text, image2)
        for key in inputs2:
            if isinstance(inputs2[key], torch.Tensor):
                inputs2[key] = inputs2[key].to(vlm.config.device)

        embeddings.clear()
        with torch.no_grad():
            _ = vlm.model(**inputs2)

        if 'activation' not in embeddings:
            raise ValueError('Failed to extract features for image2')

        embedding2 = embeddings['activation']

        # Apply pooling strategy
        if pooling == 'mean':
            # Mean pooling across sequence dimension
            if embedding1.dim() >= 2:
                embedding1_pooled = embedding1.mean(dim=1)
                embedding2_pooled = embedding2.mean(dim=1)
            else:
                embedding1_pooled = embedding1
                embedding2_pooled = embedding2

        elif pooling == 'max':
            # Max pooling across sequence dimension
            if embedding1.dim() >= 2:
                embedding1_pooled = embedding1.max(dim=1)[0]
                embedding2_pooled = embedding2.max(dim=1)[0]
            else:
                embedding1_pooled = embedding1
                embedding2_pooled = embedding2

        elif pooling == 'cls':
            # Use first token (CLS token)
            if embedding1.dim() >= 2:
                embedding1_pooled = embedding1[:, 0, :]
                embedding2_pooled = embedding2[:, 0, :]
            else:
                embedding1_pooled = embedding1
                embedding2_pooled = embedding2

        elif pooling == 'none':
            # Flatten without pooling
            embedding1_pooled = embedding1.reshape(embedding1.shape[0], -1)
            embedding2_pooled = embedding2.reshape(embedding2.shape[0], -1)
        else:
            raise ValueError(f'Unknown pooling strategy: {pooling}')

        # Ensure 2D shape [batch, features]
        if embedding1_pooled.dim() == 1:
            embedding1_pooled = embedding1_pooled.unsqueeze(0)
            embedding2_pooled = embedding2_pooled.unsqueeze(0)

        # Compute cosine similarity
        similarity = F.cosine_similarity(embedding1_pooled, embedding2_pooled, dim=1)
        similarity_value = float(similarity.mean().cpu().item())

        return similarity_value

    finally:
        hook_handle.remove()


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

        if len(tokens1) > 0 and len(tokens2) > 0:
            info_text += f'\nLayer: {selected_layer}\n'
            similarity = get_module_similarity_pooled(model, selected_layer, image1, image2, instruction)
            info_text += f'Cosine similarity between Image 1 and 2: {similarity:.3f}\n'

        return plot, info_text

    except ValueError as e:
        return None, f'Invalid model selection: {str(e)}'
    except Exception as e:
        return None, f'Error: {str(e)}'


def process_with_attention(
    model_choice: str,
    selected_layer: str,
    instruction: str,
    image: Optional[Image.Image],
    max_tokens: int = 10,
    num_heads_display: int = 8,
    aggregation_type: str = 'mean'
) -> Tuple[Optional[Figure], Optional[Figure], str]:
    """Process single image and return attention visualizations.

    Args:
        model_choice: String name of the selected model.
        selected_layer: String name of the selected layer.
        instruction: Text instruction for the model.
        image: PIL Image to process.
        max_tokens: Number of tokens to generate.
        num_heads_display: Number of attention heads to show in detail.
        aggregation_type: Aggregation method for head comparison.

    Returns:
        Tuple containing (attention_heatmap, aggregation_plot, info_text)
    """
    if image is None:
        return None, None, 'Please upload an image.'

    if not instruction.strip():
        return None, None, 'Please provide an instruction.'

    if not model_choice or not selected_layer:
        return None, None, 'Please select both model and layer.'

    try:
        # Initialize config
        model_var = ModelVariants(model_choice.lower())
        model_selection, model_path, _ = get_model_info(model_var)
        config = Config(model_selection, model_path, selected_layer, instruction)
        config.model = {
            'torch_dtype': torch.float16,
            'low_cpu_mem_usage': True,
            'device_map': 'auto'
        }

        # Load model
        model = load_model(model_var, config)

        # Extract attention weights
        attention_tensor, generated_tokens, num_image_tokens = extract_attention_weights(
            model, selected_layer, image, instruction, max_tokens
        )

        # Create visualizations
        attention_fig = visualize_image_to_text_attention(
            attention_tensor, generated_tokens, num_image_tokens, num_heads_display
        )
        scale_figure_fonts(attention_fig, factor=1.15)

        aggregation_fig = visualize_attention_head_grid(
            attention_tensor, generated_tokens, num_image_tokens, aggregation_type
        )
        scale_figure_fonts(aggregation_fig, factor=1.15)

        # Create info text
        full_response = ''.join(generated_tokens)
        info_text = f'Model: {model_choice.upper()}\n'
        info_text += f'Layer: {selected_layer}\n'
        info_text += f"Instruction: '{instruction}'\n\n"
        info_text += f'Generated Response: "{full_response}"\n\n'
        info_text += f'Number of Attention Heads: {attention_tensor.shape[0] if attention_tensor.dim() >= 3 else "N/A"}\n'
        info_text += f'Number of Image Tokens: {num_image_tokens}\n'
        info_text += f'Number of Generated Tokens: {len(generated_tokens)}\n'
        info_text += f'Attention Tensor Shape: {list(attention_tensor.shape)}\n'

        return attention_fig, aggregation_fig, info_text

    except ValueError as e:
        return None, None, f'Invalid selection: {str(e)}'
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, None, f'Error during attention extraction:\n{str(e)}\n\nDetails:\n{error_details}'


def create_demos() -> gr.Blocks:
    """Create enhanced Gradio demo with attention visualization tab.

    Returns:
        Configured Gradio Blocks interface with multiple tabs.
    """
    with gr.Blocks(title='VLM-Lens Visualizer') as demo:
        gr.Markdown("""
        # VLM-Lens

        From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models with VLM-Lens

        - 📄 [arXiv](https://arxiv.org/abs/2510.02292)
        - 🧑‍💻 [GitHub](https://github.com/compling-wat/vlm-lens)
        - 📅 [EMNLP 2025 System Demonstration](https://2025.emnlp.org)
        """)

        with gr.Tabs():

            # Tab 1: Dual image comparison
            with gr.TabItem('Token Distribution and Embedding Similairity'):
                gr.Markdown("""
                This demo processes an instruction with up to two images through various VLMs,
                computes cosine similarity between their embeddings at a specified layer,
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
                        model_dropdown1 = gr.Dropdown(
                            choices=[v.value.capitalize() for v in ModelVariants],
                            label='Select VLM',
                            value=None,
                            interactive=True
                        )

                        layer_dropdown1 = gr.Dropdown(
                            choices=[],
                            label='Select Module',
                            visible=False,
                            interactive=True
                        )

                        instruction_input1 = gr.Textbox(
                            label='Instruction',
                            placeholder='Describe what you see in this image...',
                            lines=3
                        )

                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=8,
                            step=1,
                            label='Number of Top Tokens to Display'
                        )

                        with gr.Row():
                            image1_input = gr.Image(label='Upload Image 1', type='pil')
                            image2_input = gr.Image(label='Upload Image 2', type='pil')

                        analyze_btn1 = gr.Button('Analyze', variant='primary', visible=False)

                    with gr.Column():
                        plot_output1 = gr.Plot(label='Probability Distribution Comparison')
                        info_output1 = gr.Textbox(
                            label='Analysis Info',
                            lines=8,
                            interactive=False
                        )

                model_dropdown1.change(
                    fn=update_layer_choices,
                    inputs=[model_dropdown1],
                    outputs=[layer_dropdown1, analyze_btn1]
                )

                analyze_btn1.click(
                    fn=process_dual_inputs,
                    inputs=[model_dropdown1, layer_dropdown1, instruction_input1,
                            image1_input, image2_input, top_k_slider],
                    outputs=[plot_output1, info_output1]
                )

            # Tab 2: Attention visualization
            with gr.TabItem('Attention Visualization'):
                gr.Markdown("""
                Visualize attention patterns from image tokens to generated text tokens.

                **Instructions:**
                1. Select a VLM and attention layer
                2. Upload an image
                3. Enter your instruction
                4. Configure visualization parameters
                5. Click "Visualize Attention" to see attention patterns
                """)

                with gr.Row():
                    with gr.Column():
                        model_dropdown2 = gr.Dropdown(
                            choices=[v.value.capitalize() for v in ModelVariants],
                            label='Select VLM',
                            value=None,
                            interactive=True
                        )

                        layer_dropdown2 = gr.Dropdown(
                            choices=[],
                            label='Select Attention Module',
                            visible=False,
                            interactive=True
                        )

                        instruction_input2 = gr.Textbox(
                            label='Instruction',
                            placeholder='What is in this image?',
                            lines=2
                        )

                        image_input2 = gr.Image(
                            label='Upload Image',
                            type='pil'
                        )

                        with gr.Row():
                            max_tokens_slider = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label='Max Tokens to Generate',
                                info='Number of tokens to generate for attention analysis'
                            )

                        with gr.Row():
                            num_heads_slider = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=8,
                                step=1,
                                label='Heads to Display',
                                info='Number of attention heads to show in detail view'
                            )

                        aggregation_dropdown = gr.Dropdown(
                            choices=['mean', 'max', 'sum'],
                            value='mean',
                            label='Aggregation Method',
                            info='How to aggregate attention for head comparison'
                        )

                        analyze_btn2 = gr.Button(
                            'Visualize Attention',
                            variant='primary',
                            visible=False
                        )

                    with gr.Column():
                        attention_plot = gr.Plot(
                            label='Attention Patterns by Head'
                        )
                        aggregation_plot = gr.Plot(
                            label='Attention Aggregation Across Heads'
                        )
                        info_output2 = gr.Textbox(
                            label='Attention Analysis Info',
                            lines=10,
                            interactive=False
                        )

                model_dropdown2.change(
                    fn=update_layer_choices,
                    inputs=[model_dropdown2],
                    outputs=[layer_dropdown2, analyze_btn2]
                )

                analyze_btn2.click(
                    fn=process_with_attention,
                    inputs=[model_dropdown2, layer_dropdown2, instruction_input2,
                            image_input2, max_tokens_slider, num_heads_slider,
                            aggregation_dropdown],
                    outputs=[attention_plot, aggregation_plot, info_output2]
                )

                gr.Examples(
                    examples=[
                        ['What is the main object? Answer in one word.'],
                        ['Describe the scene briefly.'],
                        ['What color is prominent in this image?'],
                    ],
                    inputs=[instruction_input2]
                )

    return demo


if __name__ == '__main__':
    # Create and launch the demo
    demo = create_demos()
    demo.launch(
        share=True,
        server_name='0.0.0.0',
        server_port=7860
    )
