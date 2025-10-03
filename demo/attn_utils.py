"""Functions for visualizing attention patterns in VLMs."""

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from src.models.base import ModelBase  # noqa: E402


def extract_attention_weights(
    vlm: ModelBase,
    module_name: str,
    image: Image.Image,
    instruction: str,
    max_new_tokens: int = 10
) -> Tuple[torch.Tensor, List[str], int]:
    """Extract attention weights from specified module during generation.

    Args:
        vlm: The loaded VLM (ModelBase instance).
        module_name: The layer/module name to extract attention from.
        image: PIL Image to process.
        instruction: Text instruction for the model.
        max_new_tokens: Number of tokens to generate.

    Returns:
        Tuple of (attention_tensor, generated_tokens, num_image_tokens)
        - attention_tensor: Shape [num_heads, seq_len, seq_len] or similar
        - generated_tokens: List of decoded output tokens
        - num_image_tokens: Number of image tokens in the sequence

    Raises:
        ValueError: If module not found or attention extraction fails.
    """
    attention_weights = {}
    target_module = None

    def attention_hook_fn(
        module: torch.nn.Module,
        input: Any,
        output: Any
    ) -> None:
        """Hook to capture attention weights from attention layers.

        Args:
            module: The module being hooked.
            input: Input to the module.
            output: Output from the module.
        """
        # Handle different attention output formats
        if isinstance(output, tuple):
            # Many transformers return (output, attention_weights, ...)
            # Try to find attention weights in the tuple
            for item in output:
                if isinstance(item, torch.Tensor):
                    # Check if this looks like attention weights
                    # Typically shape: [batch, num_heads, seq_len, seq_len]
                    if item.dim() == 4:
                        attention_weights['weights'] = item.detach()
                        break
        # Some models store attention in output.attentions
        elif hasattr(output, 'attentions') and output.attentions is not None:
            if isinstance(output.attentions, tuple):
                attention_weights['weights'] = output.attentions[-1].detach()
            else:
                attention_weights['weights'] = output.attentions.detach()

    # Find and register hook on the target module
    for name, module in vlm.model.named_modules():
        if name == module_name:
            target_module = module
            hook_handle = module.register_forward_hook(attention_hook_fn)
            break

    if target_module is None:
        raise ValueError(f"Module '{module_name}' not found in model")

    try:
        vlm.model.eval()

        # Prepare inputs
        text = vlm._generate_prompt(instruction, has_images=True)
        inputs = vlm._generate_processor_output(text, image)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(vlm.config.device)

        # Count image tokens (approximate - depends on model architecture)
        # This is a simplified approach; adjust based on your model's specifics
        if 'pixel_values' in inputs:
            # Estimate based on vision encoder output
            num_image_tokens = 576  # Common for ViT-based encoders (24x24 patches)
        else:
            num_image_tokens = 0

        # Generate with attention output
        with torch.no_grad():
            outputs = vlm.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False
            )

        # Decode generated tokens
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_tokens = [
            vlm.processor.tokenizer.decode([token_id.item()])
            for token_id in generated_ids
        ]

        # Extract attention weights
        if 'weights' not in attention_weights:
            # Try to get from outputs.attentions if available
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # outputs.attentions is typically a tuple of tuples
                # (one tuple per generation step, each containing layer attentions)
                if len(outputs.attentions) > 0:
                    # Get attention from first generation step
                    step_attentions = outputs.attentions[0]
                    if len(step_attentions) > 0:
                        # Find the layer index from module_name
                        layer_idx = extract_layer_index(module_name)
                        if layer_idx >= 0 and layer_idx < len(step_attentions):
                            attention_weights['weights'] = step_attentions[layer_idx].detach()

        if 'weights' not in attention_weights:
            raise ValueError(f"Failed to extract attention weights from module '{module_name}'")

        attention_tensor = attention_weights['weights']

        return attention_tensor, generated_tokens, num_image_tokens

    finally:
        hook_handle.remove()


def extract_layer_index(module_name: str) -> int:
    """Extract layer index from module name.

    Args:
        module_name: Full module name like 'model.layers.15.self_attn'

    Returns:
        Layer index as integer
    """
    import re
    match = re.search(r'layers?[.\[](\d+)', module_name)
    if match:
        return int(match.group(1))
    return -1


def visualize_image_to_text_attention(
    attention_tensor: torch.Tensor,
    generated_tokens: List[str],
    num_image_tokens: int,
    num_heads_to_show: int = 8
) -> Figure:
    """Create visualization of image-to-text attention patterns.

    Args:
        attention_tensor: Attention weights [batch, num_heads, seq_len, seq_len]
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image tokens in sequence
        num_heads_to_show: Number of attention heads to display

    Returns:
        Matplotlib Figure with attention heatmaps
    """
    # Remove batch dimension if present
    if attention_tensor.dim() == 4:
        attention_tensor = attention_tensor[0]  # [num_heads, seq_len, seq_len]

    num_heads = attention_tensor.shape[0]
    num_heads_to_show = min(num_heads_to_show, num_heads)

    # Calculate grid dimensions
    cols = min(4, num_heads_to_show)
    rows = (num_heads_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Select which heads to display (evenly spaced if not showing all)
    if num_heads_to_show < num_heads:
        head_indices = np.linspace(0, num_heads - 1, num_heads_to_show, dtype=int)
    else:
        head_indices = list(range(num_heads_to_show))

    for idx, head_idx in enumerate(head_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Get attention for this head
        head_attention = attention_tensor[head_idx].cpu().numpy()

        # Extract image-to-text attention
        # Assuming image tokens come first in the sequence
        num_output_tokens = len(generated_tokens)

        # Get attention from output tokens to image tokens
        # Shape: [num_output_tokens, num_image_tokens]
        if head_attention.shape[0] > num_output_tokens and num_image_tokens > 0:
            # Take the last num_output_tokens rows (generated tokens)
            # and first num_image_tokens columns (image tokens)
            img_to_text_attention = head_attention[-num_output_tokens:, :num_image_tokens]
        else:
            # Fallback: show full attention pattern
            img_to_text_attention = head_attention

        # Create heatmap
        im = ax.imshow(img_to_text_attention, cmap='viridis', aspect='auto')

        # Set labels
        ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Image Tokens', fontsize=9)
        ax.set_ylabel('Generated Tokens', fontsize=9)

        # Set y-axis labels to show generated tokens
        if len(generated_tokens) <= 20:
            ax.set_yticks(range(len(generated_tokens)))
            ax.set_yticklabels(generated_tokens, fontsize=8)
        else:
            # Show fewer labels if too many tokens
            step = len(generated_tokens) // 10
            ax.set_yticks(range(0, len(generated_tokens), step))
            ax.set_yticklabels(generated_tokens[::step], fontsize=8)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide empty subplots
    for idx in range(num_heads_to_show, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    fig.suptitle('Image-to-Text Attention Patterns by Head',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()

    return fig


def visualize_attention_head_grid(
    attention_tensor: torch.Tensor,
    generated_tokens: List[str],
    num_image_tokens: int,
    aggregation: str = 'mean'
) -> Figure:
    """Create aggregated visualization showing average attention per head.

    Args:
        attention_tensor: Attention weights [batch, num_heads, seq_len, seq_len]
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image tokens in sequence
        aggregation: How to aggregate attention ('mean', 'max', 'sum')

    Returns:
        Matplotlib Figure showing aggregated attention per head
    """
    if attention_tensor.dim() == 4:
        attention_tensor = attention_tensor[0]

    num_heads = attention_tensor.shape[0]
    num_output_tokens = len(generated_tokens)

    # Extract image-to-text attention for each head
    head_attentions = []
    for head_idx in range(num_heads):
        head_attention = attention_tensor[head_idx].cpu().numpy()

        if head_attention.shape[0] > num_output_tokens and num_image_tokens > 0:
            img_to_text = head_attention[-num_output_tokens:, :num_image_tokens]
        else:
            img_to_text = head_attention

        # Aggregate attention values
        if aggregation == 'mean':
            aggregated = img_to_text.mean()
        elif aggregation == 'max':
            aggregated = img_to_text.max()
        elif aggregation == 'sum':
            aggregated = img_to_text.sum()
        else:
            aggregated = img_to_text.mean()

        head_attentions.append(aggregated)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(num_heads), head_attentions, color='steelblue', edgecolor='navy', alpha=0.7)

    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel(f'{aggregation.capitalize()} Attention Weight', fontsize=12)
    ax.set_title(f'{aggregation.capitalize()} Image-to-Text Attention by Head',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_heads))
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, head_attentions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()

    return fig
