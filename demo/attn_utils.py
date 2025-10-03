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


def compute_spatial_attention_map(
    attention_tensor: torch.Tensor,
    generated_tokens: List[str],
    num_image_tokens: int,
    patch_size: Tuple[int, int] = (24, 24),
    token_index: int = -1,
    head_aggregation: str = 'mean'
) -> np.ndarray:
    """Compute spatial attention map for overlaying on image.

    Args:
        attention_tensor: Attention weights [num_heads, seq_len, seq_len]
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image tokens in sequence
        patch_size: Grid size of image patches (height, width)
        token_index: Which output token to visualize (-1 for all tokens averaged)
        head_aggregation: How to aggregate across heads ('mean', 'max', 'sum')

    Returns:
        2D numpy array representing spatial attention map
    """
    if attention_tensor.dim() == 4:
        attention_tensor = attention_tensor[0]

    num_heads = attention_tensor.shape[0]
    num_output_tokens = len(generated_tokens)

    # Extract image-to-text attention
    attention_maps = []
    for head_idx in range(num_heads):
        head_attention = attention_tensor[head_idx].cpu().numpy()

        if head_attention.shape[0] > num_output_tokens and num_image_tokens > 0:
            img_to_text = head_attention[-num_output_tokens:, :num_image_tokens]
        else:
            img_to_text = head_attention[:num_output_tokens, :num_image_tokens]

        attention_maps.append(img_to_text)

    # Stack attention maps: [num_heads, num_output_tokens, num_image_tokens]
    attention_maps = np.stack(attention_maps, axis=0)

    # Select token(s) to visualize
    if token_index == -1:
        # Average across all output tokens
        token_attention = attention_maps.mean(axis=1)  # [num_heads, num_image_tokens]
    else:
        token_index = min(token_index, attention_maps.shape[1] - 1)
        token_attention = attention_maps[:, token_index, :]  # [num_heads, num_image_tokens]

    # Aggregate across heads
    if head_aggregation == 'mean':
        spatial_attention = token_attention.mean(axis=0)
    elif head_aggregation == 'max':
        spatial_attention = token_attention.max(axis=0)
    elif head_aggregation == 'sum':
        spatial_attention = token_attention.sum(axis=0)
    else:
        spatial_attention = token_attention.mean(axis=0)

    # Reshape to 2D spatial grid
    # Handle case where num_image_tokens might include CLS token
    grid_h, grid_w = patch_size
    expected_tokens = grid_h * grid_w

    if len(spatial_attention) == expected_tokens + 1:
        # Remove CLS token (usually the first token)
        spatial_attention = spatial_attention[1:]
    elif len(spatial_attention) > expected_tokens:
        # Truncate to expected size
        spatial_attention = spatial_attention[:expected_tokens]
    elif len(spatial_attention) < expected_tokens:
        # Pad if necessary
        padding = expected_tokens - len(spatial_attention)
        spatial_attention = np.pad(spatial_attention, (0, padding), mode='constant')

    # Reshape to 2D grid
    attention_map_2d = spatial_attention.reshape(grid_h, grid_w)

    return attention_map_2d


def overlay_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.6,
    colormap: str = 'jet'
) -> Image.Image:
    """Overlay attention heatmap on original image.

    Args:
        image: Original PIL Image
        attention_map: 2D numpy array of attention weights
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        colormap: Matplotlib colormap name

    Returns:
        PIL Image with attention overlay
    """
    # Resize attention map to image size
    img_array = np.array(image.convert('RGB'))
    h, w = img_array.shape[:2]

    # Ensure attention_map is float64 for scipy zoom
    attention_map = attention_map.astype(np.float64)

    # Upsample attention map to image resolution using PIL for better compatibility
    # Create PIL Image from attention map for resizing
    attention_normalized_temp = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    attention_pil = Image.fromarray((attention_normalized_temp * 255).astype(np.uint8))
    attention_resized = attention_pil.resize((w, h), Image.BILINEAR)
    attention_upsampled = np.array(attention_resized).astype(np.float32) / 255.0

    # Normalize attention map
    attention_normalized = (attention_upsampled - attention_upsampled.min()) / (
        attention_upsampled.max() - attention_upsampled.min() + 1e-8
    )

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    attention_colored = cmap(attention_normalized)[:, :, :3]  # RGB only
    attention_colored = (attention_colored * 255).astype(np.uint8)

    # Blend with original image
    blended = (alpha * attention_colored + (1 - alpha) * img_array).astype(np.uint8)

    return Image.fromarray(blended)


def visualize_attention_overlay_grid(
    image: Image.Image,
    attention_tensor: torch.Tensor,
    generated_tokens: List[str],
    num_image_tokens: int,
    patch_size: Tuple[int, int] = (24, 24),
    num_tokens_to_show: int = 4,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> Figure:
    """Create grid showing attention overlay for multiple generated tokens.

    Args:
        image: Original PIL Image
        attention_tensor: Attention weights [num_heads, seq_len, seq_len]
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image tokens in sequence
        patch_size: Grid size of image patches
        num_tokens_to_show: Number of generated tokens to visualize
        alpha: Transparency of attention overlay
        colormap: Matplotlib colormap name

    Returns:
        Matplotlib Figure with attention overlays
    """
    num_output_tokens = len(generated_tokens)
    num_tokens_to_show = min(num_tokens_to_show, num_output_tokens)

    # Calculate grid dimensions
    cols = min(3, num_tokens_to_show)
    rows = (num_tokens_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Select token indices to display (evenly spaced)
    if num_tokens_to_show < num_output_tokens:
        token_indices = np.linspace(0, num_output_tokens - 1, num_tokens_to_show, dtype=int)
    else:
        token_indices = list(range(num_tokens_to_show))

    for idx, token_idx in enumerate(token_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Compute attention map for this token
        attention_map = compute_spatial_attention_map(
            attention_tensor, generated_tokens, num_image_tokens,
            patch_size, token_idx, head_aggregation='mean'
        )

        # Create overlay
        overlay_img = overlay_attention_on_image(image, attention_map, alpha, colormap)

        # Display
        ax.imshow(overlay_img)
        ax.set_title(f'Token {token_idx}: "{generated_tokens[token_idx]}"',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    # Hide empty subplots
    for idx in range(num_tokens_to_show, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    fig.suptitle('Attention Overlay on Image by Generated Token',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout()

    return fig


def visualize_attention_overlay_averaged(
    image: Image.Image,
    attention_tensor: torch.Tensor,
    generated_tokens: List[str],
    num_image_tokens: int,
    patch_size: Tuple[int, int] = (24, 24),
    alpha: float = 0.5,
    colormap: str = 'jet',
    show_original: bool = True
) -> Figure:
    """Create side-by-side comparison with averaged attention overlay.

    Args:
        image: Original PIL Image
        attention_tensor: Attention weights
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image tokens in sequence
        patch_size: Grid size of image patches
        alpha: Transparency of attention overlay
        colormap: Matplotlib colormap name
        show_original: Whether to show original image alongside

    Returns:
        Matplotlib Figure with attention overlay
    """
    # Compute averaged attention map across all tokens
    attention_map = compute_spatial_attention_map(
        attention_tensor, generated_tokens, num_image_tokens,
        patch_size, token_index=-1, head_aggregation='mean'
    )

    # Create overlay
    overlay_img = overlay_attention_on_image(image, attention_map, alpha, colormap)

    # Create figure
    if show_original:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(overlay_img)
        ax2.set_title('Attention Overlay (Averaged)', fontsize=12, fontweight='bold')
        ax2.axis('off')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(overlay_img)
        ax.set_title('Attention Overlay (Averaged)', fontsize=12, fontweight='bold')
        ax.axis('off')

    full_response = ''.join(generated_tokens)
    fig.suptitle(f'Generated: "{full_response}"',
                 fontsize=11, style='italic', y=0.02)
    fig.tight_layout()

    return fig
