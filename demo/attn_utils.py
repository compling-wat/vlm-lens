"""Functions for visualizing attention patterns in VLMs."""

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageFilter

from src.models.base import ModelBase  # noqa: E402


def build_final_attn(attn_list: List[torch.Tensor]) -> torch.Tensor:
    """Combine cached attention matrices into a full attention tensor.

    This function merges a list of attention matrices from an autoregressive
    transformer that uses a key-value (KV) cache. It reconstructs the complete
    attention matrix at the final decoding step, including both the initial
    prefix attention and the newly generated token attentions.

    Args:
        attn_list (List[torch.Tensor]):
            A list of attention tensors.
            - attn_list[0]: Tensor of shape [B, H, seq, seq] representing
              attention among the initial prefix tokens.
            - attn_list[1:]: For t = 1..n, tensors of shape [B, H, 1, seq+t]
              representing attention from the newly generated token at position
              seq+t-1 to all keys 0..(seq+t-1), including itself.

    Returns:
        torch.Tensor:
            The full attention tensor of shape [B, H, seq+n, seq+n] that
            represents the attention at the final decoding step.

    Raises:
        ValueError: If the input list is empty, has inconsistent shapes, or
            does not match the expected dimensions.
    """
    if not attn_list:
        raise ValueError('attn_list must be non-empty')

    base = attn_list[0]
    if base.ndim != 4:
        raise ValueError(f'attn_list[0] must be [B,H,seq,seq], got {base.shape}')

    B, H, seq, seq2 = base.shape
    if seq != seq2:
        raise ValueError(f'attn_list[0] last two dims must match (got {seq} vs {seq2})')

    n = len(attn_list) - 1
    if n == 0:
        # Nothing was generated; return the prefix attention as-is.
        return base

    # Validate incremental slices and infer total length (seq + n)
    for i, sl in enumerate(attn_list[1:], start=1):
        if sl.shape[:2] != (B, H) or sl.shape[2] != 1 or sl.shape[3] != (seq + i):
            raise ValueError(
                f'attn_list[{i}] must be [B,H,1,seq+{i}], got {sl.shape} '
                f'(expected last dim {seq+i})'
            )

    device = base.device
    dtype = base.dtype
    total = seq + n

    # Allocate the final attention matrix (lower-triangular; future columns stay 0)
    full = torch.zeros((B, H, total, total), device=device, dtype=dtype)

    # Top-left block: the original prefix attention
    full[:, :, :seq, :seq] = base

    # Append each new row: position r = seq + t (t from 0..n-1),
    # coming from attn_list[t+1] which has [B,H,1, seq + (t+1)].
    # This row attends to keys 0..r (inclusive). Future columns remain 0 by construction.
    for t in range(n):
        r = seq + t
        row_slice = attn_list[t + 1]  # [B,H,1, seq + t + 1]
        full[:, :, r:r+1, :r+1] = row_slice  # fill exactly the causal span

    return full


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
        - attention_tensor: Shape [num_heads, seq_len, seq_len]
        - generated_tokens: List of decoded output tokens
        - num_image_tokens: Number of image tokens in the sequence
        - img_start: Length of the input prompt tokens

    Raises:
        ValueError: If module not found or attention extraction fails.
    """
    vlm.model.to(vlm.config.device)
    vlm.model.eval()

    # Prepare inputs
    text = vlm._generate_prompt(instruction, has_images=True)
    inputs = vlm._generate_processor_output(text, image)
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(vlm.config.device)

    # Hardcoded for debug: Number of image tokens
    if 'pixel_values' in inputs:
        # Default estimate for LLaVA1.5-7B
        # Common for CLIP ViT-based encoders (e.g., 24x24 patches)
        num_image_tokens = 576
    else:
        num_image_tokens = -1

    # Find the first <image> token
    decoded_tokens = vlm.processor.tokenizer.batch_decode(
        inputs['input_ids'][0],
        skip_special_tokens=False
    )
    try:
        img_start = decoded_tokens.index('<image>')
    except ValueError:
        img_start = None
        num_image_tokens = -1

    # Try to extract layer index from module name
    layer_idx = extract_layer_index(module_name)

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
    # outputs.attentions structure for generate():
    # - Tuple of length = number of generation steps
    # - Each element is a tuple of layer attentions
    # - Each layer attention has shape [batch, num_heads, seq_len, seq_len]
    attention_tensor = None

    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        generated_len = len(outputs.attentions)
        if generated_len > 0:
            # Get attention from the first generation step

            step_attentions = outputs.attentions[0]

            if layer_idx >= 0 and layer_idx < len(step_attentions):
                # Use the specific layer
                # attention_tensor = step_attentions[layer_idx].detach()
                all_attn_frags = [outputs.attentions[i][layer_idx].detach() for i in range(generated_len)]
                attention_tensor = build_final_attn(all_attn_frags)
            else:
                # Fallback: use the last layer
                all_attn_frags = [outputs.attentions[i][-1].detach() for i in range(generated_len)]
                attention_tensor = build_final_attn(all_attn_frags)

    if attention_tensor is None:
        raise ValueError(
            f'Failed to extract attention weights. '
            f"Module '{module_name}' (layer {layer_idx}) may not be an attention layer. "
            f"Available layers: {len(step_attentions) if 'step_attentions' in locals() else 'unknown'}"
        )

    return attention_tensor, generated_tokens, num_image_tokens, img_start


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


def visualize_text_to_image_attention(
    attention_tensor: torch.Tensor,
    generated_tokens: List[str],
    num_image_tokens: int,
    img_start: int,
    num_heads_to_show: int = 8,
    colormap: Any = 'viridis'
) -> Figure:
    """Create visualization of text-to-image attention patterns.

    Args:
        attention_tensor: Attention weights [batch, num_heads, seq_len, seq_len]
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image tokens in sequence
        img_start: Starting index of image tokens in the sequence
        num_heads_to_show: Number of attention heads to display
        colormap: Matplotlib colormap name

    Returns:
        Matplotlib Figure with attention heatmaps
    """
    # Remove batch dimension if present
    if attention_tensor.dim() == 4:
        attention_tensor = attention_tensor[0]  # [num_heads, seq_len, seq_len]

    num_heads = attention_tensor.shape[0]
    num_heads_to_show = min(num_heads_to_show, num_heads)

    # Calculate grid
    cols = min(4, num_heads_to_show)
    rows = (num_heads_to_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Head indices to visualize
    if num_heads_to_show < num_heads:
        head_indices = np.linspace(0, num_heads - 1, num_heads_to_show, dtype=int)
    else:
        head_indices = list(range(num_heads_to_show))

    # Index ranges
    gen_start = attention_tensor.shape[-1] - len(generated_tokens)
    gen_end = attention_tensor.shape[-1]
    img_end = img_start + num_image_tokens

    for idx, head_idx in enumerate(head_indices):
        ax = axes[idx // cols, idx % cols]
        att = attention_tensor[head_idx].cpu().numpy()

        # FROM generated tokens (rows) → TO image tokens (cols)
        if gen_end <= att.shape[0] and img_end <= att.shape[1]:
            text_to_image = att[gen_start:gen_end, img_start:img_end]
        else:
            print(f'[WARN] Invalid slice on head {head_idx}, using full attention.')
            text_to_image = att

        im = ax.imshow(text_to_image, cmap=colormap, aspect='auto')

        ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Generated Tokens', fontsize=9)
        ax.set_ylabel('Image Tokens', fontsize=9)

        if len(generated_tokens) <= 20:
            ax.set_xticks(range(len(generated_tokens)))
            ax.set_xticklabels(generated_tokens, fontsize=8, rotation=90)
        else:
            step = max(1, len(generated_tokens) // 10)
            ax.set_xticks(range(0, len(generated_tokens), step))
            ax.set_xticklabels(generated_tokens[::step], fontsize=8, rotation=90)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
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
    img_start: int,
    aggregation: str = 'mean'
) -> Figure:
    """Create aggregated visualization showing average attention per head.

    Args:
        attention_tensor: Attention weights [batch, num_heads, seq_len, seq_len]
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image tokens in sequence
        img_start: Starting index of image tokens in the sequence
        aggregation: How to aggregate attention ('mean', 'max', 'sum')

    Returns:
        Matplotlib Figure showing aggregated attention per head

    Raises:
        ValueError: If invalid aggregation method is provided.
    """
    if attention_tensor.dim() == 4:
        attention_tensor = attention_tensor[0]

    num_heads = attention_tensor.shape[0]
    gen_start = attention_tensor.shape[-1] - len(generated_tokens)
    gen_end = attention_tensor.shape[-1]
    img_end = img_start + num_image_tokens

    head_attentions = []
    for head_idx in range(num_heads):
        head_attention = attention_tensor[head_idx].cpu().numpy()

        if img_end <= head_attention.shape[0] and gen_end <= head_attention.shape[1]:
            img_to_text = head_attention[gen_start:gen_end, img_start:img_end]
        else:
            img_to_text = head_attention

        if aggregation == 'mean':
            aggregated = img_to_text.mean()
        elif aggregation == 'max':
            aggregated = img_to_text.max()
        elif aggregation == 'sum':
            aggregated = img_to_text.sum()
        else:
            raise ValueError(f'Invalid aggregation: {aggregation}')

        head_attentions.append(aggregated)

    # Bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(num_heads), head_attentions, color='steelblue', edgecolor='navy', alpha=0.7)

    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel(f'{aggregation.capitalize()} Attention Weight', fontsize=12)
    ax.set_title(f'{aggregation.capitalize()} Image-to-Text Attention by Head',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_heads))
    ax.grid(axis='y', alpha=0.3)

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
    img_start: int,
    patch_size: Tuple[int, int] = (24, 24),
    token_index: int = -1,
    head_aggregation: str = 'mean',
    debug: bool = True
) -> np.ndarray:
    """Compute spatial attention map for overlaying on image.

    This extracts attention FROM generated tokens TO image patches.

    Args:
        attention_tensor: Attention weights [num_heads, seq_len, seq_len] or [batch, ...]
        generated_tokens: List of generated token strings
        num_image_tokens: Number of image patch tokens in sequence
        img_start: Starting index of image tokens in the sequence
        patch_size: Grid size of image patches (height, width)
        token_index: Which generated token to visualize (-1 for all tokens averaged)
        head_aggregation: How to aggregate across heads ('mean', 'max', 'sum')
        debug: Print debugging information

    Returns:
        2D numpy array [patch_h, patch_w] representing spatial attention map

    Raises:
        ValueError: If invalid head aggregation method is provided.
    """
    # Remove batch dimension if present
    if attention_tensor.dim() == 4:
        attention_tensor = attention_tensor[0]

    num_heads, seq_len, _ = attention_tensor.shape
    num_gen_tokens = len(generated_tokens)

    if debug:
        print('\n=== compute_spatial_attention_map DEBUG ===')
        print(f'Attention shape: {attention_tensor.shape}')
        print(f'Sequence length: {seq_len}')
        print(f'Generated tokens: {num_gen_tokens}')
        print(f'Image tokens: {num_image_tokens}')
        print(f'Patch grid: {patch_size}')

    # Determine sequence structure
    img_end = img_start + num_image_tokens
    gen_start = seq_len - num_gen_tokens
    gen_end = seq_len

    if debug:
        print('\nSequence positions:')
        print(f'  Image patches: {img_start} to {img_end-1}')
        print(f'  Generated tokens: {gen_start} to {gen_end-1}')

    # Validate indices
    if gen_start < 0:
        raise ValueError(
            f'Invalid sequence: {num_gen_tokens} generated tokens but sequence length is {seq_len}'
        )

    # Extract attention from image → generated
    attention_maps = []
    for head_idx in range(num_heads):
        # [seq_len, seq_len] attention for one head
        head_attention = attention_tensor[head_idx].cpu().numpy()

        # Compute indices safely
        img_end = img_start + num_image_tokens
        gen_end = gen_start + num_gen_tokens

        # Sanity checks
        if img_end > seq_len or gen_end > seq_len:
            raise ValueError(
                f'Invalid slice: seq_len={seq_len}, img_range=({img_start},{img_end}), '
                f'gen_range=({gen_start},{gen_end})'
            )

        # FROM generated tokens (rows) → TO image tokens (cols)
        text_to_image = head_attention[gen_start:gen_end, img_start:img_end]
        attention_maps.append(text_to_image)

    # Stack: [num_heads, num_image_tokens, num_gen_tokens]
    attention_maps = np.stack(attention_maps, axis=0)
    if debug:
        print(f'Number of zero elements in attention_maps: {(attention_maps == 0).sum()} out of {attention_maps.size}')

    if debug:
        print(f'\nStacked attention shape: {attention_maps.shape} [num_heads, num_image_tokens, num_gen_tokens]')

    # Select generated token(s)
    if token_index == -1:
        # mean over generated tokens → shape [heads, img]
        token_attention = attention_maps.mean(axis=1)
    else:
        token_index = min(token_index, attention_maps.shape[1] - 1)
        token_attention = attention_maps[:, token_index, :]  # [heads, img]
        if debug:
            print(f"[DEBUG] visualizing token {token_index}: '{generated_tokens[token_index]}'")

    # Aggregate across heads
    if head_aggregation == 'mean':
        spatial_attention = token_attention.mean(axis=0)
    elif head_aggregation == 'max':
        spatial_attention = token_attention.max(axis=0)
    elif head_aggregation == 'sum':
        spatial_attention = token_attention.sum(axis=0)
    else:
        raise ValueError(f'Unsupported head_aggregation: {head_aggregation}')

    # Reshape to patch grid
    grid_h, grid_w = patch_size
    expected_patches = grid_h * grid_w
    actual_patches = len(spatial_attention)

    if debug:
        print(f'\nReshaping {actual_patches} values to {grid_h}x{grid_w} = {expected_patches}')

    if actual_patches > expected_patches:
        if debug:
            print(f'WARNING: Truncating {actual_patches - expected_patches} extra patches')
        spatial_attention = spatial_attention[:expected_patches]
    elif actual_patches < expected_patches:
        if debug:
            print(f'WARNING: Padding {expected_patches - actual_patches} missing patches with zeros')
        padding = expected_patches - actual_patches
        spatial_attention = np.pad(spatial_attention, (0, padding), mode='constant')

    try:
        attention_map_2d = spatial_attention.reshape(grid_h, grid_w)
    except ValueError:
        import math
        side = int(math.sqrt(len(spatial_attention)))
        spatial_attention_trimmed = spatial_attention[:side * side]
        attention_map_2d = spatial_attention_trimmed.reshape(side, side)
        if debug:
            print(f'ERROR: Cannot reshape to {grid_h}x{grid_w}, falling back to {side}x{side}')

    return attention_map_2d


def overlay_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.6,
    colormap: str = 'jet',
    smooth: bool = False
) -> Image.Image:
    """Overlay attention heatmap on original image with proper patch alignment.

    The key fix: Use Image.NEAREST for resizing to preserve patch boundaries,
    so each patch in the attention map corresponds to the correct region in the image.

    Args:
        image: Original PIL Image
        attention_map: 2D numpy array of attention weights [patch_h, patch_w]
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        colormap: Matplotlib colormap name
        smooth: If False (default), use nearest neighbor to preserve patch boundaries.
                If True, use bilinear interpolation for smooth gradients.

    Returns:
        PIL Image with attention overlay properly aligned to patches
    """
    # Convert to RGB array
    img_array = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    img_h, img_w = img_array.shape[:2]

    # Normalize attention map to [0, 1]
    att_min, att_max = attention_map.min(), attention_map.max()
    if att_max - att_min > 1e-8:
        att_norm = (attention_map - att_min) / (att_max - att_min)
    else:
        att_norm = np.zeros_like(attention_map, dtype=np.float32)

    # Convert to PIL and resize to image resolution
    att_img = Image.fromarray((att_norm * 255).astype(np.uint8), mode='L')
    resample_mode = Image.BILINEAR if smooth else Image.NEAREST
    att_img = att_img.resize((img_w, img_h), resample=resample_mode)

    # Optional Gaussian blur for extra smoothness
    if smooth:
        patch_size = min(img_h // attention_map.shape[0],
                         img_w // attention_map.shape[1])
        blur_radius = max(1, patch_size / 2)
        att_img = att_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Convert back to normalized float array
    att_resized = np.array(att_img, dtype=np.float32) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    att_color = cmap(att_resized)[..., :3]  # RGB only

    # Blend overlay with original
    blended = (1 - alpha) * img_array + alpha * att_color
    blended_uint8 = (blended * 255).astype(np.uint8)

    return Image.fromarray(blended_uint8)


def visualize_attention_overlay_grid(
    image: Image.Image,
    attention_tensor: torch.Tensor,
    generated_tokens: List[str],
    num_image_tokens: int,
    img_start: int,
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
        img_start: Starting index of image tokens in the sequence
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
            attention_tensor, generated_tokens, num_image_tokens, img_start,
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
    img_start: int,
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
        img_start: Starting index of image tokens in the sequence
        patch_size: Grid size of image patches
        alpha: Transparency of attention overlay
        colormap: Matplotlib colormap name
        show_original: Whether to show original image alongside

    Returns:
        Matplotlib Figure with attention overlay
    """
    # Compute averaged attention map across all tokens
    attention_map = compute_spatial_attention_map(
        attention_tensor, generated_tokens, num_image_tokens, img_start,
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
