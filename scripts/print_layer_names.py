"""print_layer_names.py.

A utility for inspecting PyTorch model checkpoints hosted on Hugging Face Hub.
This tool downloads and analyzes .pth checkpoint files to display layer names,
parameter counts, and tensor shapes - useful for model architecture exploration,
debugging, and understanding fine-tuned model structures.

Usage:
    python print_layer_names.py -project:<HF_REPO_ID> -pth:<CHECKPOINT_FILE>

Arguments:
    -project, --project     Hugging Face repository ID (required)
    -pth, --pth            Checkpoint filename within the repository (required)
    --limit                Maximum layers to display (default: 10, 0 for all)
    --show-shapes          Include tensor shapes in output

Examples:
    # Basic usage - show first 10 layers
    python print_layer_names.py -project zhangtao-whu/OMG-LLaVA -pth finetuned_gcg.pth

    # Show all layers with shapes
    python print_layer_names.py --project zhangtao-whu/OMG-LLaVA -pth omg_llava_7b_finetune_8gpus.pth --limit 0 --show-shapes

    # Inspect specific fine-tuned checkpoint
    python print_layer_names.py -project zhangtao-whu/OMG-LLaVA -pth omg_llava_7b_finetune_8gpus.pth --limit 20
"""
import argparse
import sys

import torch
from huggingface_hub import HfApi, hf_hub_download


def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '-project', '--project',
        required=True,
        help='Hugging Face project/repository ID'
    )

    parser.add_argument(
        '-pth', '--pth',
        required=True,
        help='Path to the .pth file within the Hugging Face project'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum number of layers to display (default: 10, use 0 for all)'
    )

    parser.add_argument(
        '--show-shapes',
        action='store_true',
        help='Display tensor shapes alongside layer names'
    )

    return parser.parse_args()


def validate_repository(repo_id):
    """Check if the project is valid and accessible."""
    try:
        api = HfApi()
        api.repo_info(repo_id=repo_id)
        return True
    except Exception as e:
        print(f"Error: Cannot access repository '{repo_id}': {e}", file=sys.stderr)
        return False


def load_checkpoint(repo_id, filename):
    """Download and load the checkpoint file."""
    try:
        print(f'Downloading {filename} from {repo_id}...')
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename
        )

        print(f'Loading checkpoint from {checkpoint_path}...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        return checkpoint

    except Exception as e:
        print(f'Error loading checkpoint: {e}', file=sys.stderr)
        return None


def analyze_checkpoint(checkpoint, limit=10, show_shapes=True):
    """Display checkpoint structure."""
    if checkpoint is None:
        return

    print(f'\nCheckpoint type: {type(checkpoint)}')

    if isinstance(checkpoint, dict):
        print(f'Top-level keys: {list(checkpoint.keys())}')

        state_dict = None
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Found 'state_dict' key")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' key")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Found 'model' key")
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
            print('Treating checkpoint as direct state dict')

        if state_dict and isinstance(state_dict, dict):
            display_layers(state_dict, limit, show_shapes)
        else:
            print('No valid state dictionary found in checkpoint')
    else:
        print('Checkpoint is not a dictionary - unexpected format')


def display_layers(state_dict, limit=10, show_shapes=True):
    """Display layer names and optionally their shapes."""
    total_layers = len(state_dict)
    print(f'\nTotal number of layers/parameters: {total_layers}')

    if limit == 0:
        limit = total_layers

    display_count = min(limit, total_layers)
    print(f'\nDisplaying first {display_count} layers:')
    print('-' * 60)

    for idx, (key, value) in enumerate(state_dict.items()):
        if idx >= limit:
            remaining = total_layers - limit
            if remaining > 0:
                print(f'... and {remaining} more layers')
            break

        if show_shapes and hasattr(value, 'shape'):
            print(f'{idx + 1:3d}. {key:<40} | Shape: {value.shape}')
        else:
            print(f'{idx + 1:3d}. {key}')


if __name__ == '__main__':
    args = parse_arguments()

    # Check if the project is valid and accessible
    if not validate_repository(args.project):
        sys.exit(1)

    # See if the checkpoint exist
    checkpoint = load_checkpoint(args.project, args.pth)
    if checkpoint is None:
        sys.exit(1)

    # Analyze and display the layer names
    analyze_checkpoint(checkpoint, args.limit, args.show_shapes)

    print('\nDone!')
