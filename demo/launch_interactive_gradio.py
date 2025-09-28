"""Gradio demo for visualizing LLaVA first token probability distributions."""

from typing import List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

# Global variables to store model and processor
model: Optional[LlavaNextForConditionalGeneration] = None
processor: Optional[LlavaNextProcessor] = None


def load_model() -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """Load the LLaVA model and processor.

    Returns:
        Tuple containing the loaded model and processor.
    """
    global model, processor

    if model is None or processor is None:
        print('Loading LLaVA-7B model...')
        model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'

        # Load processor and model
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        print('Model loaded successfully!')

    return model, processor


def get_first_token_probabilities(
    instruction: str,
    image: Image.Image
) -> Tuple[List[str], np.ndarray]:
    """Process the instruction and image through LLaVA and return first token probabilities.

    Args:
        instruction: Text instruction for the model.
        image: PIL Image to process.

    Returns:
        Tuple containing list of top tokens and their probabilities.
    """
    try:
        # Load model if not already loaded
        model, processor = load_model()

        # Prepare the conversation format
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image'},
                ],
            },
        ]

        # Apply chat template and process inputs
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)

        # Generate with output_scores=True to get logits
        with torch.no_grad():
            outputs = model.generate(
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
        top_k = 20
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Convert tokens back to text
        top_tokens = [processor.tokenizer.decode([idx.item()]) for idx in top_indices]

        return top_tokens, top_probs.cpu().numpy()

    except Exception as e:
        print(f'Error in processing: {str(e)}')
        return [], np.array([])


def create_probability_plot(tokens: List[str], probabilities: np.ndarray) -> Figure:
    """Create a matplotlib plot of token probabilities.

    Args:
        tokens: List of token strings.
        probabilities: Array of probability values.

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
    ax.set_title('First Token Probability Distribution (LLaVA 7B)',
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


def process_inputs(instruction: str, image: Optional[Image.Image]) -> Tuple[Optional[Figure], str]:
    """Main function to process inputs and return plot.

    Args:
        instruction: Text instruction for the model.
        image: PIL Image to process, can be None.

    Returns:
        Tuple containing the plot figure and info text.
    """
    if image is None:
        return None, 'Please upload an image.'

    if not instruction.strip():
        return None, 'Please provide an instruction.'

    try:
        # Get token probabilities
        tokens, probabilities = get_first_token_probabilities(instruction, image)

        if len(tokens) == 0:
            return None, 'Error: Could not process the inputs. Please check the model loading.'

        # Create plot
        plot = create_probability_plot(tokens, probabilities)

        # Create info text
        info_text = f"Processed instruction: \'{instruction}\'\n"
        info_text += f"Top token: \'{tokens[0]}\' (probability: {probabilities[0]:.4f})"

        return plot, info_text

    except Exception as e:
        return None, f'Error: {str(e)}'


def create_demo() -> gr.Blocks:
    """Create and configure the Gradio demo interface.

    Returns:
        Configured Gradio Blocks interface.
    """
    with gr.Blocks(title='LLaVA Token Probability Visualizer') as demo:
        gr.Markdown("""
        # LLaVA First Token Probability Distribution

        This demo processes an instruction and image through LLaVA-7B and visualizes the probability distribution
        of the first token in the response.

        **Instructions:**
        1. Upload an image
        2. Enter your instruction/question about the image
        3. Click "Analyze" to see the first token probability distribution
        """)

        with gr.Row():
            with gr.Column():
                instruction_input = gr.Textbox(
                    label='Instruction',
                    placeholder='Describe what you see in this image...',
                    lines=3
                )
                image_input = gr.Image(
                    label='Upload Image',
                    type='pil'
                )
                analyze_btn = gr.Button('Analyze', variant='primary')

            with gr.Column():
                plot_output = gr.Plot(label='First Token Probability Distribution')
                info_output = gr.Textbox(
                    label='Analysis Info',
                    lines=3,
                    interactive=False
                )

        # Set up the event handler
        analyze_btn.click(
            fn=process_inputs,
            inputs=[instruction_input, image_input],
            outputs=[plot_output, info_output]
        )

        # Add examples
        gr.Examples(
            examples=[
                ['What is in this image? Describe in one word.', None],
                ['Describe the main object in the picture in one word..', None],
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
