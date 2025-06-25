# OMG-LLaVA

## Description
OMG-LLaVA is a fine-tuned multimodal vision-language model that extends the LLaVA (Large Language and Vision Assistant) architecture with object-level segmentation capabilities.

## Key Architecture Components:
**Base Architecture**: Image → Visual Processing → Language Model → Output

1. **Image Input**: Image is fed into the visual encoder
2. **OMG-Seg Visual Encoder (Frozen)**: Uses ConvNeXt-Large backbone to extract image features
3. **OMG Decoder**: Converts image features into object queries and segmentation masks
4. **Visual Projector**: Transforms visual features into visual tokens compatible with the LLM
5. **LLM**: InternLM2-Chat-7B fine-tuned using LoRA (lightweight adaptation) which is a text only model
6. **Output**: Text tokens → Direct text response

## Fine-tuning Strategy:
1. **Pretraining Stage** (`omg_llava_7b_pretrain_8gpus.py`)
2. **Fine-tuning Stage** (`omg_llava_7b_finetune_8gpus.py`)

## HuggingFace Project Analysis:
The project lacks comprehensive documentation and tutorials. The available files are:

### 1. `/internlm2-chat-7b`
A folder containing the InternLM2-Chat-7B model.

### 2. `omg_llava_7b_pretrain_8gpus.pth`
A PyTorch checkpoint containing LoRA weights from the pretraining stage.
- **Total layers**: 17
- **First 5 layers**:
  1. `llm.model.tok_embeddings.weight` | Shape: `torch.Size([92548, 4096])`
  2. `llm.output.weight` | Shape: `torch.Size([92548, 4096])`
  3. `projector.model.query_proj.weight` | Shape: `torch.Size([6144, 512])`
  4. `projector.model.query_proj.bias` | Shape: `torch.Size([6144])`
  5. `projector.model.model.0.weight` | Shape: `torch.Size([4096, 6144])`

### 3. `omg_llava_7b_finetune_8gpus.pth`
A PyTorch checkpoint containing LoRA weights from the fine-tuning stage.
- **Total layers**: 449
- **First 5 layers**:
  1. `llm.base_model.model.model.tok_embeddings.weight` | Shape: `torch.Size([92548, 4096])`
  2. `llm.base_model.model.output.base_layer.weight` | Shape: `torch.Size([92548, 4096])`
  3. `llm.base_model.model.model.layers.0.attention.wqkv.lora_A.default.weight` | Shape: `torch.Size([512, 4096])`
  4. `llm.base_model.model.model.layers.0.attention.wqkv.lora_B.default.weight` | Shape: `torch.Size([6144, 512])`
  5. `llm.base_model.model.model.layers.0.attention.wo.lora_A.default.weight` | Shape: `torch.Size([512, 4096])`

### 4. `omg_seg_convl.pth`
A PyTorch checkpoint representing the visual encoder and decoder system of OMG-LLaVA (the frozen visual component mentioned in the paper).
- **Total layers**: 310
- **First 5 layers**:
  1. `panoptic_head.pixel_decoder.input_convs.0.conv.weight` | Shape: `torch.Size([256, 1536, 1, 1])`
  2. `panoptic_head.pixel_decoder.input_convs.0.conv.bias` | Shape: `torch.Size([256])`
  3. `panoptic_head.pixel_decoder.input_convs.0.gn.weight` | Shape: `torch.Size([256])`
  4. `panoptic_head.pixel_decoder.input_convs.0.gn.bias` | Shape: `torch.Size([256])`
  5. `panoptic_head.pixel_decoder.input_convs.1.conv.weight` | Shape: `torch.Size([256, 768, 1, 1])`

### 5. `convnext_large_d_320_CocoPanopticOVDataset.pth`
A PyTorch checkpoint representing the ConvNeXt backbone layers, which are part of the OMG-Seg visual system. This appears to be a subset of the visual system checkpoint mentioned above.
- **Format**: Non-dictionary structure with unexpected format, making layer inspection difficult.

## Critical Issues:

### 1. Visual System Reconstruction
**Problem**: No clear reconstruction path for the visual system from available checkpoints.
- Extracted components show only 96 encoder parameters and 214 decoder parameters
- Missing backbone visual model integration
- Unclear relationship between `omg_seg_convl.pth` and `convnext_large_d_320_CocoPanopticOVDataset.pth`

### 2. LLM Integration
**Problem**: Inability to properly reconstruct the language model component.
- Layer count mismatch between available checkpoints and expected InternLM2-Chat-7B structure
- Paper mentions minimal fine-tuning, but checkpoint structure suggests extensive modifications
- LoRA weight integration unclear without proper base model alignment

### 3. Complete System Assembly
**Problem**: Even with individual components reconstructed, the complete OMG-LLaVA system requires sophisticated integration.
- Visual-language fusion mechanisms not documented
- Projector layer configurations unclear
- Missing architectural definitions and initialization procedures
- Would essentially require rewriting the entire model infrastructure

## Conclusion
The available checkpoints contain the trained weights but lack the architectural framework necessary for proper model reconstruction. The project requires the original codebase infrastructure to function correctly, making standalone usage extremely challenging without reverse-engineering the complete system architecture.
