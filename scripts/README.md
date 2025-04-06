# Helper Scripts
## read_tensor.py
Using the same `Config` object as before, we are trying to query for the specific prompts and embeddings saved from previously running the VLM extractor. In particular, by running:
```
python scripts/read_tensor.py --config configs/clip-base.yaml
```
one can read from the database for the specific configuration, including filtering on its model path, architecture, layers, prompts and image input.
