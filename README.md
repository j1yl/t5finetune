# Fine tuning T5 small for headline generation

## Dataset & model
- Model: `t5-small`
- Dataset: `Gigaword 10k`
- Task: `Generate a short headline from a full sentence or paragraph.`
- [https://huggingface.co/google-t5/t5-small](https://huggingface.co/google-t5/t5-small)
- [https://huggingface.co/datasets/anumafzal94/gigaword_10k_finetuning](https://huggingface.co/datasets/anumafzal94/gigaword_10k_finetuning)


## Usage

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

Parameters
- `model`
- `output_dir`
- `batch_size`
- `epochs`
- `lr`

### Evaluation

```bash
python eval.py
```

Parameters
- `results_path`
- `output_dir`
- `model_dir`

### Note
Check implementation in `train.py` and `eval.py` for default parameter values



## Project structure
- `train.py` : training script and pipeline
- `dataset.py` : class for loading dataset
- `eval.py`: evaluation script

## Training configuration
- Batch size: `32`
- LR: `3e-4`
- GPU: `NVIDIA RTX 3090 24GB`