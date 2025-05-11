# Fine tuning T5 small for headline generation

## Dataset & model
- Model: `t5-small`
- Dataset: `Gigaword 10k`
- Task: `Generate a short headline from a full sentence or paragraph.`
- [https://huggingface.co/google-t5/t5-small](https://huggingface.co/google-t5/t5-small)
- [https://huggingface.co/datasets/anumafzal94/gigaword_10k_finetuning](https://huggingface.co/datasets/anumafzal94/gigaword_10k_finetuning)

## Results
We compared different fine-tuning approaches:

1. **Full Fine-tuning**
   - Best overall performance
   - BLEU: `18.75`
   - ROUGE-1: `0.44`
   - ROUGE-2: `0.22`
   - ROUGE-L: `0.42`
   - Training time: ~`549`s

2. **LoRA Fine-tuning**
   - Good performance with minimal parameters
   - Best configuration (r=`4`):
     - BLEU: `13.43`
     - ROUGE-1: `0.39`
     - ROUGE-2: `0.17`
     - ROUGE-L: `0.37`
   - Training time: ~`426`-`462`s

3. **Adapter Fine-tuning**
   - Lower performance
   - BLEU: `5.02`
   - Training time: ~`432`s

4. **Base Model (No fine-tuning)**
   - BLEU: `4.85`
   - ROUGE-1: `0.30`
   - ROUGE-2: `0.10`
   - ROUGE-L: `0.27`

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

Parameters:
- `model`: Model to fine-tune
- `output_dir`: Directory to save results
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `lr`: Learning rate

### Evaluation
```bash
python eval.py
```

Parameters:
- `results_path`: Path to save evaluation results
- `output_dir`: Directory to save outputs
- `model_dir`: Directory containing the model

### Inference
```bash
python inference.py
```

Parameters:
- `model_path`: Path to the fine-tuned model (default: "./out/full/final_model")
- `model_type`: Type of fine-tuning used (choices: "full", "lora", "adapter")
- `text`: Text to generate headline for (optional, will prompt if not provided)

## Project Structure
- `train.py`: Training script and pipeline
- `dataset.py`: Dataset loading class
- `eval.py`: Evaluation script
- `inference.py`: Inference script for generating headlines

## Training Configuration
- Batch size: `32`
- Learning rate: `3e-4`
- GPU: `NVIDIA RTX 3090 24GB`