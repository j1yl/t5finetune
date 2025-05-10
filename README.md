# Fine-tune T5 Small for Headline Generation

## About
- Model: `t5-small`
- Dataset: `Gigaword 10k`
- Task: `Generate a short headline from a full sentence or paragraph.`

## Dataset & model
- [https://huggingface.co/google-t5/t5-small](https://huggingface.co/google-t5/t5-small)
- [https://huggingface.co/datasets/anumafzal94/gigaword_10k_finetuning](https://huggingface.co/datasets/anumafzal94/gigaword_10k_finetuning)

## Project structure
- `train.py` : training script and pipeline
- `dataset.py` : class for loading dataset

## Todo
- [ ] Adapter-based fine tuninig
- [ ] LoRA implementation
- [ ] Comparison between different fine-tuning methods
- [ ] Hyperparameter analysis (LoRA rank)
- [ ] Timing measurements
- [ ] Accuracy comparisons
- [ ] Comprehensive analysis of results$$  $$