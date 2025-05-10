# Fine-tuning Methods Comparison Report

## Overall Comparison

### Training Time, Loss, and Metrics

| fine_tune_type   |   training_time |   train_loss |   eval_loss |   rouge1 |   rouge2 |   rougeL |     bleu |
|:-----------------|----------------:|-------------:|------------:|---------:|---------:|---------:|---------:|
| base             |           0     |      0       |    0        | 0.297163 | 0.102539 | 0.267343 |  4.85301 |
| full             |         548.904 |      1.97328 |    0.395167 | 0.440796 | 0.217867 | 0.418318 | 18.7545  |
| adapter          |         431.689 |     62.339   |    8.93917  | 0.298839 | 0.103654 | 0.269144 |  5.01654 |
| lora             |         439.582 |      4.9417  |    0.477331 | 0.38938  | 0.17265  | 0.366417 | 13.435   |
| lora             |         426.996 |      4.8828  |    0.484815 | 0.382875 | 0.169171 | 0.360058 | 12.9733  |
| lora             |         462.379 |      4.81209 |    0.492509 | 0.374512 | 0.164014 | 0.350775 | 12.4805  |

## Method-specific Analysis


### BASE Fine-tuning

- Evaluation Loss: 0.0000

Metrics:
- ROUGE1: 0.2972
- ROUGE2: 0.1025
- ROUGEL: 0.2673
- BLEU: 4.8530

### FULL Fine-tuning

- Training Time: 548.90 seconds
- Training Loss: 1.9733
- Evaluation Loss: 0.3952

Metrics:
- ROUGE1: 0.4408
- ROUGE2: 0.2179
- ROUGEL: 0.4183
- BLEU: 18.7545

### ADAPTER Fine-tuning

- Training Time: 431.69 seconds
- Training Loss: 62.3390
- Evaluation Loss: 8.9392

Metrics:
- ROUGE1: 0.2988
- ROUGE2: 0.1037
- ROUGEL: 0.2691
- BLEU: 5.0165

### LORA Fine-tuning

- Training Time: 439.58 seconds
- Training Loss: 4.9417
- Evaluation Loss: 0.4773

Metrics:
- ROUGE1: 0.3894
- ROUGE2: 0.1726
- ROUGEL: 0.3664
- BLEU: 13.4350

LoRA Configuration:
- Rank: 4
- Alpha: 32
- Dropout: 0.1