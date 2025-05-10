from datasets import load_dataset
from transformers import T5Tokenizer

ds = load_dataset("anumafzal94/gigaword_10k_finetuning")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
prefix = "summarize: "


def preprocess(example):
    input_text = prefix + example["text"]
    target_text = example["summary"]

    model_inputs = tokenizer(
        input_text, max_length=512, truncation=True, padding="max_length"
    )

    labels = tokenizer(
        target_text, max_length=64, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


tokenized_ds = ds.map(preprocess, batched=False)
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# print(tokenized_ds["train"][0])
