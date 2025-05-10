from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from adapters import T5AdapterModel
import torch
import argparse

def load_model(model_path: str, model_type: str = "full"):
    """Load the appropriate model based on fine-tuning type."""
    if model_type == "full":
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    elif model_type == "lora":
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model = PeftModel.from_pretrained(base_model, model_path)
    elif model_type == "adapter":
        model = T5AdapterModel.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model

def generate_headline(text: str, model, tokenizer, max_length: int = 64):
    """Generate a headline for the given text."""
    # Add prefix for T5
    input_text = "summarize: " + text
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Move to GPU if available
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Generate headlines using fine-tuned models")
    parser.add_argument("--model_path", default="./out/full/final_model", help="Path to the fine-tuned model")
    parser.add_argument("--model_type", default="full", choices=["full", "lora", "adapter"], help="Type of fine-tuning used")
    parser.add_argument("--text", help="Text to generate headline for")
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # If text is provided as argument, use it
    if args.text:
        text = args.text
    else:
        # Otherwise, prompt user
        print("\nEnter text to generate headline for (press Ctrl+D or Ctrl+Z when done):")
        text = input("> ")
    
    # Generate and print headline
    headline = generate_headline(text, model, tokenizer)
    print("\nGenerated Headline:")
    print(headline)

if __name__ == "__main__":
    main() 