import torch
import os
import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    print("Quantizing model...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save the quantized model state dict
    # Note: Transformers doesn't support saving quantized models directly with save_pretrained in a way that loads back easily without extra steps.
    # But for this task, we can save the state dict and load it back, or just save it as a torch script if needed.
    # However, to keep it simple and compatible with the existing predict/eval scripts (which might need modification to load quantized model),
    # we will save it such that we can load it.
    
    # Actually, let's just save the state dict and the config.
    # But wait, predict.py uses AutoModelForTokenClassification.from_pretrained.
    # We might need to modify predict.py and measure_latency.py to support loading quantized models.
    
    # Let's save it as a standard pytorch model for now, and we'll create a specific runner for quantized model.
    torch.save(quantized_model.state_dict(), os.path.join(args.out_dir, "quantized_model.pt"))
    model.config.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    
    print(f"Quantized model saved to {args.out_dir}")

if __name__ == "__main__":
    main()
