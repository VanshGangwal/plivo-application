import os
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pii_dataset import PIIDataset
from labels import LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--quantized", action="store_true", help="Load quantized model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    if args.quantized:
        # Load config and create model structure
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_dir)
        model = AutoModelForTokenClassification.from_config(config)
        # Quantize structure
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        # Load state dict
        state_dict = torch.load(os.path.join(args.model_dir, "quantized_model.pt"))
        model.load_state_dict(state_dict)
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        
    model.eval()
    # CPU only for latency measurement as requested
    model.to("cpu")

    # Load dataset
    ds = PIIDataset(args.input, tokenizer, LABELS, max_length=256, is_train=False)
    
    latencies = []
    
    print(f"Measuring latency over {args.runs} runs...")
    
    with torch.no_grad():
        for _ in range(args.runs):
            # Pick a random sample
            idx = np.random.randint(0, len(ds))
            item = ds[idx]
            
            input_ids = torch.tensor([item["input_ids"]])
            attention_mask = torch.tensor([item["attention_mask"]])
            
            start = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000) # ms

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    print(f"Latency over {args.runs} runs (batch_size={args.batch_size}):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")

if __name__ == "__main__":
    main()
