import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pii_dataset import PIIDataset
from labels import ID2LABEL, LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
        # Quantized models run on CPU
        args.device = "cpu"
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        model.to(args.device)
        
    model.eval()

    ds = PIIDataset(args.input, tokenizer, LABELS, max_length=256, is_train=False)
    
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            item = ds[i]
            input_ids = torch.tensor([item["input_ids"]]).to(args.device)
            attention_mask = torch.tensor([item["attention_mask"]]).to(args.device)
            offset_mapping = item["offset_mapping"]
            text = item["text"]
            utt_id = item["id"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Decode spans
            entities = []
            current_entity = None
            
            for idx, (start, end) in enumerate(offset_mapping):
                if start == end: continue # Special token
                
                pred_id = preds[idx]
                label = ID2LABEL[pred_id]
                
                if label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "label": label[2:],
                        "start": start,
                        "end": end
                    }
                elif label.startswith("I-"):
                    if current_entity and current_entity["label"] == label[2:]:
                        current_entity["end"] = end
                    else:
                        # Invalid I-tag or new entity implicitly
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {
                            "label": label[2:],
                            "start": start,
                            "end": end
                        }
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = None
            
            if current_entity:
                entities.append(current_entity)
                
            predictions.append({
                "id": utt_id,
                "text": text,
                "entities": entities
            })
            
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
            
    print(f"Wrote predictions for {len(predictions)} utterances to {args.output}")

if __name__ == "__main__":
    main()
