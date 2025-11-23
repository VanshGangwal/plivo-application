# PII NER Solution Walkthrough

## Goal
Build a token-level NER model to identify PII entities (CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE, CITY, LOCATION) in noisy STT transcripts with p95 latency <= 20ms and PII precision >= 0.80.

## Solution Overview
- **Data Generation**: Created synthetic noisy STT data using `Faker` and custom noise injection (spelled out numbers, "dot", "at", etc.).
- **Model**: Selected `prajjwal1/bert-mini` (4 layers, 256 hidden) after determining that `distilbert-base-uncased` was too slow (~39ms) and dynamic quantization didn't help (~59ms).
- **Results**:
    - **Latency (p95)**: 16.89 ms (Target: <= 20ms)
    - **PII Precision**: 0.949 (Target: >= 0.80)
    - **PII Recall**: 0.977
    - **PII F1**: 0.963

## Implementation Details

### Data Generation
Used `src/generate_data.py` to generate 1000 training samples and 200 dev samples. The script simulates STT errors like:
- "one two three" instead of "123"
- "dot" instead of "."
- "at" instead of "@"
- Lowercase text
- Removed punctuation

### Model Optimization
1. **Baseline (`distilbert-base-uncased`)**:
   - p95 Latency: ~39 ms
   - PII Precision: ~0.98
   - *Too slow.*

2. **Quantization (Dynamic)**:
   - p95 Latency: ~59 ms
   - *Failed to improve latency on CPU with batch size 1.*

3. **Smaller Model (`prajjwal1/bert-mini`)**:
   - p95 Latency: **16.89 ms**
   - PII Precision: **0.949**
   - *Success.*

## Verification
### Latency
Run `src/measure_latency.py` with `bert-mini`:
```bash
conda run -n plivo python src/measure_latency.py --model_dir out_mini --input data/dev.jsonl --runs 50
```
Output:
```
Latency over 50 runs (batch_size=1):
  p50: 5.46 ms
  p95: 16.89 ms
```

### Precision/Recall
Run `src/eval_span_f1.py`:
```bash
conda run -n plivo python src/eval_span_f1.py --gold data/dev.jsonl --pred out_mini/dev_pred.json
```
Output:
```
Per-entity metrics:
CITY            P=0.972 R=0.972 F1=0.972
CREDIT_CARD     P=1.000 R=1.000 F1=1.000
DATE            P=0.971 R=1.000 F1=0.986
EMAIL           P=0.923 R=0.947 F1=0.935
LOCATION        P=0.969 R=1.000 F1=0.984
PERSON_NAME     P=0.881 R=0.949 F1=0.914
PHONE           P=1.000 R=1.000 F1=1.000

Macro-F1: 0.970

PII-only metrics: P=0.949 R=0.977 F1=0.963
Non-PII metrics: P=0.971 R=0.985 F1=0.978
```

## How to Run
1. **Install Dependencies**:
   ```bash
   conda create -n plivo python=3.10
   conda activate plivo
   pip install -r requirements.txt
   ```
2. **Generate Data**:
   ```bash
   python src/generate_data.py --output data/train.jsonl --count 1000
   python src/generate_data.py --output data/dev.jsonl --count 200
   ```
3. **Train**:
   ```bash
   python src/train.py --model_name prajjwal1/bert-mini --train data/train.jsonl --dev data/dev.jsonl --out_dir out_mini --epochs 5 --lr 5e-5
   ```
4. **Predict & Evaluate**:
   ```bash
   python src/predict.py --model_dir out_mini --input data/dev.jsonl --output out_mini/dev_pred.json
   python src/eval_span_f1.py --gold data/dev.jsonl --pred out_mini/dev_pred.json
   ```
5. **Measure Latency**:
   ```bash
   python src/measure_latency.py --model_dir out_mini --input data/dev.jsonl --runs 50
   ```
