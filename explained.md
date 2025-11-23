# PII NER Solution Explained

This document details the thinking process, technical approach, and evolution of the solution for the PII (Personally Identifiable Information) Named Entity Recognition (NER) task on noisy Speech-to-Text (STT) data.

## 1. The Problem
The objective was to build a system that identifies sensitive entities (Credit Cards, Phone Numbers, Emails, Names, Dates, etc.) from raw audio transcripts. 

**Key Challenges:**
*   **Noisy Input**: STT output often lacks punctuation, capitalizes incorrectly, and spells out numbers (e.g., "one two three" instead of "123").
*   **Strict Latency**: The system must process an utterance in under **20ms** (p95) on a CPU.
*   **High Precision**: False positives on PII are costly, so precision must be high (>= 0.80).

## 2. Data Strategy
Since no training data was provided, the first step was to generate a high-quality synthetic dataset that mimics the characteristics of noisy STT.

### Synthetic Generation with `Faker`
We used the `Faker` library to generate realistic PII entities. However, simply pasting these into templates isn't enough. We had to simulate STT noise:
*   **Normalization**: Converted all text to lowercase.
*   **Punctuation Removal**: Removed commas, periods, and dashes.
*   **Verbalization**:
    *   Replaced "." with " dot " (common in email dictation).
    *   Replaced "@" with " at ".
    *   Spelled out digits (0-9) as words ("zero", "one", ...).

**Why this matters**: A model trained on clean text (e.g., "Call me at 555-0199.") would fail miserably on STT output (e.g., "call me at five five five zero one nine nine").

## 3. Model Evolution & Thinking Process

### Phase 1: The Baseline (`distilbert-base-uncased`)
*   **Hypothesis**: DistilBERT is a standard, lightweight transformer that usually offers a good balance of speed and accuracy.
*   **Result**: 
    *   Accuracy was excellent (F1 ~0.98).
    *   **Latency was too high**: ~39ms per request.
*   **Conclusion**: We need to optimize for speed to meet the 20ms constraint.

### Phase 2: Dynamic Quantization
*   **Hypothesis**: Converting the model weights from 32-bit floating point (FP32) to 8-bit integers (INT8) should reduce memory bandwidth and speed up inference.
*   **Result**: 
    *   Latency **increased** to ~59ms.
*   **Why?**: On CPU with a batch size of 1, the overhead of quantizing activations on-the-fly often outweighs the compute benefits for small models. Quantization shines with larger batches or memory-bound scenarios, but here it was a bottleneck.

### Phase 3: Architecture Search (`bert-mini`)
*   **Hypothesis**: Since quantization failed, we need a physically smaller model (fewer layers, smaller hidden size). The task (NER) is relatively simple compared to complex reasoning, so a massive model is likely overkill.
*   **Selected Model**: `prajjwal1/bert-mini`
    *   Layers: 4 (vs 6 in DistilBERT)
    *   Hidden Size: 256 (vs 768 in DistilBERT)
    *   Parameters: ~11M (vs ~66M)
*   **Result**:
    *   **Latency**: **~17ms** (Success!)
    *   **Precision**: **~0.95** (Still well above the 0.80 target).

## 4. How It Works Under the Hood

1.  **Tokenization**: The input text (e.g., "my email is john at gmail dot com") is split into tokens using the BERT tokenizer.
2.  **Forward Pass**: The `bert-mini` model processes these tokens. It looks at the context of every word simultaneously (Self-Attention).
3.  **Classification Head**: On top of the BERT output, a simple linear layer predicts a label for each token (e.g., `B-EMAIL`, `I-EMAIL`, `O`).
4.  **Decoding**:
    *   The system takes the `argmax` of the logits to get the most likely label ID.
    *   It converts these IDs back to BIO tags (Begin, Inside, Outside).
    *   A helper function aggregates these tags into character-level spans (e.g., characters 12-35 are an EMAIL).

## 5. Conclusion
The final solution demonstrates that for specific, narrow tasks like NER, **smaller is often better**. By choosing the right model architecture (`bert-mini`) rather than relying on complex post-training optimizations like quantization, we achieved a **2x speedup** over DistilBERT while maintaining near-perfect accuracy.
