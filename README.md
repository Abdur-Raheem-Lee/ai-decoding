# Dynamic Message Decoder
 
A Python tool that decodes token sequences by trying multiple LLM tokenizer dialects and automatically identifying the correct one using language detection scoring.
 
---
 
## How It Works
 
The decoder tries every registered tokenizer against the input token list, scores each decoded result using a combination of heuristics and language detection, and saves the most coherent result to a file.
 
**Scoring pipeline:**
1. **Heuristic score** — checks word count, printable character ratio, average word length, common vocabulary, and sentence structure
2. **Language score** — uses `langdetect` to confirm the output reads as a real human language (55+ languages supported), returning a confidence value between 0–100
3. The candidate with the highest combined score wins; it is only saved if it clears the minimum threshold
---
 
## Approach
 
Having very limited interaction with tokenisers, my first step was to research how they work and what the key differences between them are. That research led me to an important observation: the highest token ID in the message was 79,548.
 
Most common tokenisers have vocabularies of around 50,000 tokens, which means encodings like `gpt2`, `r50k_base`, and `p50k_base` would fail outright on an ID that high. `cl100k_base`, with a vocabulary of ~100,277 tokens, was the only tiktoken encoding large enough to handle it — making it the clear first candidate to try. Its widespread use in production systems (GPT-4, ChatGPT) gave further confidence, and it turned out to be the correct dialect.
 
For AI-assisted development, I used ChatGPT to get the foundational class structure in place, and Claude to clean up the code, improve the scoring logic, and add multilingual support.

---
 
## Tokenizers Tried
 
| Dialect | Encoding / Model | Vocabulary Size |
|---|---|---|
| tiktoken | `cl100k_base` | ~100,277 |
| tiktoken | `p50k_base` | ~50,281 |
| tiktoken | `r50k_base` | ~50,257 |
| tiktoken | `gpt2` | ~50,257 |
| HuggingFace | `gpt2` | ~50,257 |
| HuggingFace | `facebook/opt-125m` | ~50,272 |
| HuggingFace | `EleutherAI/gpt-neo-125M` | ~50,257 |
| HuggingFace | `google/flan-t5-base` | ~32,100 |
| HuggingFace | `bigscience/bloom-560m` | ~250,680 |
 
---
 
## Installation
 
```bash
pip install -r requirements.txt
```
 
---
 
## Usage
 
```bash
# Use the built-in sample tokens
python message_decoder.py
 
# Pass tokens as command-line arguments
python message_decoder.py 52938 389 52829 279 ...
 
# Pipe a JSON array via stdin
echo '[52938, 389, 52829]' | python message_decoder.py
```
 
The decoded message is saved to `decoded_message.txt` if it scores above the confidence threshold.
 
---

The decoded message is saved to `decoded_message.txt` if it scores above the confidence threshold.
 
---
 
## Dependencies
 
| Package | Purpose |
|---|---|
| `tiktoken` | OpenAI-family tokenizer encodings |
| `transformers` | HuggingFace tokenizer models |
| `langdetect` | Language detection scoring (no API key required) |
