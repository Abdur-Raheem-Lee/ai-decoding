
import json
import os
import re
import sys
import warnings
from pathlib import Path

# Suppress third-party library noise before importing them 
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")    
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1") 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# ── Optional imports with graceful fallback ───────────────────────────────────

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("\n[warn] tiktoken not installed – skipping OpenAI-family decoders")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("\n[warn] transformers not installed – skipping HuggingFace decoders")

try:
    from langdetect import detect_langs, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("\n[warn] langdetect not installed – falling back to heuristic only")


class MessageDecoder:
    """
    Decodes an integer token list by trying many tokenizer 'dialects', then
    picks the most coherent result using heuristic + LLM-based scoring.
    """

    TIKTOKEN_ENCODINGS = [
        "cl100k_base",   # GPT-4, ChatGPT, text-embedding-ada-002
        "p50k_base",     # Codex, text-davinci-002/003
        "r50k_base",     # GPT-3 (davinci / curie / babbage / ada)
        "gpt2",          # GPT-2 via tiktoken shim
    ]

    HF_PUBLIC_MODELS: dict[str, str] = {
        "gpt2":         "gpt2",
        "opt-125m":     "facebook/opt-125m",
        "gpt-neo-125m": "EleutherAI/gpt-neo-125M",
        "flan-t5-base": "google/flan-t5-base",
        "bloom-560m":   "bigscience/bloom-560m",
    }

    COMMON_WORDS: set[str] = {
        # English
        "the", "and", "you", "to", "for", "with", "have", "are", "is",
        "we", "our", "your", "in", "on", "at", "be", "as", "it",
        "this", "that", "from", "please", "submit", "application",
        "let", "build", "something", "together",
        # Spanish
        "que", "de", "en", "el", "la", "los", "las", "una", "por", "con",
        # French
        "le", "les", "des", "est", "une", "et", "pas", "qui", "nous",
        # German
        "die", "der", "das", "und", "ist", "ein", "eine", "nicht", "wir",
        # Portuguese
        "que", "para", "uma", "com", "por", "seu", "sua",
    }

    # Minimum combined score required to save the file
    SCORE_THRESHOLD: float = 50.0

    def __init__(
        self,
        tokens: list[int],
        output_path: str = "decoded_message.txt",
    ) -> None:
        self.tokens = tokens
        self.output_path = Path(output_path)
        self.candidates: dict[str, str] = {}      


    def _decode_tiktoken(self, encoding_name: str) -> str | None:
        if not HAS_TIKTOKEN:
            return None
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return enc.decode(self.tokens)
        except Exception:
            return None
        

    def _decode_hf(self, model_id: str) -> str | None:
        if not HAS_TRANSFORMERS:
            return None
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            return tok.decode(self.tokens, skip_special_tokens=True)
        except Exception:
            return None


    def decode_all(self) -> dict[str, str]:
        """Run every registered decoder and store non-empty results."""
        for enc_name in self.TIKTOKEN_ENCODINGS:
            text = self._decode_tiktoken(enc_name)
            if text and text.strip():
                self.candidates[f"tiktoken:{enc_name}"] = text

        for label, model_id in self.HF_PUBLIC_MODELS.items():
            text = self._decode_hf(model_id)
            if text and text.strip():
                self.candidates[f"hf:{label}"] = text

        return self.candidates
 

    def heuristic_score(self, text: str) -> float:
        """
        Fast offline coherence estimate.
        Returns a value roughly in the range 0–50.
        """
        if not text or not text.strip():
            return 0.0

        score = 0.0
        words = text.split()

        # 1. Reasonable word count
        if len(words) >= 5:
            score += 10
        elif len(words) >= 2:
            score += 5

        # 2. Contains alphabetic characters
        if re.search(r"[a-zA-Z]", text):
            score += 10

        # 3. High ratio of printable ASCII  (most human text qualifies)
        printable = sum(32 <= ord(c) <= 126 for c in text)
        score += (printable / max(len(text), 1)) * 20

        # 4. Recognised common words (multilingual short-list)
        matched = sum(w.lower() in self.COMMON_WORDS for w in words)
        score += min(matched * 2, 10)   # capped at 10

        # 5. Penalise long non-ASCII runs (garbled bytes, wrong codec)
        non_ascii_runs = re.findall(r"[^\x00-\x7F]{4,}", text)
        score -= len(non_ascii_runs) * 5

        # 6. Penalise suspiciously long unbroken tokens (binary noise)
        long_tokens = [w for w in words if len(w) > 30]
        score -= len(long_tokens) * 3

        # 7. Reward natural average word length (real text: 4–7 chars)
        #    Garbage decodings tend to have very long or very short "words"
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            if 3 <= avg_len <= 8:
                score += 10
            elif avg_len > 15:
                score -= 10

        # 8. Reward sentence-like punctuation patterns (spaces after commas etc.)
        if re.search(r'[A-Z][^.!?]*[.!?]', text):
            score += 5

        return max(score, 0.0)

    def language_score(self, text: str) -> float:
        """
        Uses langdetect to check whether the text reads as a real human language.
        Returns 0–100 based on detection confidence; 0 if detection fails entirely
        (which is the expected result for garbled/garbage output).
        Works for 55+ languages – no API key or internet connection required.
        """
        if not HAS_LANGDETECT or not text.strip():
            return 0.0
        try:
            results = detect_langs(text)   # e.g. [en:0.98, fr:0.01]
            return round(results[0].prob * 100, 1)
        except LangDetectException:
            return 0.0

    # Orchestration 

    def get_best_candidate(self) -> tuple[str | None, str | None, float]:
        """
        Score every candidate; return (decoder_name, text, combined_score).
        Claude is only queried for candidates that pass the heuristic filter,
        to avoid wasting API calls on obvious garbage.
        """
        best_name, best_text, best_combined = None, None, -1.0

        for name, text in self.candidates.items():
            h = self.heuristic_score(text)
            l = self.language_score(text) if h >= 10 else 0.0
            combined = h + l

            preview = repr(text[:65])
            lang_col = f"  lang={l:5.1f}" if HAS_LANGDETECT else ""
            print(f"  {name:<28}  heuristic={h:4.1f}{lang_col}  {preview}")

            if combined > best_combined:
                best_combined = combined
                best_name = name
                best_text = text

        return best_name, best_text, best_combined

    def save(self, text: str) -> None:
        self.output_path.write_text(text, encoding="utf-8")
        print(f"\n✓ Decoded message saved to → {self.output_path}")

    def run(self) -> None:
        print(f"Decoding {len(self.tokens)} tokens …\n")
        self.decode_all()

        if not self.candidates:
            print(
                "No candidates produced. "
                "Check your token list and installed libraries."
            )
            return

        print(f"Scoring {len(self.candidates)} candidate(s):\n")
        name, text, score = self.get_best_candidate()

        print(f"\n── Best decoder : {name}  (combined score {score:.1f}) ──")
        print(text)

        if score >= self.SCORE_THRESHOLD:
            self.save(text)
        else:
            print(
                f"\n Score {score:.1f} is below threshold {self.SCORE_THRESHOLD}. "
                "Not saving – decoded text may be incoherent."
            )


# Token input helpers 

def parse_tokens(argv: list[str]) -> list[int]:
    """
    Resolution order:
      1. CLI positional args   → python decoder.py 52938 389 52829 ...
      2. Single JSON arg       → python decoder.py '[52938, 389]'
      3. Stdin JSON array      → echo '[52938]' | python decoder.py
      4. Built-in sample       → fallback for quick testing
    """
    if len(argv) > 1:
        # Try space-separated integers
        try:
            return list(map(int, argv[1:]))
        except ValueError:
            pass
        # Try a single JSON-encoded array
        try:
            parsed = json.loads(argv[1])
            if isinstance(parsed, list):
                return [int(t) for t in parsed]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try piped JSON
    if not sys.stdin.isatty():
        try:
            parsed = json.load(sys.stdin)
            if isinstance(parsed, list):
                return [int(t) for t in parsed]
        except Exception:
            pass

    # Built-in sample (cl100k_base encoded)
    return [
        52938, 389, 52829, 279, 5020, 13441, 15592, 8815, 0, 1226, 2351, 459,
        15592, 38710, 19074, 304, 800, 16046, 79548, 331, 13353, 14265, 15592,
        6848, 1139, 15525, 11, 41030, 11, 323, 69311, 3956, 13, 11244, 701,
        14499, 11, 14584, 3335, 11, 323, 264, 2723, 311, 701, 6425, 2082, 311,
        308, 38872, 31, 42792, 13441, 41483, 323, 597, 12, 8669, 31, 42792,
        13441, 929, 1320, 13, 763, 701, 2613, 11, 3371, 603, 902, 47058, 499,
        6818, 1176, 323, 3249, 13, 6914, 596, 1977, 2555, 24674, 3871, 13,
    ]

 

if __name__ == "__main__":
    tokens = parse_tokens(sys.argv)
    MessageDecoder(tokens).run()