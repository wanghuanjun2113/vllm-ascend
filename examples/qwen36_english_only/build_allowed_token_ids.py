#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def build(tokenizer_path: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    special_ids = set(tokenizer.all_special_ids)
    allowed = []
    for start in range(0, len(tokenizer), 8192):
        ids = list(range(start, min(start + 8192, len(tokenizer))))
        texts = tokenizer.batch_decode(
            [[token_id] for token_id in ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id, text in zip(ids, texts):
            if token_id == tokenizer.eos_token_id:
                allowed.append(token_id)
            elif token_id not in special_ids and text and text.isascii():
                allowed.append(token_id)
    digest = hashlib.sha256(
        json.dumps(allowed, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "tokenizer_path": tokenizer_path,
        "tokenizer_size": len(tokenizer),
        "allowed_count": len(allowed),
        "allowed_sha256": digest,
        "rule": "EOS or non-special token with non-empty standalone ASCII decode",
        "allowed_token_ids": allowed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    print(f"allowed_count={result['allowed_count']} sha256={result['allowed_sha256']}")


if __name__ == "__main__":
    main()
