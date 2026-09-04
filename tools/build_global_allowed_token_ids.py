#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a model-specific global output-token allow-list.

The generated JSON is accepted by
VLLM_ASCEND_GLOBAL_ALLOWED_TOKEN_IDS_PATH. Normal tokens are allowed only
when their standalone decoded text contains TAB/LF/CR or printable ASCII.
EOS is allowed by default. Other special tokens require explicit opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

SCRIPT_VERSION = 1
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "tokenizer.model",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(model_dir: Path, filenames: Iterable[str]) -> dict[str, Any]:
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for filename in filenames:
        path = model_dir / filename
        if not path.is_file():
            continue
        file_digest = sha256_file(path)
        files[filename] = file_digest
        combined.update(filename.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(file_digest))
    return {
        "files": files,
        "combined_sha256": combined.hexdigest() if files else None,
    }


def load_model_vocab_size(model_dir: Path) -> tuple[int, str]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = (
        (config.get("vocab_size"), "config.vocab_size"),
        ((config.get("text_config") or {}).get("vocab_size"), "config.text_config.vocab_size"),
    )
    for value, source in candidates:
        if isinstance(value, int) and value > 0:
            return value, source
    raise ValueError(f"No positive vocab_size found in {config_path}")


def flatten_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return list(value)
    raise TypeError(f"Unsupported token ID value: {value!r}")


def is_strict_ascii(text: str) -> bool:
    """Allow TAB, LF, CR and printable ASCII U+0020..U+007E only."""
    return bool(text) and all(char in "\t\n\r" or 0x20 <= ord(char) <= 0x7E for char in text)


def build_allow_list(
    tokenizer: Any,
    model_vocab_size: int,
    batch_size: int,
    extra_special_ids: set[int],
) -> tuple[list[int], dict[str, Any]]:
    tokenizer_size = len(tokenizer)
    if tokenizer_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer size {tokenizer_size} exceeds model vocab size {model_vocab_size}"
        )

    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    eos_ids = set(flatten_token_ids(tokenizer.eos_token_id))
    explicitly_allowed_special_ids = eos_ids | extra_special_ids

    unknown_special_ids = explicitly_allowed_special_ids - special_ids
    if unknown_special_ids:
        raise ValueError(
            "Explicitly allowed special IDs are not registered special tokens: "
            f"{sorted(unknown_special_ids)}"
        )
    out_of_range = {
        token_id
        for token_id in explicitly_allowed_special_ids
        if token_id < 0 or token_id >= model_vocab_size
    }
    if out_of_range:
        raise ValueError(f"Special token IDs outside model vocab: {sorted(out_of_range)}")

    allowed: list[int] = []
    blocked_empty = 0
    blocked_non_ascii = 0
    blocked_special = 0

    for start in range(0, tokenizer_size, batch_size):
        ids = list(range(start, min(start + batch_size, tokenizer_size)))
        texts = tokenizer.batch_decode(
            [[token_id] for token_id in ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id, text in zip(ids, texts):
            if token_id in explicitly_allowed_special_ids:
                allowed.append(token_id)
            elif token_id in special_ids:
                blocked_special += 1
            elif not text:
                blocked_empty += 1
            elif is_strict_ascii(text):
                allowed.append(token_id)
            else:
                blocked_non_ascii += 1

    allowed = sorted(set(allowed))
    stats = {
        "tokenizer_size": tokenizer_size,
        "model_vocab_size": model_vocab_size,
        "allowed_count": len(allowed),
        "blocked_count": model_vocab_size - len(allowed),
        "blocked_tokenizer_empty_count": blocked_empty,
        "blocked_tokenizer_non_ascii_count": blocked_non_ascii,
        "blocked_special_count": blocked_special,
        "blocked_model_padding_count": model_vocab_size - tokenizer_size,
        "all_special_token_ids": sorted(special_ids),
        "eos_token_ids": sorted(eos_ids),
        "explicitly_allowed_special_token_ids": sorted(explicitly_allowed_special_ids),
    }
    return allowed, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Number of singleton tokens decoded per batch",
    )
    parser.add_argument(
        "--allow-special-token-id",
        type=int,
        action="append",
        default=[],
        help="Additional registered special token ID to allow; repeat as needed",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON; compact JSON is smaller and is the default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model_vocab_size, vocab_size_source = load_model_vocab_size(model_dir)
    allowed, stats = build_allow_list(
        tokenizer=tokenizer,
        model_vocab_size=model_vocab_size,
        batch_size=args.batch_size,
        extra_special_ids=set(args.allow_special_token_id),
    )

    canonical_ids = json.dumps(allowed, separators=(",", ":")).encode("utf-8")
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    artifact = {
        "schema_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_dir),
        "vocab_size_source": vocab_size_source,
        "policy": {
            "name": "strict_ascii_v1",
            "normal_token_rule": "non-empty standalone decode containing only TAB/LF/CR or U+0020..U+007E",
            "special_token_rule": "EOS plus explicitly configured registered special token IDs",
            "default_blocked_value": True,
            "allowed_mask_value": False,
        },
        "tokenizer_decoder": str(decoder) if decoder is not None else None,
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "tokenizer_fingerprint": fingerprint_files(model_dir, TOKENIZER_FILES),
        "allowed_token_ids_sha256": sha256_bytes(canonical_ids),
        **stats,
        "allowed_token_ids": allowed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.pretty:
        content = json.dumps(artifact, ensure_ascii=True, indent=2) + "\n"
    else:
        content = json.dumps(artifact, ensure_ascii=True, separators=(",", ":")) + "\n"
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, output_path)

    print(json.dumps({
        "output": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "model_vocab_size": stats["model_vocab_size"],
        "tokenizer_size": stats["tokenizer_size"],
        "allowed_count": stats["allowed_count"],
        "blocked_count": stats["blocked_count"],
        "allowed_token_ids_sha256": artifact["allowed_token_ids_sha256"],
        "tokenizer_fingerprint_sha256": artifact["tokenizer_fingerprint"]["combined_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
