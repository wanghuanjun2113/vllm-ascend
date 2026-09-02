# Qwen3.6 request-level `allowed_token_ids`

## Purpose

Generate an ASCII-oriented token allow-list and pass it on each request. This is the lowest-risk validation and application-integration option because it does not modify model weights or the server-wide decoding path.

## Design

`examples/qwen36_english_only/build_allowed_token_ids.py` scans the tokenizer. It keeps EOS and non-special tokens whose standalone decoded text is non-empty ASCII. The resulting IDs are sent through the existing OpenAI-compatible `allowed_token_ids` request field.

## Usage

```bash
python examples/qwen36_english_only/build_allowed_token_ids.py   --tokenizer /mnt/weights/Qwen3.6-27B-w8a8   --output /path/allowed_ascii_token_ids.json
```

The application reads `allowed_token_ids` from the JSON artifact and inserts the list into each completion request.

## Advantages

- No checkpoint or vLLM-Ascend source modification.
- Per-request control and immediate rollback.
- Suitable for reproducing language leakage and validating the target token set.

## Limitations

- The large ID list is transmitted and processed per request.
- It constrains tokens, not semantic language intent; application output validation is still recommended.
- Standalone token decoding is an engineering rule, not a formal proof over every byte-level composition.

## Validation snapshot

On the pinned five language-induction cases, the ASCII allow-list removed CJK output. Use the main design document for the complete accuracy and performance comparison.
