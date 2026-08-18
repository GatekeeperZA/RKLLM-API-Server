---
name: Bug Report
about: Report a bug or unexpected behaviour in the RKLLM API Server
title: '[BUG] '
labels: bug
assignees: GatekeeperZA
---

## Describe the Bug

A clear description of what went wrong.

## Hardware & Software

| Item | Value |
|------|-------|
| Board | e.g. Orange Pi 5 Plus |
| RAM | e.g. 16 GB |
| OS | e.g. Armbian 24.x |
| NPU driver | `dmesg \| grep -i npu \| head -1` |
| RKLLM runtime | e.g. v1.2.3 |
| Python version | `python3 --version` |
| Model name | e.g. Qwen3-4B-Instruct |
| Model quantisation | e.g. w8a8 |

## Steps to Reproduce

1. 
2. 
3. 

## Expected Behaviour

What should have happened.

## Actual Behaviour

What actually happened.

## Relevant Logs

Paste the relevant section from `api.log` or `journalctl -u rkllm-api`:

```
paste logs here
```

## API Request (if applicable)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

## Additional Context

Any other context — does it happen with all models or only specific ones? Streaming vs non-streaming? etc.
