# RKLLM v1.3.0 Known Issues

Tracked issues found during the v1.3.0 upgrade (2026-08-09).
Branch was archived and rolled back to v1.2.3 pending upstream fixes.

---

## Issue 1: Two-phase thinking — phase 2 early EOS (~25 tokens)

**Severity:** Critical (makes thinking unusable)

**Description:**  
In v1.3.0, `rkllm_run` fires `RKLLM_RUN_FINISH` at the `</think>` token instead of at the
actual end-of-sequence. The caller must invoke `rkllm_run` a second time to generate the
answer (phase 2). However, when phase 2 is called with any system-prompt context in the
session, the model generates 13–27 tokens and then emits EOS mid-sentence.

**Root cause (suspected):**  
The API context (system prompt + prompt cache) causes the model to interpret the assistant
turn as already complete. Passing `role=None` instead of `role=b"assistant"` in the phase 2
`RKLLMInput` improved token count slightly but did not resolve the truncation.

**What was tried:**
- `role=b"assistant"` → inserts duplicate `<|im_start|>assistant\n` header → EOS after ~13 tokens
- `role=None` → no header injected → EOS after ~25 tokens, mid-sentence
- `RKLLMSamplingParam` has no `ignore_eos_token` field — no per-call workaround available

**Reference behaviour:**  
Without a system prompt (bare prompt), phase 2 may generate longer responses. The truncation
appears tied to the system-prompt cache or context length accounting in v1.3.0.

**Where to look when fixing:**  
- `api.py` function `_run_rkllm()` — phase 2 `rkllm_run` call
- `_phase1_active`, `_phase1_buffer`, `_phase2_start` globals
- ThinkTagParser state (`think_parser.in_thinking`) used to detect phase boundary

---

## Issue 2: `</think>` consumed silently — ThinkTagParser stuck in thinking state

**Severity:** High (caused empty content for 1.7B model)

**Description:**  
RKLLM v1.3.0 consumes the `</think>` token as its phase-1 stop signal without emitting it
as a decoded token to the callback. This leaves ThinkTagParser's `in_thinking` flag `True`
at the end of phase 1, causing all phase 2 content to be routed to `reasoning_content`
instead of `content`.

**Fix applied (in this branch):**  
After phase 1 completes (`thinking_end` signal), inject `</think>\n` into the consumer
only when `think_parser.in_thinking is True`. This closes the thinking tag explicitly and
flushes the parser to content mode before phase 2 begins.

---

## Issue 3: Qwen3-4B emits no `<think>` in phase 1

**Severity:** Medium (caused duplicate content)

**Description:**  
The Qwen3-4B model generates the final answer directly in phase 1 without emitting `<think>`.
When phase 2 is then called, the model regenerates the same answer, causing the response to
contain the answer twice.

**Fix applied (in this branch):**  
After phase 1 completes, check `if "<think>" not in phase1_text` → skip phase 2 entirely,
flush phase 1 buffer as final content, and emit `finish`.

---

## Issue 4: ABI incompatibility — v1.2.3 and v1.3.0 struct layouts differ

**Severity:** Critical (causes `rkllm_init ret=-1` silent failure)

**Description:**  
`RKLLMParam`, `RKLLMInput`, and `RKLLMInferParam` ctypes structs changed between v1.2.3
and v1.3.0. Loading a v1.2.3 `api.py` against the v1.3.0 runtime (or vice versa) causes
`rkllm_init` to return `-1` with no further error message.

**Workaround (used for rollback):**  
Copy v1.2.3 binary to `~/librkllmrt.so` on the device. Add `~/librkllmrt.so` as the first
candidate in the library search order in `api.py` so it takes precedence over `/usr/lib/`.
This allows pinning the runtime without sudo access to replace system libraries.

**Backup location on device:** `/usr/lib/librkllmrt.so.v1.2.3.bak`

---

## Issue 5: No `ignore_eos_token` in `RKLLMSamplingParam`

**Severity:** Medium (blocks phase 2 workaround)

**Description:**  
v1.3.0 `RKLLMSamplingParam` struct does not expose an `ignore_eos_token` field that would
allow overriding early EOS in phase 2. This was confirmed by inspecting the v1.3.0 header
(`rkllm.h`) — no such field exists.

**Upstream request:**  
If Rockchip adds `ignore_eos_token` or documents the correct two-phase calling convention
in a future release, Issue 1 may become solvable without runtime changes.

---

## Resolution

Rolled back to v1.2.3 on 2026-08-09. All issues above are v1.3.0-specific and do not
affect the v1.2.3 runtime. Revisit when a newer v1.3.x release is available from
[airockchip/rknn-llm](https://github.com/airockchip/rknn-llm).
