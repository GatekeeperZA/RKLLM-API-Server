"""
RKLLM Model Converter — v1.3.0 toolkit, RK3588 target

RK3588 NPU hardware constraints (confirmed against runtime v1.3.0):
  - Supported quantization: w8a8 ONLY
  - w4a16/w4a16_g32/w4a16_g64/w4a16_g128 — NOT supported (RK3576+ only)
  - w8a8_g128/g256/g512 — toolkit accepts them but runtime rejects on load

Usage:
    python convert_rkllm.py --model Qwen/Qwen3-1.7B --quant w8a8
    python convert_rkllm.py --model Qwen/Qwen3-4B-Instruct --quant w8a8 --context 8192
    python convert_rkllm.py --model /local/path/to/model --quant w8a8

Output: ./output/<model-slug>-rk3588-<quant>.rkllm
Then scp to: armbian@192.168.2.180:/home/armbian/models/<ModelName>/
"""
import argparse
import os
import re
import sys
from pathlib import Path

from rkllm.api import RKLLM


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "-", name.split("/")[-1]).strip("-")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to .rkllm")
    parser.add_argument("--model", required=True, help="HF repo ID or local path")
    parser.add_argument("--quant", default="w8a8",
                        choices=["w8a8"],
                        help="Quantization dtype (default: w8a8). RK3588 only supports w8a8.")
    parser.add_argument("--context", type=int, default=8192,
                        help="Max context length (default: 8192)")
    parser.add_argument("--hybrid-rate", type=float, default=0.0,
                        help="Hybrid quantization rate 0.0-1.0 (default: 0.0)")
    parser.add_argument("--output", default=None,
                        help="Output .rkllm path (auto-generated if omitted)")
    args = parser.parse_args()

    slug = slugify(args.model)
    script_dir = Path(__file__).parent
    out_dir = script_dir / "output"
    out_path = args.output or str(out_dir / f"{slug}-rk3588-{args.quant}.rkllm")
    out_dir.mkdir(exist_ok=True)

    print(f"Model  : {args.model}")
    print(f"Quant  : {args.quant}")
    print(f"Context: {args.context}")
    print(f"Output : {out_path}")
    print()

    llm = RKLLM()

    # Download model to local cache if it's a HF repo ID
    local_path = args.model
    if not os.path.isdir(args.model):
        print("Downloading model from HuggingFace...")
        from huggingface_hub import snapshot_download
        # Read token from cache file directly — newer hf CLI stores it there
        _token_file = os.path.expanduser("~/.cache/huggingface/token")
        _token = open(_token_file).read().strip() if os.path.exists(_token_file) else None
        local_path = snapshot_download(repo_id=args.model, token=_token)
        print(f"Downloaded to: {local_path}")

    print("Loading model weights...")
    ret = llm.load_huggingface(
        model=local_path,
        device="cpu",
        dtype="float32",   # float16 not supported on CPU
    )
    if ret != 0:
        print(f"ERROR: load_huggingface failed (ret={ret})")
        sys.exit(1)

    print("Building / quantizing...")
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=args.quant,
        quantized_algorithm="normal",
        target_platform="rk3588",
        num_npu_core=3,
        hybrid_rate=args.hybrid_rate,
        max_context=args.context,
    )
    if ret != 0:
        print(f"ERROR: build failed (ret={ret})")
        sys.exit(1)

    print(f"Exporting to {out_path} ...")
    ret = llm.export_rkllm(out_path)
    if ret != 0:
        print(f"ERROR: export failed (ret={ret})")
        sys.exit(1)

    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"\nDone — {out_path} ({size_mb:.0f} MB)")
    print(f"\nDeploy with:")
    print(f"  scp {out_path} root@192.168.2.180:/home/armbian/models/{slug}/")


if __name__ == "__main__":
    main()
