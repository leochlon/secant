#!/usr/bin/env python3
"""Capture post-RoPE Q/K/V at one layer so the identity can be checked without a model.

Mirrors the `paper` profile: Qwen2.5-0.5B at the pinned revision, layer 23,
511-token prefixes, all 14 query heads. Tensors are stored in the model's own
float32; the verification casts to float64, which is what produces the float64
residuals the paper reports.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
LAYER, PREFIX_TOKENS = 23, 511

PROMPTS = [
    "The history of cartography is in part a history of the instruments available to "
    "surveyors, and in part a history of what states wished to be able to see. ",
    "In numerical linear algebra the conditioning of a problem is distinct from the "
    "stability of an algorithm used to solve it, and conflating the two is a common error. ",
]


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def capture(out_path, contexts):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager").eval()
    attn = model.model.layers[LAYER].self_attn
    config = model.config
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim = config.hidden_size // heads
    saved = {}

    grabbed = {}

    def hook(module, args, kwargs):
        # Capture the inputs the layer actually receives, rather than
        # reimplementing the stack that produces them.
        grabbed["hidden"] = kwargs.get("hidden_states", args[0] if args else None)
        grabbed["pos"] = kwargs.get("position_embeddings")
        return None

    handle = attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for index, prompt in enumerate(contexts):
            ids = tok(prompt * 40, return_tensors="pt")["input_ids"][:, :PREFIX_TOKENS]
            with torch.no_grad():
                model(input_ids=ids)
                hidden, (cos, sin) = grabbed["hidden"], grabbed["pos"]
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
                q = rope(attn.q_proj(hidden).view(1, -1, heads, dim).transpose(1, 2), cos, sin)
                k = rope(attn.k_proj(hidden).view(1, -1, kv_heads, dim).transpose(1, 2), cos, sin)
                v = attn.v_proj(hidden).view(1, -1, kv_heads, dim).transpose(1, 2)
            saved[f"context_{index}_query"] = q[0, :, -1].numpy()   # H, D — the final query
            saved[f"context_{index}_keys"] = k[0].numpy()           # G, N, D
            saved[f"context_{index}_values"] = v[0].numpy()         # G, N, D
            print(f"  context {index}: {ids.shape[1]} tokens, "
                  f"q{tuple(saved[f'context_{index}_query'].shape)} "
                  f"k{tuple(saved[f'context_{index}_keys'].shape)}")
    finally:
        handle.remove()

    meta = dict(model=MODEL, revision=REVISION, layer=LAYER, heads=heads,
                kv_heads=kv_heads, head_dim=dim, contexts=len(contexts),
                scale=float(attn.scaling), dtype="float32", attention="eager")
    np.savez(out_path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **saved)
    print(f"wrote {out_path} ({Path(out_path).stat().st_size/1e6:.2f} MB)")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reference_capture.npz")
    parser.add_argument("--contexts", type=int, default=len(PROMPTS))
    capture(parser.parse_args().out, PROMPTS[:parser.parse_args().contexts])
