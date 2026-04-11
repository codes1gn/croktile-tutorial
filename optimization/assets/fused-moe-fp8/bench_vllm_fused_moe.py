#!/usr/bin/env python3
"""Benchmark vLLM's Triton fused_moe kernel for a fused MoE grouped GEMM.

Matches Croqtile's workload exactly:
  M=128 tokens, N=512, K=2048, 256 experts, top-8 routing, FP8 (e4m3)

This is NOT the full MLP (gate+up -> SiLU -> down). It is ONE grouped GEMM
like Croqtile's fused_moe_grouped_wgmma_fp8, so the TFLOPS are comparable.

Two timed configurations:

  1. GEMM-only:    just dispatch_fused_moe_kernel (no Croqtile equivalent)
  2. End-to-end:   zero + route + quant + align + GEMM + reduce
                   (matches Croqtile launch_end_to_end)

FLOP formula (same as Croqtile):
  FLOPs = 2 * expanded_m * N * K = 2 * (M * topk) * N * K

Usage:
  python bench_vllm_fused_moe.py --gpu 1
  python bench_vllm_fused_moe.py --m 256 --gpu 0
"""

import argparse

import torch
import triton.language as tl
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    dispatch_fused_moe_kernel,
    moe_align_block_size,
    moe_kernel_quantize_input,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()


def bench_single_gemm(
    m: int = 128,
    n: int = 512,
    k: int = 2048,
    num_experts: int = 256,
    topk: int = 8,
    num_warmup: int = 50,
    num_iters: int = 500,
    device: str = "cuda",
    config_override: dict | None = None,
):
    expanded_m = m * topk

    # --- Setup (not timed) ---
    a_bf16 = torch.randn((m, k), device=device, dtype=torch.bfloat16) / 10
    w_bf16 = torch.randn(
        (num_experts, n, k), device=device, dtype=torch.bfloat16
    ) / 10
    gating = torch.randn((m, num_experts), device=device, dtype=torch.float32)

    # Pre-quantize weights (weights are FP8 with per-expert scales)
    w_fp8 = w_bf16.to(FP8_DTYPE)
    w_scale = (
        w_bf16.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / 448.0
    )

    # Pre-quantize activations for GEMM-only path
    a_fp8_pre = a_bf16.to(FP8_DTYPE)
    a_scale_pre = (
        a_bf16.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / 448.0
    )

    # Pre-route for GEMM-only path
    topk_weights_pre, topk_ids_pre, _ = fused_topk(
        a_bf16, gating, topk, renormalize=True
    )

    # Output buffers
    intermediate = torch.zeros(
        (m, topk, n), device=device, dtype=torch.bfloat16
    )
    final_output = torch.zeros((m, n), device=device, dtype=torch.bfloat16)

    # Get Triton autotuning config
    if config_override:
        config = config_override
        print(f"  Triton config (OVERRIDE): {config}")
    else:
        config = try_get_optimal_moe_config(
            w1_shape=w_fp8.shape,
            w2_shape=w_fp8.shape,
            top_k=topk,
            dtype=str(FP8_DTYPE),
            M=m,
            block_shape=None,
        )
        print(f"  Triton config (auto): {config}")

    # Pre-align for GEMM-only path
    sorted_ids_pre, expert_ids_pre, ntp_pre = moe_align_block_size(
        topk_ids_pre, config["BLOCK_SIZE_M"], num_experts
    )

    def dispatch_gemm(a_q, a_s, tw, st, ei, ntp):
        dispatch_fused_moe_kernel(
            A=a_q,
            B=w_fp8,
            C=intermediate,
            A_scale=a_s,
            B_scale=w_scale,
            B_zp=None,
            topk_weights=tw,
            sorted_token_ids=st,
            expert_ids=ei,
            num_tokens_post_padded=ntp,
            mul_routed_weight=True,
            top_k=topk,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            block_shape=None,
        )

    # --- Configuration 1: GEMM only ---
    def run_gemm_only():
        dispatch_gemm(
            a_fp8_pre, a_scale_pre, topk_weights_pre,
            sorted_ids_pre, expert_ids_pre, ntp_pre,
        )

    # --- Configuration 2: End-to-end ---
    # Matches Croqtile launch_end_to_end():
    #   zero → route → count → build → quantize → sort → GEMM → scatter
    def run_end_to_end():
        final_output.zero_()
        intermediate.zero_()
        tw, ti, _ = fused_topk(a_bf16, gating, topk, renormalize=True)
        a_q, a_s = moe_kernel_quantize_input(
            A=a_bf16, A_scale=None, quant_dtype=FP8_DTYPE,
            per_act_token_quant=True, block_shape=None,
        )
        st, ei, ntp = moe_align_block_size(
            ti, config["BLOCK_SIZE_M"], num_experts
        )
        dispatch_gemm(a_q, a_s, tw, st, ei, ntp)
        ops.moe_sum(intermediate.view(m, -1, n), final_output)

    flops = 2.0 * expanded_m * n * k

    def measure(fn, label):
        for _ in range(num_warmup):
            fn()
        torch.cuda.synchronize()

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_iters)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_iters)
        ]
        for i in range(num_iters):
            torch.cuda.synchronize()
            start_events[i].record()
            fn()
            end_events[i].record()
        torch.cuda.synchronize()

        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_ms = sum(times_ms) / len(times_ms)
        min_ms = min(times_ms)
        max_ms = max(times_ms)
        tflops_val = (flops / (avg_ms / 1000.0)) / 1e12
        print(f"  [{label}]")
        print(
            f"    Avg: {avg_ms:.4f} ms  "
            f"Min: {min_ms:.4f} ms  Max: {max_ms:.4f} ms  "
            f"TFLOPS: {tflops_val:.3f}"
        )
        return avg_ms, tflops_val

    print(f"\n{'='*65}")
    print(f"vLLM Triton · single grouped GEMM (FP8)")
    print(f"  M={m}, N={n}, K={k}, experts={num_experts}, topk={topk}")
    print(f"  expanded_m = {expanded_m} (= M * topk)")
    print(f"  FLOPs = 2 * {expanded_m} * {n} * {k} = {flops/1e9:.3f} GFLOP")
    print(f"  vLLM version: ", end="")
    try:
        import vllm
        print(vllm.__version__)
    except Exception:
        print("unknown")
    print(f"  FP8 dtype: {FP8_DTYPE}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Warmup: {num_warmup}, Iterations: {num_iters}")
    print(f"{'='*65}")

    gemm_ms, gemm_tf = measure(run_gemm_only, "GEMM only")
    e2e_ms, e2e_tf = measure(
        run_end_to_end, "End-to-end (zero+route+quant+align+GEMM+reduce)"
    )

    print(f"{'='*65}")
    print(f"\n  Comparison with Croqtile (same FLOP formula, same hardware):")
    print(f"  Croqtile numbers from 05_cuda_optimized.co, 500-rep timing.")
    print()
    print(f"  {'Metric':<35} {'Croqtile':>10} {'vLLM':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    print(f"  {'GEMM-only latency':<35} {'n/a':>10} {gemm_ms:>9.4f}ms")
    print(f"  {'End-to-end latency':<35} {'~0.164ms':>10} {e2e_ms:>9.4f}ms")
    print(f"  {'End-to-end TFLOPS':<35} {'13.18':>10} {e2e_tf:>10.3f}")
    print(f"  {'Speedup (Croqtile / vLLM)':<35} {'':>10} {13.18/e2e_tf:>9.2f}x")
    print()

    return e2e_ms, e2e_tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM Triton single grouped GEMM (FP8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m", type=int, default=128, help="Number of tokens")
    parser.add_argument("--n", type=int, default=512, help="Output dimension")
    parser.add_argument(
        "--k", type=int, default=2048, help="Input/hidden dimension"
    )
    parser.add_argument(
        "--experts", type=int, default=256, help="Number of experts"
    )
    parser.add_argument("--topk", type=int, default=8, help="Top-k routing")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iters")
    parser.add_argument("--iters", type=int, default=500, help="Timing iters")
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU index (CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help='Override Triton config as JSON, e.g. \'{"BLOCK_SIZE_M":16,...}\''
    )
    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config_override = None
    if args.config:
        import json
        config_override = json.loads(args.config)

    bench_single_gemm(
        m=args.m,
        n=args.n,
        k=args.k,
        num_experts=args.experts,
        topk=args.topk,
        num_warmup=args.warmup,
        num_iters=args.iters,
        config_override=config_override,
    )
