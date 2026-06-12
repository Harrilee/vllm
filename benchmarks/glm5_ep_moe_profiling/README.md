# GLM-5 EP MoE Profiling on vLLM (GB200 NVL72)

Per-kernel profiling of GLM-5 expert-parallel MoE on vLLM, reproducing an sglang
DeepEP study on oci-hsg-cs-001 (GB200 NVL72, 4 GPU/node, aarch64). torch.profiler
captures CUDA-graph-internal kernel times; `analyze_moe_perf_vllm.py` parses the
Chrome traces into per-(EP, bs, op) p50/mean/p90.

## Layout
- `data/vllm_results.csv` — all results. Schema: `job,kind,phase,op,meta,n,p50_us,mean_us,p90_us,scope,src`.
  `meta` tags the config, e.g. `ep8,fp8,bs8,deepep_ll_fabric`.
- `reports/VLLM_EP_PERF_SWEEP.html` — a2a-backend comparison (NCCL allgather, DeepEP-HT,
  flashinfer_nvlink_one_sided) + cross-framework sglang-NCCL baseline.
- `reports/VLLM_FP8_DEEPEP_LL_GRID.html` — the full FP8 **DeepEP low-latency** grid
  EP{8,16,32,64} × bs{1,2,8,32,64,128} (real `internode_ll::dispatch/combine`).
- `scripts/` — launchers, profiling driver, sweep queues, analyzer.

## The DeepEP cross-node fix
The published `vllm/vllm-openai` image bundles DeepEP commit `73b6ea4`, whose
`allow_mnnvl` path fails cross-node at `deep_ep.cpp:226 runtime.sync` with
`CUDA error: invalid resource handle` (cudaIpc handles can't cross node boundaries).

The fix mirrors what sglang does: build DeepEP's **`hybrid-ep`** branch (commit
`d28bd67`), which exposes `Buffer(use_fabric=True)` → `CU_MEM_HANDLE_TYPE_FABRIC`
shareable handles (the cross-node-capable path), and have vLLM pass it through.

Two pieces:
1. **Code patch** (`vllm/distributed/device_communicators/all2all.py`): both the
   DeepEP HT and LL managers pass `use_fabric=True` when `VLLM_DEEPEP_USE_FABRIC=1`.
   No-op on a DeepEP build without the kwarg.
2. **Runtime overlay**: build hybrid-ep `deep_ep_cpp` into a dir on PYTHONPATH ahead
   of the image's DeepEP (only `deep_ep_cpp`; drop `hybrid_ep_cpp` which needs DOCA).
   CUDA dev headers (`cusparse.h`) come from pip `nvidia-cusparse/cublas/cusolver-cu13`
   since the runtime image lacks them.

Verified: 8-rank cross-node `deep_ep.Buffer(use_fabric=True)` init succeeds, then the
full EP8–64 FP8 DeepEP-LL grid runs with real `internode_ll` kernels.

## Key findings (reproduced from the sglang study)
- a2a (dispatch/combine) is **latency-bound**: dispatch p50 stays ~11–15 µs flat across
  bs 1→128 and across EP 8→64.
- expert compute (`deep_gemm sm100_fp8`) **shrinks with EP** (more experts ⇒ fewer per
  rank): ~9.8 µs (EP8) → ~8.8 µs (EP64) at bs1.
- combine grows modestly with batch size.
- Backend availability on this stack: DeepEP-LL needs the `use_fabric` overlay above;
  NVFP4 + DeepEP-LL is additionally blocked by a FlashInfer-CUTEDSL ↔ cutlass-MLIR
  codegen bug in the image (use FP8 weights, which route to `BatchedDeepGemmExperts`).

> Profiling/orchestration scripts are cluster-specific (SLURM + pyxis/enroot on
> oci-hsg-cs-001) and provided for reference, not as a portable harness.
