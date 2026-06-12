#!/usr/bin/env python3
"""vLLM variant of analyze_moe_perf.py — unify vLLM torch-profiler MoE EP traces
into the SAME CSV row-set as the sglang baseline, so the two frameworks compare
directly (runbook §4, §7).

This is a COPY of scripts/analyze_moe_perf.py with three vLLM-specific changes:
  1. Per-rank file naming: sglang wrote `...TP-<n>...`; vLLM's
     tensorboard_trace_handler names files `dp<d>_pp<p>_tp<t>_dcp<c>_ep<e>_rank<r>
     .<ts>.pt.trace.json.gz`. We extract rank from `rank<N>` (fallback dp/ep).
  2. EXPERT_KERNELS / dispatch / combine matchers are vLLM's. DeepEP's
     dispatch/combine come from the SAME deep_ep lib so the names likely match
     sglang's `internode_ll::dispatch/combine` — confirm with --list-kernels on
     the first job, then adjust the tuples below.
  3. A `--list-kernels` mode that dumps the top kernels by total GPU time from a
     trace, so you can READ the real names before committing them here (runbook
     §4: "先跑一个 job ... 列出 kernel 名, 再填进 analyzer 的 EXPERT_KERNELS").

Usage:
  # 1) discover kernel names from the first EP8 trace:
  analyze_moe_perf_vllm.py --list-kernels 'vllm_traces/vllm_ep8_*/torchprof/*.gz'
  # 2) once names confirmed, parse to CSV (same schema as sweep_results.csv):
  analyze_moe_perf_vllm.py --job <jid> --torch 'vllm_traces/vllm_ep8_*/torchprof/*.gz' \
      --phase decode --meta ep8,nvfp4,bs8 --csv sweep_data/vllm_results.csv
"""
import argparse, gzip, json, glob, statistics, sys, csv, os, re, collections

# a2a dispatch/combine kernel names — VERIFIED from vLLM EP4 traces via
# --list-kernels (job 3259631). vLLM uses one of two a2a backends:
#   * DeepEP  (deepep_low_latency): internode_ll::dispatch / combine  (same
#     deep_ep lib as sglang — matches sglang names). NOT yet runnable on this
#     site (NVSHMEM/IBGDA fabric wall) but kept so DeepEP traces classify too.
#   * allgather_reducescatter (the working backend here): NCCL collectives
#     ncclDevKernel_AllGather_*  (dispatch)  and  ReduceScatter_*  (combine).
# Match on the distinguishing verb so dispatch≠combine.
# Covers all vLLM a2a backends seen on this stack:
#   DeepEP:      internode_ll::dispatch/combine (LL), intranode::dispatch/combine (HT)
#   allgather:   ncclDevKernel_AllGather (dispatch) / ReduceScatter (combine)
#   flashinfer_nvlink_one_sided: trtllm moe_alltoall::moeA2ADispatch/CombineKernel
# Keep verbs distinct so dispatch≠combine. "notify_dispatch"/"cached_notify_combine"
# (DeepEP HT setup kernels) are deliberately NOT matched (they lack ::dispatch/::combine
# and the A2ADispatchKernel/A2ACombineKernel spellings).
A2A_DISPATCH = ("internode_ll::dispatch", "::dispatch", "AllGather", "all_gather",
                "moeA2ADispatchKernel", "A2ADispatch")
A2A_COMBINE = ("internode_ll::combine", "::combine", "ReduceScatter", "reduce_scatter",
               "moeA2ACombineKernel", "A2ACombine")
# expert-GEMM kernels — VERIFIED NVFP4 names from the EP4 trace. The NVFP4 expert
# FFN GEMM shows up as cutlass FP4 (DeviceGemmFp4GemmSm100) and the E2M1 batched
# matmuls (bmm_*E2m1*; E2M1 IS the NVFP4 element format), plus the MoE finalize
# kernel. EXCLUDE nvjet_sm100 / generic cublasLt: those are bf16 cublasLt GEMMs
# serving ATTENTION QKV/O + dense projections, NOT the FP4 expert FFN — counting
# them inflates moe_compute (they dominated the misclassified first pass).
#   NVFP4 -> DeviceGemmFp4 (cutlass) + bmm_*E2m1* (flashinfer FP4 grouped) + moe finalize
#   FP8   -> deep_gemm sm100 fp8 gemm (add when an FP8 trace confirms the symbol)
#   BF16  -> nvjet_sm100 cublasLt (but that also serves attention — treat as upper bound)
EXPERT_KERNELS = (
    "DeviceGemmFp4", "E2m1", "moe::dev::finalize", "finalizeKernel",
    "grouped_gemm_masked", "fp8_fp4_gemm_1d1d",
)
# kernels to EXCLUDE from moe_compute even if a broad pattern would match (attn etc.)
# NB: use "flash_attn"/"flashattn", NOT bare "flash" — "flash" also matches
# "flashinfer", and the flashinfer DeviceGemmFp4 IS a real NVFP4 expert GEMM.
EXCLUDE_KERNELS = ("paged_mqa_logits", "mla", "attention", "fmha",
                   "flash_attn", "flashattn", "rope", "rmsnorm")
CAPTURE_OUTLIER_US = 1000.0


def _p(vals, q):
    vals = sorted(vals)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, int(len(vals) * q))]


def _rank_from_name(path):
    """vLLM trace filename: dp0_pp0_tp0_dcp0_ep0_rank0.<ts>.pt.trace.json.gz."""
    base = os.path.basename(path)
    for pat in (r"rank(\d+)", r"ep(\d+)", r"dp(\d+)", r"TP-(\d+)"):
        m = re.search(pat, base)
        if m:
            return int(m.group(1))
    return -1


def _classify(nm):
    """Map a kernel name to dispatch/combine/moe_compute or None.

    Order matters: combine before dispatch (DeepEP's internode_ll::combine
    contains neither's verb in a way that'd collide, but NCCL ReduceScatter ↔
    AllGather must stay distinct), then the attention exclude-list before the
    EXPERT_KERNELS match so attention GEMMs don't leak into moe_compute."""
    if any(k in nm for k in A2A_COMBINE):
        return "combine"
    if any(k in nm for k in A2A_DISPATCH):
        return "dispatch"
    low = nm.lower()
    if any(x in low for x in EXCLUDE_KERNELS):
        return None
    if any(k in nm for k in EXPERT_KERNELS):
        return "moe_compute"
    return None


def load_torch(paths, phase_hint):
    """vLLM torch chrome trace -> event dicts. Layer inferred by dispatch order."""
    evs = []
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p) as fh:
            data = json.load(fh)
        tr = data.get("traceEvents", data) if isinstance(data, dict) else data
        rank = _rank_from_name(p)
        for e in tr:
            if e.get("cat") != "kernel" or "dur" not in e:
                continue
            op = _classify(e["name"])
            if op is None:
                continue
            evs.append({"op": op, "phase": phase_hint, "layer": None,
                        "rank": rank, "lat_us": e["dur"], "ts_us": e["ts"]})
    return evs


def list_kernels(paths, topn=40):
    """Dump top kernels by total GPU time — read these to fill EXPERT_KERNELS."""
    agg = collections.defaultdict(lambda: [0.0, 0])  # name -> [total_us, count]
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p) as fh:
            data = json.load(fh)
        tr = data.get("traceEvents", data) if isinstance(data, dict) else data
        for e in tr:
            if e.get("cat") != "kernel" or "dur" not in e:
                continue
            a = agg[e["name"]]
            a[0] += e["dur"]; a[1] += 1
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    print(f"# {len(paths)} trace file(s), {len(agg)} distinct kernels")
    print(f"{'total_ms':>10} {'n':>7} {'mean_us':>9}  kernel")
    for nm, (tot, n) in rows[:topn]:
        cls = _classify(nm) or "-"
        print(f"{tot/1000:>10.2f} {n:>7} {tot/max(n,1):>9.2f}  [{cls:>11}] {nm[:90]}")


def per_op(evs):
    by = {}
    for e in evs:
        if e["lat_us"] is None:
            continue
        by.setdefault((e["phase"], e["op"]), []).append(e["lat_us"])
    out = []
    for (ph, op), v in sorted(by.items()):
        clean = [x for x in v if x < CAPTURE_OUTLIER_US]
        use = clean if clean else v
        out.append({"phase": ph, "op": op, "n": len(use),
                    "p50_us": round(_p(use, .5), 2),
                    "mean_us": round(statistics.mean(use), 2),
                    "p90_us": round(_p(use, .9), 2)})
    return out


def flow_spans(evs):
    """Per phase: single-rank flow (dispatch_start->combine_end paired by index)
    and same-node cross-rank span. Identical logic to the sglang analyzer."""
    out = []
    phases = sorted(set(e["phase"] for e in evs if e["phase"]))
    for ph in phases:
        d, c = {}, {}
        for e in evs:
            if e["phase"] != ph or e["ts_us"] is None or e["lat_us"] is None:
                continue
            if e["op"] == "dispatch":
                d.setdefault(e["rank"], []).append((e["ts_us"], e["lat_us"]))
            elif e["op"] == "combine":
                c.setdefault(e["rank"], []).append((e["ts_us"], e["lat_us"]))
        sr = []
        for r in d:
            ds = sorted(x[0] for x in d[r])
            ce = sorted(x[0] + x[1] for x in c.get(r, []))
            for i in range(min(len(ds), len(ce))):
                span = ce[i] - ds[i]
                if 0 < span < CAPTURE_OUTLIER_US * 5:
                    sr.append(span)
        if sr:
            out.append({"phase": ph, "scope": "single_rank", "p50_us": round(_p(sr, .5), 1), "n": len(sr)})
        ranks = sorted(set(d) | set(c))
        if len(ranks) >= 2:
            ndd = min((len(d[r]) for r in d), default=0)
            ncc = min((len(c[r]) for r in c), default=0)
            cr = []
            for i in range(min(ndd, ncc)):
                ds_i = [sorted(x[0] for x in d[r])[i] for r in d if len(d[r]) > i]
                ce_i = [sorted(x[0] + x[1] for x in c[r])[i] for r in c if len(c[r]) > i]
                if ds_i and ce_i:
                    span = max(ce_i) - min(ds_i)
                    if 0 < span < CAPTURE_OUTLIER_US * 10:
                        cr.append(span)
            if cr:
                out.append({"phase": ph, "scope": f"cross_rank_{len(ranks)}r_samenode",
                            "p50_us": round(_p(cr, .5), 1), "n": len(cr)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="vllm")
    ap.add_argument("--torch", nargs="*", default=[])
    ap.add_argument("--list-kernels", nargs="*", default=None,
                    help="glob(s) of traces; dump top kernels by GPU time and exit")
    ap.add_argument("--phase", default="decode", help="phase hint for torch traces")
    ap.add_argument("--csv", help="append rows to this CSV")
    ap.add_argument("--meta", default="", help="config tag, e.g. ep8,nvfp4,bs8")
    a = ap.parse_args()

    if a.list_kernels is not None:
        paths = [p for g in a.list_kernels for p in glob.glob(g)]
        if not paths:
            ap.error("--list-kernels matched no files")
        list_kernels(paths)
        return

    if not a.torch:
        ap.error("need --torch (or --list-kernels)")
    paths = [p for g in a.torch for p in glob.glob(g)]
    evs = load_torch(paths, a.phase)
    src = "torch"
    print(f"# torch: {len(paths)} files, {len(evs)} matched events", file=sys.stderr)

    rows = []
    print(f"\n=== {a.job} [{src}] {a.meta} — per-op latency (us) ===")
    print(f"{'phase':8s} {'op':12s} {'n':>5} {'p50':>9} {'mean':>9} {'p90':>9}")
    for r in per_op(evs):
        print(f"{str(r['phase']):8s} {r['op']:12s} {r['n']:>5} {r['p50_us']:>9} {r['mean_us']:>9} {r['p90_us']:>9}")
        rows.append({"job": a.job, "src": src, "meta": a.meta, "kind": "per_op", **r})
    print(f"\n=== {a.job} [{src}] {a.meta} — flow span (us) ===")
    for r in flow_spans(evs):
        print(f"  {str(r['phase']):8s} {r['scope']:24s} p50={r['p50_us']:>9}  n={r['n']}")
        rows.append({"job": a.job, "src": src, "meta": a.meta, "kind": "flow", **r})

    if not rows:
        print("\nNO MATCHED EVENTS — run --list-kernels to discover the real kernel "
              "names and update EXPERT_KERNELS/A2A_* in this file.", file=sys.stderr)

    if a.csv:
        new = not os.path.exists(a.csv)
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n# appended {len(rows)} rows to {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
