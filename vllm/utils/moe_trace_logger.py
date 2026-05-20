"""MoE A2A and compute tracing logger (vendored, JSONL per-rank).

Records the fields enumerated in the cross-node-moe research plan: for each
A2A operation — op_type (dispatch/combine), phase (prefill/decode), layer,
expert, from_rank, to_rank, payload_bytes, latency, timestamp; for each MoE
compute span — kernel name, layer, phase, latency, timestamp.

Event kinds emitted:
    "a2a"          one event per dispatch/combine call (call level)
    "a2a_peer"     one event per (src, dst) pair within a call (peer level)
    "moe_compute"  one event per MoE compute kernel invocation

Per-peer events inherit op_type/phase/layer/backend/latency_ms from the
enclosing time_a2a span — call span.emit_peer(...) inside the with-block.

Disabled by default. Enable with:
    MOE_TRACE_ENABLE=1
Configure with:
    MOE_TRACE_DIR=/abs/path        # default: ./moe_traces
    MOE_TRACE_FLUSH_EVERY=1024     # auto-flush after this many CUDA spans

Each rank writes one file: {MOE_TRACE_DIR}/moe_trace_rank{N}.jsonl

Typical use at an instrumentation site:

    from vllm.utils.moe_trace_logger import get_moe_trace_logger

    tracer = get_moe_trace_logger()
    with tracer.time_a2a(op_type="dispatch", layer=layer_id,
                         phase="prefill", payload_bytes=nbytes,
                         backend="deepep_high_throughput") as span:
        out = a2a_dispatch(...)
        span.to_rank = peer_rank
        # optional per-peer breakdown (only when peer info is CPU-side and cheap)
        for peer, peer_bytes in peer_table.items():
            span.emit_peer(to_rank=peer, payload_bytes=peer_bytes)

    with tracer.time_moe_compute(kernel="grouped_gemm",
                                  layer=layer_id, phase="prefill"):
        out = grouped_gemm(...)

When MOE_TRACE_ENABLE != "1" the returned object is a no-op stand-in, so
instrumentation can stay in place at zero runtime cost.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


__all__ = ["get_moe_trace_logger", "MoETraceLogger"]


_SINGLETON: Any = None
_SINGLETON_LOCK = threading.Lock()


def _detect_rank() -> int:
    if torch is not None:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
    for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(var)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


class _A2ASpan:
    """Mutable handle yielded by time_a2a. Callers may set to_rank / expert /
    payload_bytes for fields known only after the op starts, and may call
    emit_peer() to queue per-peer events that inherit the span's latency."""

    __slots__ = ("event", "to_rank", "expert", "payload_bytes",
                 "_events_list", "_from_rank")

    def __init__(self, event: dict, events_list: Optional[list] = None) -> None:
        self.event = event
        self.to_rank: Optional[int] = None
        self.expert: Optional[Any] = None
        self.payload_bytes: Optional[int] = None
        self._events_list = events_list
        self._from_rank = event.get("from_rank")

    def emit_peer(
        self,
        *,
        to_rank: int,
        payload_bytes: Optional[int] = None,
        expert: Optional[Any] = None,
    ) -> None:
        """Queue a per-peer event tied to this A2A call.

        The peer event inherits op_type/phase/layer/backend from the enclosing
        time_a2a span; latency_ms is stamped when the span finalizes. No-op
        when the span was produced by the noop logger.
        """
        if self._events_list is None:
            return
        parent = self.event
        self._events_list.append({
            "kind": "a2a_peer",
            "op_type": parent.get("op_type"),
            "phase": parent.get("phase"),
            "layer": parent.get("layer"),
            "from_rank": self._from_rank,
            "to_rank": to_rank,
            "expert": expert,
            "payload_bytes": payload_bytes,
            "backend": parent.get("backend"),
            "ts_ns": time.time_ns(),
        })


class MoETraceLogger:
    def __init__(self, output_path: str, rank: int, flush_every: int = 1024):
        self.output_path = output_path
        self.rank = rank
        self.flush_every = max(1, flush_every)
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fp = open(output_path, "a", buffering=1)
        self._lock = threading.Lock()
        # Each pending entry: (start_event, end_event, [evt, peer_evt, ...])
        # All events in the inner list get stamped with the same latency_ms
        # (the CUDA-event-measured wall time of the span).
        self._pending: list[tuple[Any, Any, list]] = []
        self._cuda_ok = torch is not None and torch.cuda.is_available()
        atexit.register(self.close)

    # ---------- synchronous recording (caller already has latency) ----------

    def record_a2a(
        self,
        *,
        op_type: str,
        layer: int,
        phase: str,
        payload_bytes: Optional[int] = None,
        latency_ms: Optional[float] = None,
        to_rank: Optional[int] = None,
        expert: Optional[Any] = None,
        backend: Optional[str] = None,
    ) -> None:
        self._write({
            "kind": "a2a",
            "op_type": op_type,
            "phase": phase,
            "layer": layer,
            "expert": expert,
            "from_rank": self.rank,
            "to_rank": to_rank,
            "payload_bytes": payload_bytes,
            "latency_ms": latency_ms,
            "backend": backend,
            "ts_ns": time.time_ns(),
        })

    def record_a2a_peer(
        self,
        *,
        op_type: str,
        layer: int,
        phase: str,
        to_rank: int,
        payload_bytes: Optional[int] = None,
        latency_ms: Optional[float] = None,
        expert: Optional[Any] = None,
        backend: Optional[str] = None,
    ) -> None:
        self._write({
            "kind": "a2a_peer",
            "op_type": op_type,
            "phase": phase,
            "layer": layer,
            "expert": expert,
            "from_rank": self.rank,
            "to_rank": to_rank,
            "payload_bytes": payload_bytes,
            "latency_ms": latency_ms,
            "backend": backend,
            "ts_ns": time.time_ns(),
        })

    def record_moe_compute(
        self,
        *,
        kernel: str,
        layer: int,
        phase: str,
        latency_ms: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> None:
        evt = {
            "kind": "moe_compute",
            "kernel": kernel,
            "phase": phase,
            "layer": layer,
            "from_rank": self.rank,
            "latency_ms": latency_ms,
            "ts_ns": time.time_ns(),
        }
        if extra:
            evt.update(extra)
        self._write(evt)

    # ---------- CUDA-event-timed spans (deferred sync) ----------

    @contextmanager
    def time_a2a(
        self,
        *,
        op_type: str,
        layer: int,
        phase: str,
        payload_bytes: Optional[int] = None,
        to_rank: Optional[int] = None,
        expert: Optional[Any] = None,
        backend: Optional[str] = None,
    ):
        evt = {
            "kind": "a2a",
            "op_type": op_type,
            "phase": phase,
            "layer": layer,
            "expert": expert,
            "from_rank": self.rank,
            "to_rank": to_rank,
            "payload_bytes": payload_bytes,
            "backend": backend,
            "ts_ns": time.time_ns(),
        }
        events: list[dict] = [evt]
        span = _A2ASpan(evt, events)

        if not self._cuda_ok:
            t0 = time.perf_counter_ns()
            try:
                yield span
            finally:
                self._finalize_a2a_span(span)
                latency = (time.perf_counter_ns() - t0) / 1e6
                with self._lock:
                    for e in events:
                        e["latency_ms"] = latency
                        self._fp.write(json.dumps(e) + "\n")
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield span
        finally:
            end.record()
            self._finalize_a2a_span(span)
            with self._lock:
                self._pending.append((start, end, events))
                if len(self._pending) >= self.flush_every:
                    self._drain_locked()

    @contextmanager
    def time_moe_compute(
        self,
        *,
        kernel: str,
        layer: int,
        phase: str,
        extra: Optional[dict] = None,
    ):
        evt = {
            "kind": "moe_compute",
            "kernel": kernel,
            "phase": phase,
            "layer": layer,
            "from_rank": self.rank,
            "ts_ns": time.time_ns(),
        }
        if extra:
            evt.update(extra)

        if not self._cuda_ok:
            t0 = time.perf_counter_ns()
            try:
                yield evt
            finally:
                evt["latency_ms"] = (time.perf_counter_ns() - t0) / 1e6
                self._write(evt)
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield evt
        finally:
            end.record()
            with self._lock:
                self._pending.append((start, end, [evt]))
                if len(self._pending) >= self.flush_every:
                    self._drain_locked()

    # ---------- lifecycle ----------

    def flush(self) -> None:
        with self._lock:
            self._drain_locked()
            self._fp.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if not self._fp.closed:
                self._fp.close()

    # ---------- internals ----------

    @staticmethod
    def _finalize_a2a_span(span: _A2ASpan) -> None:
        evt = span.event
        if span.to_rank is not None:
            evt["to_rank"] = span.to_rank
        if span.expert is not None:
            evt["expert"] = span.expert
        if span.payload_bytes is not None:
            evt["payload_bytes"] = span.payload_bytes

    def _drain_locked(self) -> None:
        for start, end, events in self._pending:
            end.synchronize()
            latency = start.elapsed_time(end)
            for evt in events:
                evt["latency_ms"] = latency
                self._fp.write(json.dumps(evt) + "\n")
        self._pending.clear()

    def _write(self, evt: dict) -> None:
        with self._lock:
            self._fp.write(json.dumps(evt) + "\n")


class _NoopLogger:
    """Returned when MOE_TRACE_ENABLE != "1". Cheap no-ops so instrumentation
    sites can call into the tracer unconditionally."""

    rank = -1

    def record_a2a(self, **_kwargs) -> None:
        pass

    def record_a2a_peer(self, **_kwargs) -> None:
        pass

    def record_moe_compute(self, **_kwargs) -> None:
        pass

    @contextmanager
    def time_a2a(self, **_kwargs):
        yield _A2ASpan({}, None)

    @contextmanager
    def time_moe_compute(self, **_kwargs):
        yield {}

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def get_moe_trace_logger():
    """Process-wide singleton. Safe to call from any thread or any rank."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            return _SINGLETON
        if os.environ.get("MOE_TRACE_ENABLE", "0") != "1":
            _SINGLETON = _NoopLogger()
        else:
            out_dir = os.environ.get("MOE_TRACE_DIR", "./moe_traces")
            flush_every = int(os.environ.get("MOE_TRACE_FLUSH_EVERY", "1024"))
            rank = _detect_rank()
            path = os.path.join(out_dir, f"moe_trace_rank{rank}.jsonl")
            _SINGLETON = MoETraceLogger(path, rank, flush_every=flush_every)
        return _SINGLETON
