# SPDX-License-Identifier: Apache-2.0
"""Phase + layer-index helpers for moe_trace_logger instrumentation.

These helpers wrap reads into the live ``ForwardContext`` so the
instrumentation sites in the A2A backends and the modular MoE kernel
remain a one-liner.  They are intentionally defensive: a missing
forward context, a missing attn metadata or an out-of-bounds counter
must never raise (instrumentation is best-effort).
"""

from __future__ import annotations

from typing import Optional


def phase_of_current_forward() -> str:
    """Best-effort prefill/decode classifier for the current forward pass.

    At vLLM v0.21.0 there is no single "phase" flag on ``ForwardContext``.
    We inspect the per-layer attention metadata (a ``dict[layer_name, meta]``)
    and use ``max_query_len`` as the discriminator:

    * ``max_query_len == 1`` across every metadata → ``"decode"``
    * ``max_query_len > 1`` everywhere → ``"prefill"``
    * mixed (some metadatas at 1, some > 1) → ``"mixed"``  (chunked prefill)
    * cannot classify → ``"unknown"``
    """
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        return "unknown"

    if not is_forward_context_available():
        return "unknown"

    try:
        ctx = get_forward_context()
    except Exception:
        return "unknown"

    attn_metadata = getattr(ctx, "attn_metadata", None)
    if attn_metadata is None:
        return "unknown"

    # DBO/microbatched path: attn_metadata can be a list of dicts.
    metas: list = []
    if isinstance(attn_metadata, list):
        for elem in attn_metadata:
            if isinstance(elem, dict):
                metas.extend(elem.values())
            elif elem is not None:
                metas.append(elem)
    elif isinstance(attn_metadata, dict):
        metas.extend(attn_metadata.values())
    else:
        metas.append(attn_metadata)

    seen_prefill = False
    seen_decode = False
    for m in metas:
        if m is None:
            continue
        mql = getattr(m, "max_query_len", None)
        if mql is None:
            continue
        if mql == 1:
            seen_decode = True
        elif mql > 1:
            seen_prefill = True

    if seen_prefill and seen_decode:
        return "mixed"
    if seen_prefill:
        return "prefill"
    if seen_decode:
        return "decode"
    return "unknown"


def layer_index_of_current_moe() -> int:
    """Return the MoE layer index for the call currently in flight.

    ``ForwardContext.moe_layer_index`` is bumped by ``get_layer_from_name``
    when the custom op resolves the FusedMoE layer for this call, so by
    the time prepare/finalize runs the relevant index is ``index - 1``.
    Falls back to ``-1`` when nothing can be determined (e.g. no forward
    context, no static MoE layer registry).
    """
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        return -1

    if not is_forward_context_available():
        return -1

    try:
        ctx = get_forward_context()
    except Exception:
        return -1

    all_moe_layers = getattr(ctx, "all_moe_layers", None)
    moe_idx = getattr(ctx, "moe_layer_index", None)
    if all_moe_layers and moe_idx is not None and moe_idx > 0:
        layer_name = all_moe_layers[moe_idx - 1]
        idx = _extract_layer_index(layer_name)
        if idx is not None:
            return idx
        return moe_idx - 1

    if moe_idx is not None:
        # No registry available; use the (post-increment) counter as a
        # best-effort monotonic id.  At call time the counter has *just*
        # been bumped, so subtract one when positive.
        return max(0, moe_idx - 1)
    return -1


def _extract_layer_index(layer_name: Optional[str]) -> Optional[int]:
    if not layer_name:
        return None
    int_vals: list[int] = []
    for sub in str(layer_name).split("."):
        try:
            int_vals.append(int(sub))
        except ValueError:
            continue
    if not int_vals:
        return None
    # Same convention as vllm.model_executor.models.utils.extract_layer_index
    return int_vals[0]
