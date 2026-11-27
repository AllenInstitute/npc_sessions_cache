"""Diagnostics for sessions whose sync file was patched (`session.sync_patch`, built by
`npc_sessions.timing`): replayed ('invalid') intervals, where sync data was lost to dropped
acquisition buffers and events were reconstructed from other sources.

The checks here are strict: each one compares the patched events with something measured
independently of the reconstruction, or with a property the recorded data always has.

- `plot_sync_patch_spans`: what was patched, per line and span (status, method, error bound)
- `plot_edges_near_invalid_intervals`: edges as recorded vs as patched, around each interval
- `plot_patched_line_alternation`: rising/falling edges must alternate on every patched line
- `plot_ephys_barcode_residuals`: sync barcodes minus each probe's timing model, in ms
- `plot_trial_onset_frame_consistency`: `stim_start_time - quiescent_stop_time`, per trial:
  ~constant, so a 1-frame (16.7 ms) offset in a reconstructed trial stands out
- `plot_camera_intervals_near_invalid_intervals`: camera exposures must stay periodic through
  the reconstructed spans
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import npc_ephys
import npc_ephys.openephys
import npc_sessions
import npc_sessions.timing
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import npc_sync
    import numpy.typing as npt

logger = logging.getLogger(__name__)

FRAME_SEC = 1 / 60
NEAR_INVALID_PAD_SEC = 0.05
"""Events within this many seconds of an invalid interval count as reconstructed"""
STATUS_COLORS = {"reconstructed": "tab:green", "unknown": "tab:red", "no_edges": "0.6"}
MAX_INTERVALS_PLOTTED = 8
GATE_MS = 0.5
"""npc_ephys per-device gate on the robust barcode residual (AP, NI-DAQ)"""


def _invalid_intervals(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[tuple[float, float], ...]:
    if not session.is_sync:
        return ()
    return tuple(session.sync_data.invalid_intervals)


def _near(
    t: npt.ArrayLike,
    intervals: Sequence[tuple[float, float]],
    pad: float = NEAR_INVALID_PAD_SEC,
) -> npt.NDArray[np.bool_]:
    t = np.asarray(t, dtype=float)
    m = np.zeros(t.shape, dtype=bool)
    for a, b in intervals:
        m |= (t >= a - pad) & (t <= b + pad)
    return m


def _line_arg(sync: npc_sync.SyncDataset, key: str) -> str | int:
    """Patch line key ('<label>' or 'bit<n>') -> argument for `get_rising_edges`"""
    if key.startswith("bit") and key[3:].isdigit() and key not in sync.line_labels:
        return int(key[3:])
    return key


def _edges(
    sync: npc_sync.SyncDataset, key: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    line = _line_arg(sync, key)
    return (
        np.asarray(sync.get_rising_edges(line, units="seconds"), dtype=float),
        np.asarray(sync.get_falling_edges(line, units="seconds"), dtype=float),
    )


def _span_label(s: npc_sync.SpanPatch) -> str:
    if s.status == "reconstructed" and s.method == "no_edges":
        return "no_edges"
    return s.status


def get_patch_span_table(session: npc_sessions.DynamicRoutingSession) -> pd.DataFrame:
    """One row per (line, span) in the session's sync patch."""
    patch = session.sync_patch
    if patch is None:
        return pd.DataFrame()
    rows = []
    for line, spans in patch.lines.items():
        for s in spans:
            d = s.diagnostics or {}
            rows.append(
                dict(
                    line=line,
                    start_time=round(s.span[0], 4),
                    stop_time=round(s.span[1], 4),
                    status=_span_label(s),
                    method=s.method,
                    n_rising=len(s.rising),
                    n_falling=len(s.falling),
                    error_bound_ms=(
                        None
                        if s.error_bound_sec is None
                        else round(1e3 * s.error_bound_sec, 3)
                    ),
                    anchor_residual_ms=(
                        round(1e3 * d["anchor_residual_sec"], 4)
                        if isinstance(d.get("anchor_residual_sec"), (int, float))
                        else None
                    ),
                    reason=s.reason,
                )
            )
    return pd.DataFrame(rows)


def plot_sync_patch_spans(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, str] | None:
    """What the sync patch did, per line: a timeline of every span (green reconstructed, grey
    no edges to reconstruct, red unknown: events on that line are missing there), faults in the
    file (replays, drops, uncorrected timebase discontinuities), and a table of every span with
    its method, error bound and anchor residual."""
    if not session.is_sync:
        return None
    patch = session.sync_patch
    invalid = _invalid_intervals(session)
    if patch is None or not (invalid or patch.faults or patch.timebase):
        return None
    table = get_patch_span_table(session)
    # lines with nothing to reconstruct anywhere (no edges in any span) are listed, not drawn
    empty = sorted(
        k for k, v in patch.lines.items() if all(_span_label(s) == "no_edges" for s in v)
    )
    lines = sorted(set(patch.lines) - set(empty))
    fig, ax = plt.subplots(figsize=(10, 1 + 0.25 * max(len(lines), 1)))
    total = session.sync_data.total_seconds
    for i, line in enumerate(lines):
        for s in patch.lines[line]:
            a, b = s.span
            # spans are ~1-10 s in a ~1-2 h file: draw at least 0.2% of the width
            w = max(b - a, 0.002 * total)
            ax.barh(i, w, left=a, height=0.7, color=STATUS_COLORS[_span_label(s)], lw=0)
    for a, b in invalid:
        ax.axvspan(a, b, color="r", alpha=0.1, lw=0)
    for f in patch.faults:
        ax.axvline(
            f.start,
            c={"replay": "r", "drop": "b", "timebase_discontinuity": "m"}[f.kind],
            lw=0.5,
            ls=":",
        )
    ax.set_yticks(range(len(lines)), lines, fontsize=6)
    ax.set_xlim(0, total)
    ax.set_xlabel("time (s, true sync clock)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in STATUS_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper right")
    ax.set_title(
        f"{session.id}: builder {patch.builder!r}, {len(invalid)} invalid interval(s), "
        f"faults: {pd.Series([f.kind for f in patch.faults]).value_counts().to_dict()}",
        fontsize=8,
    )
    fig.set_layout_engine("tight")

    text = [
        f"builder: {patch.builder}",
        f"invalid intervals (s): {[tuple(round(v, 4) for v in i) for i in invalid]}",
        f"faults: {[(f.kind, round(f.start, 4), round(f.stop, 4), round(f.shift, 6)) for f in patch.faults]}",
        f"timebase: {'yes' if patch.timebase else 'no'}",
        f"timebase discontinuities left after patch: {session.sync_data.timebase_discontinuities}",
        f"summary (spans per line, removed edges): {patch.summary()}",
        f"photodiode blocks: {patch.diode_summary()}",
        f"lines with no edges in any span (not drawn): {empty}",
        "",
    ]
    if patch.notes:
        text += ["notes:", *patch.notes, ""]
    with pd.option_context(
        "display.width", 250, "display.max_columns", None, "display.max_rows", None
    ):
        text.append(table.to_string() if not table.empty else "no spans")
    return fig, "\n".join(text)


def plot_edges_near_invalid_intervals(
    session: npc_sessions.DynamicRoutingSession,
    pad_sec: float = 0.5,
) -> tuple[plt.Figure, ...] | None:
    """For each invalid interval (longest first, up to 8): edges on each patched line,
    as recorded (grey, timebase applied, unpatched) and as patched (black rising, blue
    falling), from `pad_sec` before to `pad_sec` after the interval (shaded).

    - reconstructed edges should continue the rhythm of the measured edges on both sides
      (vsyncs, photodiode, cameras) without a gap, a doubled edge or a 1-frame step
    - lines whose span is 'unknown' (red label) have no patched edges inside the interval
    """
    if not session.is_sync:
        return None
    invalid = _invalid_intervals(session)
    patch = session.sync_patch
    if not invalid or patch is None:
        return None
    raw = npc_sessions.timing.true_clock_view(session._raw_sync_data)
    sync = session.sync_data
    order = sorted(invalid, key=lambda i: i[0] - i[1])[:MAX_INTERVALS_PLOTTED]
    figs = []
    for a, b in sorted(order):
        keys = [
            k
            for k, spans in patch.lines.items()
            if any(s.span[0] <= b and s.span[1] >= a for s in spans)
        ]
        t0, t1 = a - pad_sec, b + pad_sec
        empty = []
        for k in list(keys):
            span = next(s for s in patch.lines[k] if s.span[0] <= b and s.span[1] >= a)
            if _span_label(span) != "no_edges":
                continue
            try:
                n = sum(
                    int(((e >= t0) & (e <= t1)).sum())
                    for view in (raw, sync)
                    for e in _edges(view, k)
                )
            except Exception:
                n = 0
            if n == 0:  # nothing on this line near the interval
                keys.remove(k)
                empty.append(k)
        keys.sort(key=lambda k: (k.startswith("bit"), k))
        fig, ax = plt.subplots(figsize=(10, 1 + 0.35 * max(len(keys), 1)))
        labels = []
        for i, key in enumerate(keys):
            span = next(
                s for s in patch.lines[key] if s.span[0] <= b and s.span[1] >= a
            )
            status = _span_label(span)
            for view, y, colors in (
                (raw, i + 0.2, ("0.5", "0.7")),
                (sync, i - 0.2, ("k", "tab:blue")),
            ):
                try:
                    r, f = _edges(view, key)
                except Exception as exc:  # line not on file
                    logger.debug(f"{session.id} {key}: {exc!r}")
                    continue
                for edges, c in zip((r, f), colors):
                    e = edges[(edges >= t0) & (edges <= t1)]
                    ax.vlines(e, y - 0.15, y + 0.15, color=c, lw=0.5)
            labels.append((key, status))
        ax.axvspan(a, b, color="r", alpha=0.08, lw=0)
        ax.set_yticks(range(len(labels)), [k for k, _ in labels], fontsize=6)
        for tick, (_, status) in zip(ax.get_yticklabels(), labels):
            tick.set_color(
                {"unknown": "tab:red", "no_edges": "0.5"}.get(status, "k")
            )
        ax.set_xlim(t0, t1)
        ax.set_xlabel("time (s, true sync clock)")
        ax.set_title(
            f"{session.id}: invalid interval {a:.4f}-{b:.4f} s ({b - a:.3f} s)\n"
            "per line: top grey = recorded (unpatched), bottom = patched (black rising, blue falling); "
            f"red label = unknown; {len(empty)} line(s) with no edges in the window not shown",
            fontsize=7,
        )
        fig.set_layout_engine("tight")
        figs.append(fig)
    return tuple(figs)


def _alternation_pairs(
    rising: npt.NDArray[np.float64], falling: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """(n, 2) times of adjacent same-kind edges (empty if the line alternates)"""
    t = np.concatenate([rising, falling])
    order = np.argsort(t, kind="stable")
    t = t[order]
    k = np.r_[np.ones(len(rising)), np.zeros(len(falling))][order]
    same = np.flatnonzero(k[1:] == k[:-1])
    return np.c_[t[same], t[same + 1]]


def plot_patched_line_alternation(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[str, dict] | None:
    """Rising and falling edges alternate on every recorded line, so they must on the patched
    lines too. For each line with patch spans: pairs of adjacent same-kind edges (RR or FF),
    split into those touching a reconstructed span (+-50 ms) - a reconstruction error - and
    the rest (in the recorded data; the same count as the unpatched line outside spans).
    Pairs bracketing an 'unknown' span are not counted: edges there are missing by design.
    """
    if not session.is_sync:
        return None
    invalid = _invalid_intervals(session)
    patch = session.sync_patch
    if not invalid or patch is None:
        return None
    sync = session.sync_data
    out: dict[str, Any] = {}
    lines_text = []
    for key, spans in sorted(patch.lines.items()):
        try:
            r, f = _edges(sync, key)
        except Exception as exc:
            out[key] = {"error": repr(exc)}
            continue
        pairs = _alternation_pairs(r, f)
        unknown = [s.span for s in spans if s.status == "unknown"]
        reconstructed = [s.span for s in spans if s.status == "reconstructed"]
        if len(pairs):
            across_unknown = np.array(
                [any(p0 < z and a < p1 for a, z in unknown) for p0, p1 in pairs]
            )
            pairs = pairs[~across_unknown]
        touching = (
            _near(pairs[:, 0], reconstructed) | _near(pairs[:, 1], reconstructed)
            if len(pairs)
            else np.array([], dtype=bool)
        )
        out[key] = {
            "n_spans_reconstructed": len(reconstructed),
            "n_spans_unknown": len(unknown),
            "n_non_alternating_pairs": int(len(pairs)),
            "n_touching_reconstructed_span": int(touching.sum()),
            "first_pairs_touching_span_s": [
                [round(float(x), 5) for x in p] for p in pairs[touching][:10]
            ],
        }
        flag = "FAIL" if touching.any() else "ok"
        lines_text.append(
            f"{flag:4s} {key:28s} spans {len(reconstructed)} reconstructed / {len(unknown)} unknown;"
            f" non-alternating pairs {len(pairs)} ({int(touching.sum())} at a reconstructed span)"
            + (
                f": first at {out[key]['first_pairs_touching_span_s'][:3]}"
                if touching.any()
                else ""
            )
        )
    n_fail = sum(
        v.get("n_touching_reconstructed_span", 0) > 0 for v in out.values()
    )
    text = (
        f"{session.id}: {n_fail} line(s) with non-alternating edges at a reconstructed span\n"
        + "\n".join(lines_text)
    )
    return text, out


def _device_barcodes(
    info: npc_ephys.EphysTimingInfo, nominal_rate: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Barcode start (samples, relative to the first recorded sample) and values"""
    d = info.device
    ch = npc_ephys.openephys.BARCODE_TTL_CHANNEL
    t, ids = npc_ephys.extract_barcodes_from_times(
        on_times=d.ttl_sample_numbers[d.ttl_states == ch] / nominal_rate,
        off_times=d.ttl_sample_numbers[d.ttl_states == -ch] / nominal_rate,
        total_time_on_line=d.ttl_sample_numbers[-1] / nominal_rate,
    )
    return np.asarray(t) * nominal_rate, np.asarray(ids)


def plot_ephys_barcode_residuals(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, dict] | None:
    """Sync barcode time minus the device's timing model (`to_sync_time`) at the same barcode,
    per probe, in ms, vs time. Barcodes are matched by value, so a misassigned barcode or a step
    in the model shows as an outlier.

    - residuals should be flat and within the npc_ephys gate (+-0.5 ms, dashed; typically
      < 0.1 ms) everywhere, including next to invalid intervals (red) and the device's own
      dropped samples (dotted)
    - sync barcodes inside (padded) invalid intervals are not used, as for the timing fit
    """
    if not (session.is_sync and session.is_ephys):
        return None
    nominal = 30000.0
    sync_t, sync_ids = npc_ephys.openephys._sync_barcodes(session.sync_data)
    sync_t, sync_ids = np.asarray(sync_t), np.asarray(sync_ids)
    invalid = _invalid_intervals(session)
    devices = [
        i
        for i in session.ephys_timing_qc
        if "NI-DAQmx" not in i.device.name and "LFP" not in i.device.name
    ]
    if not devices:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    summary: dict[str, dict] = {}
    for info in devices:
        name = info.device.name.split(".")[-1]
        if not getattr(info, "is_valid", True):
            summary[name] = {"status": info.status, "reason": info.reason}
            continue
        dev_s, dev_ids = _device_barcodes(info, nominal)
        common, i_sync, i_dev = np.intersect1d(sync_ids, dev_ids, return_indices=True)
        if not common.size:
            summary[name] = {"status": info.status, "error": "no barcodes matched"}
            continue
        t = sync_t[i_sync]
        res = (t - info.to_sync_time(dev_s[i_dev])) * 1e3
        order = np.argsort(t)
        t, res = t[order], res[order]
        near = _near(t, invalid, pad=60)
        (line,) = ax.plot(t, res, ".-", ms=2, lw=0.5, label=name)
        for drop_idx, _ in info.drops:
            ax.axvline(
                float(info.to_sync_time([drop_idx])[0]), c=line.get_color(), lw=0.5, ls=":"
            )
        summary[name] = {
            "status": info.status,
            "n_barcodes_matched": int(common.size),
            "n_sync_barcodes": int(sync_ids.size),
            "n_device_barcodes": int(dev_ids.size),
            "max_abs_residual_ms": float(np.max(np.abs(res))),
            "max_abs_residual_ms_within_60s_of_invalid": (
                float(np.max(np.abs(res[near]))) if near.any() else None
            ),
            "n_drops": len(info.drops),
            "npc_ephys_qc_max_residual_ms": info.qc_metrics.get("max_residual_ms"),
        }
    for a, b in invalid:
        ax.axvspan(a, b, color="r", alpha=0.3, lw=0)
    for y in (-GATE_MS, GATE_MS):
        ax.axhline(y, c="k", lw=0.5, ls="--")
    ax.set_xlabel("sync time (s)")
    ax.set_ylabel("sync - model (ms)")
    ax.legend(fontsize=6, ncol=6)
    worst = max(
        (v.get("max_abs_residual_ms", 0) for v in summary.values()), default=0
    )
    ax.set_title(
        f"{session.id}: barcode residuals, max |res| {worst:.3f} ms (gate {GATE_MS} ms); "
        f"red: invalid intervals, dotted: device sample drops",
        fontsize=8,
    )
    fig.set_layout_engine("tight")
    return fig, summary


def plot_trial_onset_frame_consistency(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure, dict] | None:
    """`stim_start_time - quiescent_stop_time` for every vis and aud task trial, vs time.

    `quiescent_stop_time` is the vsync of the frame before the stim frame; `stim_start_time`
    is the display time of the stim frame (vis) or the measured sound onset (aud, from the
    sound line / NI-DAQ, independent of the frame clock). The difference is ~constant (one frame
    plus a latency), so a 1-frame error in a reconstructed vsync or display time shifts it by
    16.7 ms.

    - red points: trials with an onset inside an invalid interval (+-50 ms); they should be
      within half a frame of the median (grey band)
    - other trials off by a frame (e.g. a vis stim frame displayed late) are counted for
      comparison: if they're common, a replayed trial off by a frame is weaker evidence
    """
    if not (session.is_sync and session.is_task):
        return None
    trials = session.trials[:]
    if "quiescent_stop_time" not in trials:
        return None
    invalid = _invalid_intervals(session)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    summary: dict[str, dict] = {}
    for ax, modality in zip(axes, ("vis", "aud")):
        sel = trials[trials[f"is_{modality}_stim"].astype(bool)]
        if "is_opto" in sel:
            sel = sel[~sel["is_opto"].astype(bool)]
        d = 1e3 * (sel["stim_start_time"] - sel["quiescent_stop_time"]).to_numpy(float)
        t = sel["stim_start_time"].to_numpy(float)
        ok = np.isfinite(d)
        rep = _near(t, invalid) & ok
        other = d[ok & ~rep]
        if not other.size:
            ax.set_title(f"{modality}: no trials", fontsize=7)
            continue
        med = float(np.median(other))
        half = 1e3 * FRAME_SEC / 2
        lo, hi = med - half, med + half
        ax.axhspan(lo, hi, color="0.85", lw=0)
        ax.axhline(med, c="k", lw=0.5)
        for y in (med - 1e3 * FRAME_SEC, med + 1e3 * FRAME_SEC):
            ax.axhline(y, c="b", lw=0.5, ls=":")
        ax.scatter(t[ok & ~rep], d[ok & ~rep], s=2, c="k")
        ax.scatter(t[rep], d[rep], s=15, c="r", zorder=3)
        for a, b in invalid:
            ax.axvspan(a, b, color="r", alpha=0.3, lw=0)
        out_of_band = rep & ((d < lo) | (d > hi))
        n_other_off = int(np.sum((other < lo) | (other > hi)))
        summary[modality] = {
            "n_trials": int(ok.sum()),
            "n_nan": int((~ok).sum()),
            "other_trials_median_ms": med,
            "other_trials_ms_p0.5_p99.5": [float(v) for v in np.percentile(other, [0.5, 99.5])],
            "n_other_trials_off_by_half_frame_or_more": n_other_off,
            "replayed_trials": [
                {
                    "trial_index": int(i),
                    "stim_start_time": float(tt),
                    "d_ms": float(dd),
                    "in_band": bool(lo <= dd <= hi),
                }
                for i, tt, dd in zip(sel.index[rep], t[rep], d[rep])
            ],
            "n_replayed_out_of_band": int(out_of_band.sum()),
        }
        ax.set_ylabel(f"{modality}: stim_start -\nquiescent_stop (ms)")
        ax.set_title(
            f"{modality}: median {med:.2f} ms (band: +-half frame), {n_other_off} of {other.size} other trials outside; "
            f"{int(rep.sum())} trial(s) in invalid intervals, {int(out_of_band.sum())} outside"
            f"; dotted: +-1 frame",
            fontsize=7,
        )
        # outliers far from the band would squash it: clip the view
        ax.set_ylim(med - 1e3 * FRAME_SEC - 8, med + 1e3 * FRAME_SEC + 8)
    axes[-1].set_xlabel("stim_start_time (s)")
    fig.suptitle(f"{session.id}: stim onset vs preceding frame", fontsize=8)
    fig.set_layout_engine("tight")
    return fig, summary


def plot_camera_intervals_near_invalid_intervals(
    session: npc_sessions.DynamicRoutingSession,
    pad_sec: float = 2.0,
) -> tuple[plt.Figure, dict] | None:
    """Camera exposure intervals (patched `*_cam_exposing` rising edges) / median period,
    from `pad_sec` before to `pad_sec` after each invalid interval, aligned to the interval
    start. Exposures inside the intervals are interpolated by the patch, so every interval
    should be 1 period (dotted lines at 0.9 and 1.1), with no skip or doubled exposure where
    measured and interpolated edges meet."""
    invalid = _invalid_intervals(session)
    if not invalid:
        return None
    sync = session.sync_data
    lines = [lab for lab in sync.line_labels if lab.endswith("_cam_exposing")]
    if not lines:
        return None
    fig, axes = plt.subplots(
        len(lines), 1, figsize=(8, 1.8 * len(lines)), sharex=True, squeeze=False
    )
    summary: dict[str, dict] = {}
    for ax, line in zip(axes[:, 0], lines):
        try:
            r = np.asarray(sync.get_rising_edges(line, units="seconds"))
        except Exception as exc:
            ax.set_title(f"{line}: {exc!r}", fontsize=6)
            continue
        unknown = sync.get_unknown_intervals(line)
        if r.size < 3:
            continue
        period = np.median(np.diff(r))
        n_bad = 0
        for a, b in invalid:
            e = r[(r >= a - pad_sec) & (r <= b + pad_sec)]
            if e.size < 2:
                continue
            ratio = np.diff(e) / period
            mid = e[:-1] + np.diff(e) / 2
            bad = np.abs(ratio - 1) > 0.1
            if unknown:  # edges are missing by design in unknown spans
                bad &= ~_near(mid, unknown, pad=period)
            n_bad += int(bad.sum())
            ax.plot(mid - a, ratio, ".-", ms=1.5, lw=0.3)
            ax.scatter((mid - a)[bad], ratio[bad], s=10, c="r", zorder=3)
        for y in (0.9, 1.1):
            ax.axhline(y, c="k", lw=0.5, ls=":")
        ax.axvline(0, c="r", lw=0.5)
        summary[line] = {
            "median_period_ms": float(1e3 * period),
            "n_irregular_intervals_near_invalid": n_bad,
            "n_unknown_spans": len(unknown),
        }
        ax.set_ylabel("interval /\nperiod", fontsize=6)
        ax.set_title(
            f"{line}: period {1e3 * period:.2f} ms, {n_bad} irregular interval(s) near invalid intervals"
            + (f", {len(unknown)} unknown span(s)" if unknown else ""),
            fontsize=7,
        )
    axes[-1, 0].set_xlabel("time from start of invalid interval (s)")
    fig.suptitle(
        f"{session.id}: camera exposures around {len(invalid)} invalid interval(s)",
        fontsize=8,
    )
    fig.set_layout_engine("tight")
    return fig, summary
