"""QC of long (dropped) frames in the task block: stim file vs npc_sessions display times.

The stim file's `frameIntervals` are measured on the stim computer's clock, which isn't
synced with the sync clock, so frame *times* from `cumsum(frameIntervals)` drift away from
the display times on sync. Nothing here uses cumulative times: long frames are compared by
their position (frame index) and by their own length (one interval), each measured in units
of the median refresh period of its own clock. A clock-rate difference of tens of ppm
changes a 50 ms interval by ~1 us, so it can't be mistaken for a 1-frame (~16.7 ms)
deviation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Any

import matplotlib.pyplot as plt
import npc_sessions
import npc_stim
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

LONG_FRAME_THRESHOLD = 1.5
"""An interval is a long frame if it's longer than this many median refresh periods (as
`npc_stim.display_times.LONG_FRAME_THRESHOLD`)"""
MAX_GLOBAL_OFFSET_FRAMES = 600
"""Largest systematic index offset searched for between the two sets of long frames"""

__all__ = ["plot_task_long_frame_alignment", "get_long_frame_alignment"]


@dataclasses.dataclass
class LongFrameAlignment:
    n_stim_frames: int
    """frames in the stim file: `len(frameIntervals) + 1`"""
    n_display_frames: int
    stim_period: float
    display_period: float
    stim_intervals: npt.NDArray[np.float64]
    display_intervals: npt.NDArray[np.float64]
    stim_long: npt.NDArray[np.int64]
    """indices of long intervals in `frameIntervals` (interval k is frame k -> k+1)"""
    display_long: npt.NDArray[np.int64]
    global_offset: int
    """most common index offset (display - stim) between nearby long frames"""
    pair_stim: npt.NDArray[np.int64]
    pair_display: npt.NDArray[np.int64]
    stim_only: npt.NDArray[np.int64]
    display_only: npt.NDArray[np.int64]

    @property
    def index_offset(self) -> npt.NDArray[np.int64]:
        return self.pair_display - self.pair_stim

    @property
    def length_diff(self) -> npt.NDArray[np.float64]:
        """display interval - stim-file interval (s), for paired long frames"""
        return self.display_intervals[self.pair_display] - self.stim_intervals[self.pair_stim]

    @property
    def refresh_diff(self) -> npt.NDArray[np.int64]:
        """difference in length, in whole refreshes (each in its own clock's period)"""
        return np.round(self.display_intervals[self.pair_display] / self.display_period).astype(
            int
        ) - np.round(self.stim_intervals[self.pair_stim] / self.stim_period).astype(int)

    @property
    def is_exact(self) -> npt.NDArray[np.bool_]:
        return (self.index_offset == 0) & (self.refresh_diff == 0)

    @property
    def n_deviations(self) -> int:
        return int((~self.is_exact).sum() + len(self.stim_only) + len(self.display_only))

    @property
    def passed(self) -> bool:
        return self.n_stim_frames == self.n_display_frames and self.n_deviations == 0


def _mutual_nearest(a: npt.NDArray, b: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Indices (into a, into b) of pairs that are each other's nearest neighbour."""
    if not len(a) or not len(b):
        return np.array([], dtype=int), np.array([], dtype=int)

    def nearest(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        i = np.clip(np.searchsorted(y, x), 1, len(y) - 1) if len(y) > 1 else np.zeros(len(x), int)
        if len(y) == 1:
            return i
        left, right = y[i - 1], y[i]
        return np.where(np.abs(x - left) <= np.abs(right - x), i - 1, i)

    a_to_b = nearest(a, b)
    b_to_a = nearest(b, a)
    ia = np.flatnonzero(b_to_a[a_to_b] == np.arange(len(a)))
    return ia, a_to_b[ia]


def get_long_frame_alignment(
    stim_frame_intervals: npt.ArrayLike, display_times: npt.ArrayLike
) -> LongFrameAlignment:
    """Pair long frames in the stim file's `frameIntervals` with long frames in the display
    times (one-to-one, mutual nearest index after removing the most common index offset),
    without using cumulative times."""
    s = np.asarray(stim_frame_intervals, dtype=float)
    d = np.diff(np.asarray(display_times, dtype=float))
    s_period, d_period = float(np.nanmedian(s)), float(np.nanmedian(d))
    s_long = np.flatnonzero(s > LONG_FRAME_THRESHOLD * s_period)
    d_long = np.flatnonzero(d > LONG_FRAME_THRESHOLD * d_period)
    global_offset = 0
    if len(s_long) and len(d_long):
        diffs = (d_long[None, :] - s_long[:, None]).ravel() if len(s_long) * len(d_long) < 2e7 else np.array([])
        diffs = diffs[np.abs(diffs) <= MAX_GLOBAL_OFFSET_FRAMES]
        if len(diffs):
            values, counts = np.unique(diffs, return_counts=True)
            # prefer 0 on ties
            global_offset = int(values[np.lexsort((np.abs(values), -counts))[0]])
    ia, ib = _mutual_nearest(s_long + global_offset, d_long)
    return LongFrameAlignment(
        n_stim_frames=len(s) + 1,
        n_display_frames=len(d) + 1,
        stim_period=s_period,
        display_period=d_period,
        stim_intervals=s,
        display_intervals=d,
        stim_long=s_long,
        display_long=d_long,
        global_offset=global_offset,
        pair_stim=s_long[ia],
        pair_display=d_long[ib],
        stim_only=np.setdiff1d(s_long, s_long[ia]),
        display_only=np.setdiff1d(d_long, d_long[ib]),
    )


def _offset_colors(offset: npt.NDArray) -> list[str]:
    return ["0.6" if o == 0 else "tab:orange" if abs(o) == 1 else "tab:red" for o in offset]


def _alignment_figure(
    a: LongFrameAlignment,
    title: str,
    vsync_intervals: npt.NDArray | None = None,
    frame_source: npt.NDArray | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.3, 1.1, 0.35, 1.2])
    ax_idx = fig.add_subplot(gs[0, :])
    ax_len = fig.add_subplot(gs[1, :], sharex=ax_idx)
    ax_un = fig.add_subplot(gs[2, :], sharex=ax_idx)
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_sc = fig.add_subplot(gs[3, 1])
    ax_txt = fig.add_subplot(gs[3, 2])
    ax_txt.axis("off")
    period_ms = a.stim_period * 1e3
    x = a.pair_stim
    offset = a.index_offset
    rdiff = a.refresh_diff
    ldiff_ms = a.length_diff * 1e3
    exact = offset == 0

    # 1. index offset of each paired long frame
    ax_idx.axhline(0, color="k", lw=0.5)
    for v in (-1, 1):
        ax_idx.axhline(v, color="0.6", lw=0.5, ls=":")
    ax_idx.scatter(x[exact], offset[exact], s=6, c="0.6", lw=0, label=f"same index ({int(exact.sum())})")
    for sel, color, label in (
        (np.abs(offset) == 1, "tab:orange", "off by 1 frame"),
        (np.abs(offset) > 1, "tab:red", "off by > 1 frame"),
    ):
        if sel.any():
            ax_idx.scatter(x[sel], offset[sel], s=22, c=color, edgecolor="k", lw=0.4, zorder=3, label=f"{label} ({int(sel.sum())})")
    ax_idx.set_yscale("symlog", linthresh=2, linscale=1.5)
    lim = max(3, int(np.abs(offset).max(initial=0)) + 1)
    ax_idx.set_ylim(-lim * 1.5, lim * 1.5)
    ticks = [0, 1, 2, -1, -2] + [s * 10**e for e in range(1, len(str(lim))) for s in (1, -1) if 10**e <= lim]
    ax_idx.set_yticks(sorted(ticks), [f"{t:+d}" if t else "0" for t in sorted(ticks)])
    ax_idx.yaxis.set_minor_locator(plt.NullLocator())
    ax_idx.set_ylabel("index offset (frames)\ndisplay times - stim file")
    ax_idx.set_title(
        "position: each long frame in the stim file paired with the nearest long frame in the display times "
        "(a systematic offset shifts every dot off 0)",
        fontsize=8,
    )
    ax_idx.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1), borderaxespad=0)
    ax_idx.tick_params(labelbottom=False)

    # 2. length difference of each paired long frame
    for v in (-period_ms, period_ms):
        ax_len.axhline(v, color="0.6", lw=0.5, ls=":")
    for v in (-period_ms / 2, period_ms / 2):
        ax_len.axhline(v, color="tab:red", lw=0.5, ls="--")
    ax_len.axhline(0, color="k", lw=0.5)
    same = rdiff == 0
    ax_len.scatter(x[same], ldiff_ms[same], s=6, c="0.6", lw=0, label=f"same length ({int(same.sum())})")
    if (~same).any():
        ax_len.scatter(
            x[~same], ldiff_ms[~same], s=22, c="tab:red", edgecolor="k", lw=0.4, zorder=3,
            label=f"length differs by >= 1 refresh ({int((~same).sum())})",
        )
    span = max(1.3 * period_ms, 1.1 * np.abs(ldiff_ms).max(initial=0))
    ax_len.set_ylim(-span, span)
    ax_len.set_ylabel("length difference (ms)\ndisplay times - stim file")
    ax_len.set_title(
        f"length of each paired long frame (dotted: +-1 refresh = {period_ms:.1f} ms; dashed: half a refresh)",
        fontsize=8,
    )
    ax_len.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1), borderaxespad=0)
    ax_len.tick_params(labelbottom=False)

    # 3. long frames without a partner
    ax_un.set_yticks([0, 1], ["only in display times", "only in stim file"], fontsize=7)
    ax_un.set_ylim(-0.6, 1.6)
    if len(a.stim_only):
        ax_un.plot(a.stim_only, np.ones(len(a.stim_only)), "|", ms=12, mew=1.5, color="tab:red")
    if len(a.display_only):
        ax_un.plot(a.display_only, np.zeros(len(a.display_only)), "|", ms=12, mew=1.5, color="tab:red")
    ax_un.set_xlabel("frame index in stim file")
    ax_un.set_xlim(0, max(a.n_stim_frames, a.n_display_frames))
    ax_un.set_title(
        f"unpaired long frames: {len(a.stim_only)} only in stim file, {len(a.display_only)} only in display times",
        fontsize=8,
    )
    top = ax_idx.secondary_xaxis(
        "top", functions=(lambda f: f * a.stim_period / 60, lambda m: m * 60 / a.stim_period)
    )
    top.set_xlabel("minutes into block (approx.)", fontsize=7)
    top.tick_params(labelsize=7)

    # 4. histogram of index offsets
    if len(offset):
        values, counts = np.unique(offset, return_counts=True)
        ax_hist.bar(range(len(values)), counts, color=_offset_colors(values))
        ax_hist.set_xticks(range(len(values)), [f"{v:+d}" if v else "0" for v in values], fontsize=7)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("index offset (frames)")
    ax_hist.set_ylabel("paired long frames")
    ax_hist.set_title("index offsets", fontsize=8)

    # 5. lengths of paired long frames
    lim_ms = 1e3 * max(
        a.stim_intervals[a.stim_long].max(initial=0), a.display_intervals[a.display_long].max(initial=0)
    ) * 1.05
    lo = LONG_FRAME_THRESHOLD * period_ms * 0.9
    ax_sc.plot([lo, lim_ms], [lo, lim_ms], "k:", lw=0.5)
    ax_sc.scatter(
        a.stim_intervals[a.pair_stim] * 1e3, a.display_intervals[a.pair_display] * 1e3,
        s=[6 if (o == 0 and r == 0) else 22 for o, r in zip(offset, rdiff)],
        c=["tab:red" if r else c for r, c in zip(rdiff, _offset_colors(offset))],
        lw=0.3, edgecolor="k",
    )
    ax_sc.set_xlim(lo, lim_ms)
    ax_sc.set_ylim(lo, lim_ms)
    ax_sc.set_xlabel("stim file interval (ms)")
    ax_sc.set_ylabel("display times interval (ms)")
    ax_sc.set_title("paired long frame lengths", fontsize=8)

    # frames whose display time isn't measured (npc_stim `BlockDisplayTimes.frame_source`)
    n_source = {}
    if frame_source is not None:
        for value, color, label in ((1, "tab:purple", "interpolated (replayed sync)"), (2, "tab:cyan", "constant lag in diode block")):
            is_value = np.asarray(frame_source) == value
            n_source[label] = int(is_value.sum())
            if not is_value.any():
                continue
            edges = np.diff(np.concatenate(([0], is_value.astype(int), [0])))
            min_width = 0.003 * max(a.n_stim_frames, a.n_display_frames)  # keep short spans visible
            for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                pad = max(0.0, (min_width - (stop - start)) / 2)
                for ax in (ax_idx, ax_len, ax_un):
                    ax.axvspan(start - 0.5 - pad, stop - 0.5 + pad, color=color, alpha=0.3, lw=0)
            ax_len.plot([], [], "s", color=color, alpha=0.4, label=f"{label} ({n_source[label]} frames)")
        if any(n_source.values()):
            ax_len.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1), borderaxespad=0)

    # 6. summary
    length_ok = a.n_stim_frames == a.n_display_frames
    lines = [
        f"frames: stim file {a.n_stim_frames}, display times {a.n_display_frames}"
        + ("  OK" if length_ok else f"  MISMATCH ({a.n_display_frames - a.n_stim_frames:+d})"),
        f"median period: stim file {a.stim_period * 1e3:.4f} ms, display {a.display_period * 1e3:.4f} ms",
        f"long frames (> {LONG_FRAME_THRESHOLD} periods): stim file {len(a.stim_long)}, display {len(a.display_long)}",
        f"most common index offset: {a.global_offset:+d}",
        f"paired: {len(offset)}, exact (same index & length): {int(a.is_exact.sum())}",
        f"  index offset != 0: {int((offset != 0).sum())}",
        f"  length differs: {int((rdiff != 0).sum())}",
        f"unpaired: stim file {len(a.stim_only)}, display {len(a.display_only)}",
    ] + [f"display frames {k}: {v}" for k, v in n_source.items() if v]
    deviants = [
        ("pair", int(s), int(d)) for s, d, e in zip(a.pair_stim, a.pair_display, a.is_exact) if not e
    ] + [("stim only", int(s), None) for s in a.stim_only] + [("display only", None, int(d)) for d in a.display_only]
    deviants.sort(key=lambda r: r[1] if r[1] is not None else r[2])
    if deviants:
        lines.append("first deviations (ms: stim / display / vsync):")
    for kind, s, d in deviants[:8]:
        k = s if s is not None else d
        vs = f"{vsync_intervals[k] * 1e3:.1f}" if vsync_intervals is not None and k < len(vsync_intervals) else "-"
        si = f"{a.stim_intervals[k] * 1e3:.1f}" if k < len(a.stim_intervals) else "-"
        k_d = d if d is not None else k
        di = f"{a.display_intervals[k_d] * 1e3:.1f}" if k_d < len(a.display_intervals) else "-"
        where = f"{s}" if d is None or s == d else f"{s}->{d}" if s is not None else f"{d}"
        src = "" if frame_source is None or k_d >= len(frame_source) or not frame_source[k_d] else f" [source {frame_source[k_d]}]"
        lines.append(f"  {kind:<12} {where:>13}: {si} / {di} / {vs}{src}")
    if len(deviants) > 8:
        lines.append(f"  ... {len(deviants) - 8} more")
    ax_txt.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=6.5, family="monospace", transform=ax_txt.transAxes)

    verdict = "MATCH" if a.passed else (
        "DEVIATIONS" + ("" if length_ok else f", frame count differs by {a.n_display_frames - a.n_stim_frames:+d}")
    )
    fig.suptitle(f"{title} | long frames, stim file vs display times: {verdict}", fontsize=9, color="k" if a.passed else "tab:red")
    fig.set_layout_engine("tight")
    return fig


def _summary_dict(a: LongFrameAlignment) -> dict[str, Any]:
    offset, rdiff = a.index_offset, a.refresh_diff
    return {
        "passed": a.passed,
        "n_stim_frames": a.n_stim_frames,
        "n_display_frames": a.n_display_frames,
        "n_long_stim": len(a.stim_long),
        "n_long_display": len(a.display_long),
        "global_index_offset": a.global_offset,
        "n_paired": len(offset),
        "n_exact": int(a.is_exact.sum()),
        "n_index_offset": int((offset != 0).sum()),
        "n_length_diff": int((rdiff != 0).sum()),
        "stim_only": a.stim_only.tolist()[:200],
        "display_only": a.display_only.tolist()[:200],
        "deviant_pairs": [
            {
                "stim_index": int(s),
                "display_index": int(d),
                "stim_interval_ms": round(float(a.stim_intervals[s] * 1e3), 2),
                "display_interval_ms": round(float(a.display_intervals[d] * 1e3), 2),
            }
            for s, d, e in zip(a.pair_stim, a.pair_display, a.is_exact)
            if not e
        ][:200],
    }


def plot_task_long_frame_alignment(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure | dict, ...] | None:
    """Task block: do the long (dropped) frames in the stim file's `frameIntervals` occur at
    the same frame index, with the same length, as in the npc_sessions frame display times
    (`session._stim_display_times`, photodiode or vsync + constant lag)? And do both have the
    same number of frames?

    Only indices and single intervals are compared (not cumsum(frameIntervals), whose clock
    isn't synced with sync). Grey dots: long frame at the same index with the same length.
    Orange: off by 1 frame; red: off by more, or a different length, or no partner.
    A systematic offset moves every dot off 0 in the top panel.
    """
    if not session.is_sync or not session.is_task:
        return None
    from npc_sessions_cache.plots.display_qc import get_display_block

    stems = [p.stem for p in session.stim_paths if p.stem.startswith(session.task_stim_name)]
    if not stems:
        return None
    out: list[plt.Figure | dict] = []
    stats: dict[str, Any] = {}
    for stem in stems:
        block = get_display_block(session, stem)
        if isinstance(block, Exception) or block.display_times is None:
            raise ValueError(f"{session.id} {stem}: no display times: {block!r}")
        stim_data = npc_stim.get_h5_stim_data(block.stim_path)
        try:
            frame_intervals = np.asarray(stim_data["frameIntervals"][:], dtype=float)
        finally:
            with contextlib.suppress(Exception):
                stim_data.close()
        a = get_long_frame_alignment(frame_intervals, block.display_times)
        out.append(
            _alignment_figure(
                a,
                f"{session.id} {stem} (display times: {block.source})",
                vsync_intervals=np.diff(np.asarray(block.vsync_times, dtype=float)),
                frame_source=block.frame_source,
            )
        )
        stats[stem] = {"display_time_source": block.source} | _summary_dict(a)
    out.append({"session_id": str(session.id), "blocks": stats})
    return tuple(out)
