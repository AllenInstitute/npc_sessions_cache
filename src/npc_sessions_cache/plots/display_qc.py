"""QC of visual stim (display) times against neural responses and the raw photodiode.

Two plots (hub handoff `constant-lag-fail-fast.md`, task 5; ported from hub
`experiments/constant-lag-qc`):

- `plot_quartile_visual_latency`: every sorted session with a task or RFMapping block.
  Population PSTHs per quarter of the block, half-max latency per quarter and per-stim onset
  shift. It calls `npc_sessions.qc.quartile_visual_latency`, the function that decides whether
  a block with vsync + constant-lag display times is valid, so the plot shows exactly what
  decided validity (the result on the session's validity record is reused when it has
  everything the plot needs).
- `plot_constant_lag_display_times`: only sessions with a task or RFMapping block whose
  display times are vsync + constant lag (`display_time_source == 'constant_lag'`). Raw
  photodiode timeline, photodiode latency on 60-s chunks, rising vs falling edges, per-stim
  shift vs diode noise, long (dropped) frames on sync vs in the stim file, and the block's
  validity record.

Neither needs `session.trials`: visual stim onsets come from the stim file and the candidate
display times (`npc_sessions.qc.get_stim_onsets`), so they work for blocks that were excluded.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import textwrap
import time
from collections.abc import Iterator
from typing import Any

import matplotlib.pyplot as plt
import npc_sessions
import npc_stim
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FRAME_MS = 1e3 / 60
REVIEW_MAX_LATENCY_RANGE_MS = 8.0
"""Review if the visual half-max latency varies by more than this over quarters (also the
limit for a constant-lag block to be valid: `npc_sessions.qc.MAX_LATENCY_RANGE_MS`)"""
REVIEW_MAX_FRAC_SHIFTED = 0.10
"""Review a task block if more than this fraction of its visual stims have an estimated
onset shift over `npc_sessions.qc.MAX_STIM_SHIFT_MS` (and the auditory control isn't
unstable)"""
AUD_STABLE_MAX_RANGE_MS = 8.0
CHUNK_FRAMES = 3600
"""Frames per chunk (~60 s) for the chunked photodiode estimate"""
CHUNK_MIN_FRAMES = 600
CHUNK_MAX_EDGE_COUNT_DIFF = 100
"""Chunks whose number of diode edges differs from their number of frames by more than this
aren't passed to npc_stim's estimator: it removes anomalous intervals one at a time (minutes
per chunk) and then rejects chunks that are still off by more than 5"""
CHUNK_MAX_ANOMALOUS_ITERATIONS = 2000
CHUNK_TIME_BUDGET_S = 300.0
"""Total time allowed for the chunked photodiode estimate of one block: chunks after that
are not estimated"""
SHORT_VSYNC_PULSE_S = 0.010
QUARTILE_COLORS = [plt.cm.viridis(k / 3) for k in range(4)]


def _qc():  # noqa: ANN202
    try:
        import npc_sessions.qc as qc
    except ImportError as exc:
        raise ImportError(
            "npc_sessions.qc (quartile_visual_latency) isn't in this version of npc_sessions "
            f"({npc_sessions.__version__ if hasattr(npc_sessions, '__version__') else '?'})"
        ) from exc
    return qc


# blocks, display times, validity ---------------------------------------------------- #


def get_visual_stim_stems(session: npc_sessions.DynamicRoutingSession) -> list[str]:
    """Stems of the session's stim files with visual stims whose times come from display
    times: the task and RFMapping, in the order of `session.stim_paths`."""
    stems = []
    for path in session.stim_paths:
        if path.stem.startswith(session.task_stim_name) or "RFMapping" in path.stem:
            stems.append(path.stem)
    return stems


def _is_task_stem(session: npc_sessions.DynamicRoutingSession, stem: str) -> bool:
    return stem.startswith(session.task_stim_name)


def get_display_block(
    session: npc_sessions.DynamicRoutingSession, stem: str
) -> npc_stim.BlockDisplayTimes | Exception:
    """The stim's candidate display times with provenance, also for a block that was
    excluded: from the session, or (if the session doesn't keep them for the block) from
    npc_stim directly, with the constant-lag fallback."""
    block: Any = None
    try:
        block = session._stim_display_times.get(stem)
    except Exception as exc:  # noqa: BLE001
        block = exc
    if isinstance(block, npc_stim.BlockDisplayTimes) and block.display_times is not None:
        return block
    cached = session.__dict__.get("_qc_plot_display_blocks")
    if cached is None:
        try:
            cached = {
                str(getattr(k, "stem", k)): v
                for k, v in npc_stim.get_stim_display_times(
                    *session.stim_paths, sync=session.sync_data, fallback="constant_lag"
                ).items()
            }
        except Exception as exc:  # noqa: BLE001
            cached = {"__error__": exc}
        session.__dict__["_qc_plot_display_blocks"] = cached
    if stem in cached:
        return cached[stem]
    if isinstance(block, Exception):
        return block
    return cached.get("__error__", ValueError(f"no display times for {stem}"))


def get_stim_validity(
    session: npc_sessions.DynamicRoutingSession, stem: str
) -> Any | None:  # noqa: ANN401
    """The session's validity record for the stim (`session.stim_validity[stem]`), or None
    if this version of npc_sessions has none, or the block has no record."""
    try:
        validity = getattr(session, "stim_validity", None)
        if validity is None:
            return None
        return validity.get(stem)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{session.id}: stim_validity not available: {exc!r}")
        return None


def _summary(obj: Any) -> str | None:  # noqa: ANN401
    if obj is None:
        return None
    return str(getattr(obj, "summary", obj))


def _load_units_once(session: npc_sessions.DynamicRoutingSession) -> None:
    """Make the session's units table if it isn't made yet, so that
    `npc_sessions.qc.get_unit_spike_times` takes spike times from it rather than loading the
    sorted data a second time (the other QC plots use `session.units`). If the table can't
    be made (e.g. no trials for the activity-drift metric), the check loads its own."""
    if "_qc_unit_spike_times" in session.__dict__ or "_units" in session.__dict__:
        return
    try:
        _ = session._units
    except Exception as exc:  # noqa: BLE001
        logger.info(f"{session.id}: units table not available ({exc!r}): the check loads spikes itself")


def _task_aud_onsets(
    session: npc_sessions.DynamicRoutingSession,
    stem: str,
    display_times: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64] | None, str]:
    """Onsets of the task's auditory stims without opto (the control for the visual
    latencies: sound onsets don't use display times), and where they came from."""
    try:
        if session.is_task and session.task_path.stem == stem:
            trials = session.trials[:]
            keep = trials["is_aud_stim"].astype(bool)
            if "is_opto" in trials:
                keep &= ~trials["is_opto"].astype(bool)
            return trials[keep]["stim_start_time"].dropna().to_numpy(dtype=float), "trials table"
    except Exception as exc:  # noqa: BLE001
        logger.info(f"{session.id}: no trials table for auditory onsets: {exc!r}")
    # no trials table (e.g. the block was excluded): from the stim file, as the trials
    # table would have them
    import npc_sessions.trials as TaskControl  # noqa: N812

    try:
        kwargs: dict[str, Any] = {"vis_display_times": display_times}
        if session.is_ephys:
            kwargs["ephys_recording_dirs"] = session.ephys_recording_dirs
        obj = TaskControl.DynamicRouting1(
            next(p for p in session.stim_paths if p.stem == stem),
            session.sync_data,
            **kwargs,
        )
        keep = np.asarray(obj.is_aud_stim, dtype=bool)
        if obj._is_opto:
            keep &= ~np.asarray(obj.is_opto, dtype=bool)
        onsets = np.asarray(obj.stim_start_time, dtype=float)[keep]
        return onsets[np.isfinite(onsets)], "stim file + sound alignment (no trials table)"
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{session.id}: auditory onsets not available for {stem}: {exc!r}")
        return None, f"not available: {exc!r}"


def get_quartile_latency(
    session: npc_sessions.DynamicRoutingSession, stem: str
) -> tuple[Any, dict[str, Any]]:
    """`npc_sessions.qc.quartile_visual_latency` for a stim block, with the block's
    candidate display times, plus (for the task) the auditory control. Returns the
    `QuartileLatencyResult` and a dict about how it was made. Kept on the session, so the
    two plots here run it once per block."""
    cache = session.__dict__.setdefault("_qc_plot_quartile_latency", {})
    if stem in cache:
        return cache[stem]
    qc = _qc()
    block = get_display_block(session, stem)
    if isinstance(block, Exception) or block.display_times is None:
        raise ValueError(f"{session.id}: no display times for {stem}: {block!r}")
    is_task = _is_task_stem(session, stem)
    validity = get_stim_validity(session, stem)
    recorded = getattr(validity, "latency_check", None)
    info: dict[str, Any] = {
        "stim": "task" if is_task else "rf_mapping",
        "display_time_source": block.source,
        "vis_onsets_from": "stim file + display times (npc_sessions.qc.get_stim_onsets)",
        "aud_onsets_from": None,
        "from_validity_record": False,
        "matches_validity_record": None,
    }
    if recorded is not None and (not is_task or recorded.aud is not None):
        # the result that decided validity has everything the plot shows
        info["from_validity_record"] = True
        info["matches_validity_record"] = True
        if recorded.aud is not None:
            info["aud_onsets_from"] = "validity record"
        cache[stem] = (recorded, info)
        return cache[stem]
    aud = None
    if is_task:
        aud, info["aud_onsets_from"] = _task_aud_onsets(
            session, stem, np.asarray(block.display_times, dtype=float)
        )
    _load_units_once(session)
    t = time.time()
    result = qc.quartile_visual_latency(
        session, stem, display_times=block.display_times, aud_onsets=aud
    )
    logger.info(f"{session.id} {stem}: latency check itself in {time.time() - t:.1f} s")
    if recorded is not None:
        # same function and inputs, plus the auditory control: the verdict can't differ
        same = recorded.verdict == result.verdict and np.allclose(
            recorded.vis.latencies_ms, result.vis.latencies_ms, equal_nan=True
        )
        info["matches_validity_record"] = bool(same)
        if not same:
            logger.error(
                f"{session.id} {stem}: latency check here ({result.summary}) differs from "
                f"the validity record ({recorded.summary})"
            )
    cache[stem] = (result, info)
    return cache[stem]


def _review(result: Any, is_task: bool) -> tuple[bool, list[str], bool | None]:  # noqa: ANN401
    """Review flag for a block, its reasons, and whether the auditory control is stable
    (None: no auditory latencies).

    - visual latency range over quarters > `REVIEW_MAX_LATENCY_RANGE_MS`: always (this is
      the limit for validity of a constant-lag block)
    - task only: more than `REVIEW_MAX_FRAC_SHIFTED` of the visual stims shifted, unless the
      auditory latency is unstable too (then the units, not the display, changed). The
      shifted fraction is ~40% in any RFMapping block (one small grating drives few units)
    """
    aud_range = result.aud_range_ms
    aud_stable = (
        bool(aud_range <= AUD_STABLE_MAX_RANGE_MS) if np.isfinite(aud_range) else None
    )
    reasons = []
    if np.isfinite(result.vis_range_ms) and result.vis_range_ms > REVIEW_MAX_LATENCY_RANGE_MS:
        reasons.append(
            f"visual latency range {result.vis_range_ms:.1f} ms > {REVIEW_MAX_LATENCY_RANGE_MS:g} ms"
            + (
                ""
                if aud_stable is None
                else f" (auditory range {aud_range:.1f} ms: {'stable' if aud_stable else 'also unstable'})"
            )
        )
    frac = result.frac_shifted
    if is_task and np.isfinite(frac) and frac > REVIEW_MAX_FRAC_SHIFTED and aud_stable is not False:
        reasons.append(
            f"{frac:.0%} of visual stims shifted > {result.stim_shifts.max_shift_ms:g} ms"
            + ("" if aud_stable is None else f" (auditory range {aud_range:.1f} ms: stable)")
        )
    return bool(reasons), reasons, aud_stable


def _latency_stats(result: Any, info: dict[str, Any], is_task: bool) -> dict[str, Any]:  # noqa: ANN401
    review, reasons, aud_stable = _review(result, is_task)
    return {
        **info,
        **result.to_dict(include_arrays=False),
        "aud_stable": aud_stable,
        "review": review,
        "review_reasons": reasons,
    }


def get_quartile_rates(
    session: npc_sessions.DynamicRoutingSession, stem: str, result: Any  # noqa: ANN401
) -> dict[str, list[float]] | None:
    """Firing rates of the visually responsive units of the latency check in each quarter of
    the block's visual stims: mean baseline rate (-100-0 ms) and peak evoked rate (mean PSTH,
    5-ms boxcar, 20-150 ms, minus baseline), over all of the block's visual stims (RFMapping:
    all small gratings, so evoked rates are diluted).

    A display or sync timing error shifts the response without changing its size; a change in
    the animal's state (arousal, running, engagement) changes baseline and evoked rates, and
    the half-max latency with them: weaker responses peak later and rise more slowly
    (hub experiments/timing-quartile: 733891_2024-09-18/19, 741148_2024-10-18)."""
    qc = _qc()
    if result.vis is None or not result.vis.quartiles:
        return None
    block = get_display_block(session, stem)
    if isinstance(block, Exception) or block.display_times is None:
        return None
    onsets = np.asarray(qc.get_stim_onsets(session, stem, block.display_times).vis, dtype=float)
    onsets = np.sort(onsets[np.isfinite(onsets)])
    spike_times = qc.get_unit_spike_times(session)
    units = [np.asarray(spike_times[u]) for u in result.vis.unit_ids if u in spike_times]
    if not units:
        return None
    edges = np.arange(-0.1, 0.15 + 0.001, 0.001)
    centers = edges[:-1] + 0.0005
    baseline, evoked = [], []
    for q in result.vis.quartiles:
        on = onsets[(onsets >= q.start_time) & (onsets <= q.stop_time)]
        psth = np.zeros(len(centers))
        for st in units:
            lo = np.searchsorted(st, on + edges[0])
            hi = np.searchsorted(st, on + edges[-1])
            rel = np.concatenate([st[a:b] - t for a, b, t in zip(lo, hi, on)])
            psth += np.histogram(rel, edges)[0]
        psth = np.convolve(psth / len(units) / len(on) / 0.001, np.ones(5) / 5, mode="same")
        b = float(psth[centers < 0].mean())
        baseline.append(b)
        evoked.append(float(psth[(centers >= 0.02) & (centers < 0.15)].max() - b))
    return {"vis_baseline_hz": baseline, "vis_evoked_hz": evoked}


def _rate_change_note(stats: dict[str, Any]) -> str:
    """'evoked rate x-y Hz, baseline x-y Hz' over quarters, if known"""
    ev, bl = stats.get("vis_evoked_hz"), stats.get("vis_baseline_hz")
    if not ev or not bl:
        return ""
    return (
        f"evoked rate {min(ev):.1f}-{max(ev):.1f} Hz (x{max(ev) / max(min(ev), 1e-9):.2f}), "
        f"baseline {min(bl):.1f}-{max(bl):.1f} Hz over quarters"
    )


# quartile PSTH plot ---------------------------------------------------------------- #


def _wrap(text: str, width: int = 150) -> str:
    return "\n".join(textwrap.wrap(text, width)) if text else ""


def _plot_quartile_psths(ax: plt.Axes, stats: Any, label: str) -> None:  # noqa: ANN401
    if stats is None or not stats.quartiles:
        n = "no onsets" if stats is None else f"{stats.n_units} responsive units of {stats.n_units_tested}, {stats.n_trials} stims"
        ax.text(0.5, 0.5, f"{label}: no PSTHs\n({n})", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    t = stats.psth_time * 1e3
    for q, color in zip(stats.quartiles, QUARTILE_COLORS):
        ax.plot(t, q.psth, color=color, lw=0.9, label=f"Q{q.index + 1}: {q.latency_ms:.1f} ms (n={q.n_trials})")
        if np.isfinite(q.latency_ms):
            ax.axvline(q.latency_ms, color=color, lw=0.5, ls=":")
    ax.axvline(0, color="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-20, 150)
    ax.set_xlabel("time from stim onset (ms)")
    ax.set_title(
        f"{label}: {stats.n_units} responsive units of {stats.n_units_tested}, {stats.n_trials} stims",
        fontsize=7,
    )
    ax.legend(fontsize=5.5, loc="upper right", title="half-max latency", title_fontsize=5.5)


def _plot_latencies(ax: plt.Axes, result: Any, block_stats: dict[str, Any] | None = None) -> None:  # noqa: ANN401
    drawn = False
    for stats, color, label, offset in (
        (result.vis, "tab:blue", "vis", -0.05),
        (result.aud, "tab:orange", "aud", 0.05),
    ):
        if stats is None or not stats.quartiles:
            continue
        lat = np.array(stats.latencies_ms)
        lo = np.array([q.ci_low_ms for q in stats.quartiles])
        hi = np.array([q.ci_high_ms for q in stats.quartiles])
        yerr = np.clip(np.nan_to_num(np.array([lat - lo, hi - lat])), 0, None)
        ax.errorbar(
            np.arange(len(lat)) + offset,
            lat,
            yerr=yerr,
            fmt="o-",
            ms=3,
            lw=0.8,
            color=color,
            capsize=2,
            label=f"{label}: range {stats.latency_range_ms:.1f} ms",
        )
        drawn = True
    if drawn:
        n = len(result.vis.quartiles) or (len(result.aud.quartiles) if result.aud else 4)
        ax.set_xticks(range(n), [f"Q{k + 1}" for k in range(n)])
        ax.legend(fontsize=6)
    ax.set_ylabel("half-max latency (ms)\n(bars: 95% bootstrap CI)")
    ax.set_xlabel("quarter of the block's stims")
    ev = (block_stats or {}).get("vis_evoked_hz")
    bl = (block_stats or {}).get("vis_baseline_hz")
    if ev and bl and ev[0] > 0 and bl[0] > 0:
        # a timing error moves the latency without changing the rates; a change of state
        # changes both
        ax2 = ax.twinx()
        x = np.arange(len(ev))
        ax2.plot(x, np.array(ev) / ev[0], "s--", ms=2.5, lw=0.7, color="tab:green", label="vis evoked rate")
        ax2.plot(x, np.array(bl) / bl[0], "^--", ms=2.5, lw=0.7, color="0.5", label="vis baseline rate")
        ax2.set_ylabel("rate / Q1 rate", fontsize=6)
        ax2.tick_params(labelsize=6)
        ax2.legend(fontsize=5.5, loc="lower right")
    ax.set_title(f"limit: vis range <= {REVIEW_MAX_LATENCY_RANGE_MS:g} ms", fontsize=7)


def _plot_stim_shifts(
    ax: plt.Axes, result: Any, t0: float, legend: bool = True  # noqa: ANN401
) -> None:
    shifts = result.stim_shifts
    if shifts is None:
        ax.text(
            0.5, 0.5, f"no per-stim onset shifts ({result.verdict}: {result.reason})",
            ha="center", va="center", transform=ax.transAxes, fontsize=7, wrap=True,
        )
        return
    minutes = (shifts.onsets - t0) / 60
    over = np.abs(shifts.shift_ms) > shifts.max_shift_ms + 1e-6
    ax.plot(minutes[~over], shifts.shift_ms[~over], "o", ms=2.2, color="tab:blue", label=f"|shift| <= {shifts.max_shift_ms:g} ms")
    ax.plot(minutes[over], shifts.shift_ms[over], "o", ms=2.2, color="tab:red", label=f"|shift| > {shifts.max_shift_ms:g} ms ({shifts.frac_shifted:.0%} of stims)")
    for k in (-2, -1, 0, 1, 2):
        ax.axhline(k * FRAME_MS, color="0.6", lw=0.5, ls=":" if k else "-")
    ax.axhspan(-shifts.max_shift_ms, shifts.max_shift_ms, color="0.93", zorder=-1)
    for q in result.vis.quartiles[1:]:
        ax.axvline((q.start_time - t0) / 60, color="k", lw=0.5, ls="--")
    for q, color in zip(result.vis.quartiles, QUARTILE_COLORS):
        ax.text(
            ((q.start_time + q.stop_time) / 2 - t0) / 60, 50, f"Q{q.index + 1}",
            ha="center", va="top", fontsize=6, color=color,
        )
    ax.set_ylim(-52, 52)
    ax.set_ylabel("per-stim onset shift (ms)")
    if legend:
        ax.legend(fontsize=6, loc="lower left", ncol=2)


def _quartile_figure(
    session: npc_sessions.DynamicRoutingSession,
    stem: str,
    result: Any,  # noqa: ANN401
    stats: dict[str, Any],
    t0: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    ax_vis = fig.add_subplot(gs[0, 0])
    ax_aud = fig.add_subplot(gs[0, 1])
    ax_lat = fig.add_subplot(gs[0, 2])
    ax_shift = fig.add_subplot(gs[1, :])
    _plot_quartile_psths(ax_vis, result.vis, "visual stims")
    ax_vis.set_ylabel("population PSTH\n(mean of peak-normalised units)")
    if result.aud is None:
        ax_aud.text(
            0.5, 0.5,
            "no auditory control"
            + ("" if stats["stim"] == "task" else "\n(RFMapping: small gratings only)"),
            ha="center", va="center", transform=ax_aud.transAxes, fontsize=7,
        )
        ax_aud.axis("off")
    else:
        _plot_quartile_psths(ax_aud, result.aud, "auditory stims (control)")
    _plot_latencies(ax_lat, result, stats)
    _plot_stim_shifts(ax_shift, result, t0)
    ax_shift.set_xlabel(
        "minutes into block (template: Q1 stims; dotted: +-1, 2 frames; dashed: quarter boundaries)"
    )
    ax_shift.set_title(
        "per-stim maximum-likelihood onset shift of the visual response (positive: response later than in Q1)"
        + (
            ""
            if stats["stim"] == "task"
            else "\nRFMapping: a small grating drives few units, so ~40% of stims over 8 ms is usual: look for a step"
        ),
        fontsize=7,
    )
    review = (
        "REVIEW: " + "; ".join(stats["review_reasons"]) if stats["review"] else "no review flag"
    )
    fig.suptitle(
        f"{session.id} {stem} | display times: {stats['display_time_source']} | "
        f"latency check {result.verdict.upper()}\n{_wrap(result.reason)}\n{_wrap(review)}"
        + (
            "\n" + _wrap(f"BLOCK INVALID: {stats.get('invalid_reason')}")
            if stats.get("is_valid") is False
            else ""
        ),
        fontsize=7.5,
        color="tab:red" if stats["review"] or stats.get("is_valid") is False else "k",
    )
    fig.set_layout_engine("tight")
    return fig


def plot_quartile_visual_latency(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure | dict, ...] | None:
    """Is the latency of the population visual response stable over each block with visual
    stims (the task; RFMapping)? Stim onsets come from frame display times, so a display
    that runs late from partway through a block shows as a step in latency.

    One figure per block, then the stats (dict):

    - top left, middle: population PSTH of the visually (auditory: control, task only)
      responsive units in each quarter of the block's stims; dotted lines: half-max latency
    - top right: half-max latency per quarter with 95% bootstrap CIs
    - bottom: per-stim onset shift (maximum likelihood; template: the first quarter's stims)

    Stats, per block: `verdict` ('passed': visual latency range over quarters <= 8 ms with
    >= 5 responsive units; 'failed'; 'untestable'), `vis_range_ms`, `aud_range_ms`,
    `frac_shifted` (stims with |shift| > 8 ms), per quarter latency / CI / n units / n
    trials, and `review`: True if the visual range is > 8 ms, or (task) > 10% of stims are
    shifted while the auditory latency is stable. Top-level `review`: any block.

    Computed by `npc_sessions.qc.quartile_visual_latency`: the check that decides whether a
    block with vsync + constant-lag display times is valid. Doesn't need a trials table.
    """
    if not (session.is_sync and session.is_sorted):
        return None
    stems = get_visual_stim_stems(session)
    if not stems:
        return None
    figs: list[plt.Figure | dict] = []
    blocks: dict[str, dict[str, Any]] = {}
    for stem in stems:
        t = time.time()
        try:
            result, info = get_quartile_latency(session, stem)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{session.id} {stem}: latency check not run: {exc!r}")
            blocks[stem] = {"error": repr(exc)}
            continue
        is_task = _is_task_stem(session, stem)
        stats = _latency_stats(result, info, is_task)
        try:
            stats.update(get_quartile_rates(session, stem, result) or {})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{session.id} {stem}: quartile firing rates not computed: {exc!r}")
        if stats["review"] and (note := _rate_change_note(stats)):
            stats["review_reasons"] = [*stats["review_reasons"], note]
        validity = get_stim_validity(session, stem)
        # the block's validity (npc_sessions): not only from this check, e.g. a constant-lag
        # block can pass it and be invalid for its latency vs a photodiode block
        stats["is_valid"] = None if validity is None else bool(validity.is_valid)
        stats["invalid_reason"] = (
            str(validity.reason) if validity is not None and not validity.is_valid else None
        )
        blocks[stem] = stats
        if is_task or result.vis.quartiles:
            # RFMapping blocks without enough responsive units: stats only
            block = get_display_block(session, stem)
            figs.append(
                _quartile_figure(session, stem, result, stats, float(block.vsync_times[0]))
            )
        logger.info(f"{session.id} {stem}: quartile latency in {time.time() - t:.0f} s")
    if not any("error" not in b for b in blocks.values()):
        raise ValueError(f"{session.id}: latency check not run for any block: {blocks}")
    figs.append(
        {
            "session_id": str(session.id),
            "review": any(b.get("review", False) for b in blocks.values()),
            "blocks": blocks,
        }
    )
    return tuple(figs)


# constant-lag blocks: raw photodiode ---------------------------------------------------- #


@dataclasses.dataclass
class DiodeData:
    """Raw photodiode edges of a block against its vsyncs."""

    vsyncs: npt.NDArray[np.float64]
    """Falling edges, one per frame"""
    vsync_rising: npt.NDArray[np.float64]
    edges: npt.NDArray[np.float64]
    """Photodiode edges within the block, sorted"""
    is_rising: npt.NDArray[np.bool_]
    frames_per_flip: int
    toggle_frames: npt.NDArray[np.intp]
    """Frames at which the diode box flips"""
    latency: npt.NDArray[np.float64]
    """Per toggle frame: first diode edge within half a frame of the expected flip time,
    minus the frame's vsync (s); nan if none"""
    edge_idx: npt.NDArray[np.intp]
    """Per toggle frame: index in `edges` of that edge (meaningless where latency is nan)"""
    is_explained: npt.NDArray[np.bool_]
    """Per edge: used as a toggle frame's flip"""


def get_diode_data(
    session: npc_sessions.DynamicRoutingSession, block: npc_stim.BlockDisplayTimes
) -> DiodeData:
    import npc_stim.display_times as dt

    sync = session.sync_data
    vsyncs = np.asarray(block.vsync_times, dtype=float)
    try:
        rising = np.asarray(sync.get_rising_edges("stim_photodiode", units="seconds"))
        falling = np.asarray(sync.get_falling_edges("stim_photodiode", units="seconds"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{session.id}: no photodiode edges on sync: {exc!r}")
        rising = falling = np.array([])
    edges = np.concatenate([rising, falling])
    is_rising = np.concatenate([np.ones(len(rising), bool), np.zeros(len(falling), bool)])
    order = np.argsort(edges, kind="stable")
    edges, is_rising = edges[order], is_rising[order]
    in_block = (edges > vsyncs[0] - 0.1) & (edges < vsyncs[-1] + 0.1)
    edges, is_rising = edges[in_block], is_rising[in_block]
    n = block.diode_box_frame_interval
    if n is None:
        n = dt.get_diode_box_frame_interval(block.stim_path) if block.stim_path else 1
    toggle = np.arange(0, len(vsyncs), n)
    half = dt.FRAME_INTERVAL / 2
    lo = vsyncs[toggle] + dt.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME - half
    if len(edges):
        i = np.searchsorted(edges, lo)
        idx = np.clip(i, 0, len(edges) - 1)
        ok = (i < len(edges)) & (edges[idx] < lo + dt.FRAME_INTERVAL)
        latency = np.where(ok, edges[idx] - vsyncs[toggle], np.nan)
    else:
        idx = np.zeros(len(toggle), dtype=np.intp)
        ok = np.zeros(len(toggle), dtype=bool)
        latency = np.full(len(toggle), np.nan)
    explained = np.zeros(len(edges), dtype=bool)
    explained[idx[ok]] = True
    return DiodeData(
        vsyncs=vsyncs,
        vsync_rising=np.asarray(dt.get_vsync_rising_edges(sync, vsyncs), dtype=float),
        edges=edges,
        is_rising=is_rising,
        frames_per_flip=int(n),
        toggle_frames=toggle,
        latency=latency,
        edge_idx=idx,
        is_explained=explained,
    )


def get_diode_minutes(diode: DiodeData) -> dict[str, npt.NDArray]:
    """Per minute of the block: fraction of toggle frames with a diode edge, diode edges
    not explained by a frame, fraction of vsync pulses shorter than 10 ms, and the median
    latency (edge - vsync + 8 ms: centre of the screen, comparable with the constant lag) of
    the rising and of the falling diode edges (nan with fewer than 100)."""
    import npc_stim.display_times as dt

    t0 = diode.vsyncs[0]
    n_min = int((diode.vsyncs[-1] - t0) // 60) + 1
    frame_min = ((diode.vsyncs - t0) // 60).astype(int)
    toggle_min = frame_min[diode.toggle_frames]
    edge_min = np.clip(((diode.edges - t0) // 60).astype(int), 0, n_min - 1)
    has_edge = np.isfinite(diode.latency)
    n_toggle = np.bincount(toggle_min, minlength=n_min)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.bincount(toggle_min, weights=has_edge, minlength=n_min) / n_toggle
        short = np.bincount(
            frame_min,
            weights=(diode.vsyncs - diode.vsync_rising) < SHORT_VSYNC_PULSE_S,
            minlength=n_min,
        ) / np.bincount(frame_min, minlength=n_min)
    unexplained = np.bincount(edge_min[~diode.is_explained], minlength=n_min)
    rise = np.full(n_min, np.nan)
    fall = np.full(n_min, np.nan)
    if len(diode.edges):
        polarity = diode.is_rising[diode.edge_idx]
        lat_ms = (diode.latency + dt.MONITOR_CENTER_REFRESH_TIME) * 1e3
        for k in range(n_min):
            m = (toggle_min == k) & has_edge
            for out, sel in ((rise, m & polarity), (fall, m & ~polarity)):
                if sel.sum() >= 100:
                    out[k] = np.median(lat_ms[sel])
    return {
        "minute": np.arange(n_min),
        "coverage": coverage,
        "unexplained_edges": unexplained,
        "short_vsync_pulse_frac": short,
        "rise_latency_ms": rise,
        "fall_latency_ms": fall,
    }


class _NotConverged(Exception):
    pass


@contextlib.contextmanager
def _estimator_guards() -> Iterator[list[int]]:
    """npc_stim's block estimator can take very long, or (npc_stim <= 0.1.28) never return,
    on a noisy chunk of diode edges: cap its anomalous-interval removal loop, and stop it
    when no missing flip can be added at the end of the block. Yields the call counter
    (reset it per chunk)."""
    import npc_stim.display_times as dt

    calls = [0]
    orig_anomalous = dt.anomalous_interval_indices
    orig_add = dt.add_missing_diode_flips_for_truncated_sync

    def anomalous(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        calls[0] += 1
        if calls[0] > CHUNK_MAX_ANOMALOUS_ITERATIONS:
            raise _NotConverged("anomalous-interval removal did not converge")
        return orig_anomalous(*args, **kwargs)

    def add(flips: Any, vsyncs: Any) -> Any:  # noqa: ANN401
        out = orig_add(flips, vsyncs)
        if len(out) == len(flips):
            raise _NotConverged("flips missing, but not at the end of the chunk")
        return out

    stim_logger = logging.getLogger(dt.__name__)
    level = stim_logger.level
    dt.anomalous_interval_indices = anomalous
    dt.add_missing_diode_flips_for_truncated_sync = add
    stim_logger.setLevel(logging.CRITICAL)
    try:
        yield calls
    finally:
        dt.anomalous_interval_indices = orig_anomalous
        dt.add_missing_diode_flips_for_truncated_sync = orig_add
        stim_logger.setLevel(level)


def _chunk_category(n_edges: int, n_frames: int, ok: bool, reason: str | None) -> str:
    """'ok', 'no_diode' (fewer than half the expected edges), 'flip_deficit' (diode present,
    6-150 fewer edges than frames, or 1-3 short at the end: the display skipped frames that
    vsync reported; hub constant-lag-qc `rule.py`) or 'other' (e.g. misaligned)."""
    if ok:
        return "ok"
    if reason and reason.startswith("not estimated"):
        return "not_estimated"
    if n_edges < n_frames / 2:
        return "no_diode"
    if -150 <= n_edges - n_frames < -5 or (reason and "not at the end" in reason):
        return "flip_deficit"
    return "other"


def get_chunk_latencies(
    diode: DiodeData, name: str
) -> tuple[npt.NDArray[np.float64], list[dict[str, Any]]]:
    """Photodiode display latency (display time - vsync, ms; comparable with the constant
    lag) from npc_stim's estimator on `CHUNK_FRAMES`-frame chunks of the raw diode edges:
    the parts of a rejected block where the diode works. Returns the latency per frame (nan
    where a chunk has no estimate) and a record per chunk. Never raises for a chunk, and
    stops estimating after `CHUNK_TIME_BUDGET_S`."""
    import npc_stim.display_times as dt

    vsyncs = diode.vsyncs
    rising, falling = diode.edges[diode.is_rising], diode.edges[~diode.is_rising]
    latency = np.full(len(vsyncs), np.nan)
    records: list[dict[str, Any]] = []
    t_start = time.time()
    with _estimator_guards() as calls:
        for c0 in range(0, len(vsyncs), CHUNK_FRAMES):
            v = vsyncs[c0 : c0 + CHUNK_FRAMES]
            if len(v) < CHUNK_MIN_FRAMES:
                continue
            r = rising[(rising > v[0]) & (rising < v[-1] + 0.2)]
            f = falling[(falling > v[0]) & (falling < v[-1] + 0.2)]
            n_edges = (len(r) + len(f)) * diode.frames_per_flip
            record: dict[str, Any] = {
                "start_frame": int(c0),
                "start_min": float((v[0] - vsyncs[0]) / 60),
                "stop_min": float((v[-1] - vsyncs[0]) / 60),
                "n_frames": int(len(v)),
                "n_diode_edges": int(len(r) + len(f)),
                "ok": False,
                "reason": None,
            }
            estimate = None
            if time.time() - t_start > CHUNK_TIME_BUDGET_S:
                record["reason"] = f"not estimated: over {CHUNK_TIME_BUDGET_S:g} s for the block"
            elif abs(n_edges - len(v)) > CHUNK_MAX_EDGE_COUNT_DIFF:
                record["reason"] = f"{len(r) + len(f)} diode edges for {len(v)} frames"
            else:
                calls[0] = 0
                try:
                    estimate = dt._estimate_block_display_times(
                        sync_data_name=name,
                        block_idx=0,
                        stim_id=f"{name}_chunk{c0}",
                        vsyncs=v,
                        vsync_rising_edges=diode.vsync_rising[c0 : c0 + CHUNK_FRAMES],
                        rising=r,
                        falling=f,
                        first_frame_start=float(v[0]),
                        vsyncs_per_diode_flip=diode.frames_per_flip,
                        expected_vsyncs=len(v),
                        diode_cleaned=False,
                        raise_on_error=False,
                    )
                    if estimate.reason is not None or estimate.times is None:
                        record["reason"] = str(estimate.reason)[:200]
                        estimate = None
                except Exception as exc:  # noqa: BLE001
                    record["reason"] = f"{type(exc).__name__}: {exc}"[:200]
            if estimate is not None:
                lat = (estimate.times + dt.MONITOR_CENTER_REFRESH_TIME - v) * 1e3
                latency[c0 : c0 + len(v)] = lat
                record.update(ok=True, latency_median_ms=float(np.median(lat)))
            record["category"] = _chunk_category(n_edges, len(v), record["ok"], record["reason"])
            records.append(record)
    return latency, records


def _chunk_summary(
    latency: npt.NDArray[np.float64], records: list[dict[str, Any]], seconds: float
) -> dict[str, Any]:
    import npc_stim.display_times as dt

    lag_ms = dt.CONSTANT_MONITOR_LAG * 1e3
    categories = [r["category"] for r in records]
    run = longest = 0
    onset = None
    for r in records:
        run = run + 1 if r["category"] == "flip_deficit" else 0
        longest = max(longest, run)
        if run == 3 and onset is None:
            onset = records[records.index(r) - 2]["start_min"]
    finite = latency[np.isfinite(latency)]
    return {
        "chunk_frames": CHUNK_FRAMES,
        "n_chunks": len(records),
        "n_chunks_by_category": {c: categories.count(c) for c in sorted(set(categories))},
        "longest_flip_deficit_run": longest,
        "flip_deficit_run_onset_min": onset,
        "frac_frames_with_estimate": float(np.isfinite(latency).mean()) if len(latency) else None,
        "latency_ms_median_p1_p99": (
            [float(v) for v in np.percentile(finite, [50, 1, 99])] if finite.size else None
        ),
        "abs_diff_from_constant_lag_ms_p99": (
            float(np.percentile(np.abs(finite - lag_ms), 99)) if finite.size else None
        ),
        "constant_lag_ms": lag_ms,
        "seconds": round(seconds, 1),
        "failed_chunks": [
            {k: r[k] for k in ("start_min", "n_frames", "n_diode_edges", "category", "reason")}
            for r in records
            if not r["ok"]
        ],
    }


CHUNK_CATEGORY_COLORS = {
    "flip_deficit": "tab:red",
    "no_diode": "0.75",
    "other": "tab:olive",
    "not_estimated": "tab:purple",
}
CHUNK_CATEGORY_LABELS = {
    "flip_deficit": "chunk failed: flip deficit (fewer diode flips than vsyncs)",
    "no_diode": "chunk failed: diode absent",
    "other": "chunk failed: other (e.g. misaligned)",
    "not_estimated": "chunk not estimated (time budget)",
}


def _diode_figure(
    session: npc_sessions.DynamicRoutingSession,
    stem: str,
    block: npc_stim.BlockDisplayTimes,
    diode: DiodeData,
    minutes: dict[str, npt.NDArray],
    chunk_latency: npt.NDArray[np.float64],
    chunks: list[dict[str, Any]],
    latency_result: Any | None,  # noqa: ANN401
    latency_error: str | None,
) -> plt.Figure:
    import npc_stim.display_times as dt

    lag_ms = dt.CONSTANT_MONITOR_LAG * 1e3
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    x = minutes["minute"] + 0.5
    frame_min = (diode.vsyncs - diode.vsyncs[0]) / 60

    # coverage, unexplained edges, short vsync pulses
    ax = axes[0]
    ax.fill_between(x, 0, minutes["coverage"], step="mid", color="0.7", label="frames with a diode edge (fraction)")
    ax.plot(x, minutes["short_vsync_pulse_frac"], ".-", ms=2, color="tab:green", lw=1, label=f"vsync pulses < {SHORT_VSYNC_PULSE_S * 1e3:g} ms (fraction)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction per minute")
    twin = ax.twinx()
    twin.plot(x, minutes["unexplained_edges"], ".-", ms=2, color="tab:orange", lw=1, label="unexplained diode edges / min (right)")
    twin.set_yscale("symlog", linthresh=10)
    twin.set_ylim(0, 5000)
    twin.set_ylabel("edges / min")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=6, loc="lower left", ncol=3)
    ax.set_title(
        f"raw photodiode vs vsyncs: {np.isfinite(diode.latency).mean():.0%} of frames have a diode edge; "
        f"{int((~diode.is_explained).sum())} of {len(diode.edges)} edges unexplained",
        fontsize=7,
    )

    # chunk photodiode latency vs constant lag
    ax = axes[1]
    ok = np.isfinite(chunk_latency)
    step = max(1, int(ok.sum() // 20000))
    ax.plot(frame_min[ok][::step], chunk_latency[ok][::step], ".", ms=1, color="tab:blue", label="photodiode latency, npc_stim estimator on 60-s chunks")
    ax.axhline(lag_ms, color="r", lw=0.8, ls="--", label=f"constant lag ({lag_ms:.2f} ms)")
    seen = set()
    for c in chunks:
        if c["ok"]:
            continue
        ax.axvspan(
            c["start_min"], c["stop_min"], color=CHUNK_CATEGORY_COLORS[c["category"]], alpha=0.3, lw=0,
            label=None if c["category"] in seen else CHUNK_CATEGORY_LABELS[c["category"]],
        )
        seen.add(c["category"])
    ax.set_ylim(lag_ms - FRAME_MS, lag_ms + 1.5 * FRAME_MS)
    ax.set_ylabel("display time - vsync (ms)")
    ax.legend(fontsize=6, loc="upper left", ncol=2, markerscale=6)
    n_ok = sum(c["ok"] for c in chunks)
    ax.set_title(
        f"photodiode latency where the diode can be read: {n_ok}/{len(chunks)} chunks, {ok.mean():.0%} of frames"
        + (f"; median {np.median(chunk_latency[ok]):.2f} ms, |latency - constant lag| p99 {np.percentile(np.abs(chunk_latency[ok] - lag_ms), 99):.1f} ms" if ok.any() else ""),
        fontsize=7,
    )

    # rising vs falling edges
    ax = axes[2]
    rise, fall = minutes["rise_latency_ms"], minutes["fall_latency_ms"]
    ax.plot(x, rise, ".-", ms=3, lw=0.8, color="tab:blue", label="rising diode edges")
    ax.plot(x, fall, ".-", ms=3, lw=0.8, color="tab:orange", label="falling diode edges")
    ax.plot(x, (rise + fall) / 2, "k-", lw=1, label="mean of the two")
    ax.axhline(lag_ms, color="r", lw=0.8, ls="--")
    ax.set_ylim(lag_ms - FRAME_MS / 2 - 1, lag_ms + FRAME_MS / 2 + 1)
    ax.set_ylabel("edge - vsync + 8 ms (ms)\nmedian per minute")
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.set_title(
        "rising vs falling diode edges (first edge within +-half a frame of the expected flip): a display delay moves both the same way; "
        "opposite drift = sensor/comparator",
        fontsize=7,
    )

    # per-stim shift vs diode noise
    ax = axes[3]
    if latency_result is None:
        ax.text(0.5, 0.5, f"no latency check: {latency_error}", ha="center", va="center", transform=ax.transAxes, fontsize=7, wrap=True)
    else:
        _plot_stim_shifts(ax, latency_result, float(diode.vsyncs[0]))
    twin = ax.twinx()
    twin.plot(x, minutes["unexplained_edges"], color="tab:orange", lw=0.8, alpha=0.8, label="unexplained diode edges / min (right)")
    twin.plot(x, minutes["short_vsync_pulse_frac"] * 600, color="tab:green", lw=0.8, alpha=0.8, label="short vsync pulses (fraction x 600, right)")
    twin.set_ylim(0, 700)
    twin.legend(fontsize=6, loc="lower right")
    ax.set_title("per-stim onset shift of the visual response (template: first quarter of stims; dotted: 0, +-1, 2 frames) vs diode noise", fontsize=7)
    ax.set_xlabel("minutes into block")
    ax.set_xlim(0, frame_min[-1])
    fig.suptitle(
        f"{session.id} {stem} | display times: {block.source}\n"
        + _wrap(f"photodiode not used: {block.reason}", 170),
        fontsize=7.5,
    )
    fig.set_layout_engine("tight")
    return fig


# constant-lag blocks: frame index check (criterion a) ----------------------------------- #


def get_frame_index_check(block: npc_stim.BlockDisplayTimes) -> Any | None:  # noqa: ANN401
    """The block's `npc_stim.FrameIndexCheck` (as made with its display times, else made
    here); None if this version of npc_stim has no such check."""
    check = getattr(block, "frame_index_check", None)
    if check is not None:
        return check
    import npc_stim.display_times as dt

    if not hasattr(dt, "get_frame_index_check"):
        return None
    return dt.get_frame_index_check(block.vsync_times, block.stim_path)


def _frame_index_figure(
    session: npc_sessions.DynamicRoutingSession,
    stem: str,
    block: npc_stim.BlockDisplayTimes,
    check: Any | None,  # noqa: ANN401
) -> tuple[plt.Figure, dict[str, Any]]:
    """Long (dropped) frames in the vsyncs vs the stim file's `frameIntervals`, with the
    frames that don't match listed."""
    import npc_stim.display_times as dt

    vsyncs = np.asarray(block.vsync_times, dtype=float)
    stim_data = npc_stim.stim.get_h5_stim_data(block.stim_path)
    try:
        stim_intervals = np.asarray(stim_data["frameIntervals"][:], dtype=float)
    finally:
        with contextlib.suppress(Exception):
            stim_data.close()
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    ax_c = fig.add_subplot(gs[0, :])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_step = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")
    listing: list[dict[str, Any]] = []
    lines = [_summary(check) or "frame index check: not available in this version of npc_stim"]
    if len(stim_intervals) == len(vsyncs) - 1 and len(vsyncs) > 2:
        period = float(np.median(stim_intervals))
        threshold = getattr(dt, "LONG_FRAME_THRESHOLD", 1.5) * period
        max_step = getattr(dt, "FRAME_INDEX_CHECK_MAX_STEP", 0.25) * period
        w = getattr(dt, "FRAME_INDEX_CHECK_WINDOW", 3)
        sync_intervals = np.diff(vsyncs)
        c = (vsyncs - vsyncs[0]) - np.concatenate(([0.0], np.cumsum(stim_intervals)))
        long_sync = sync_intervals > threshold
        long_stim = stim_intervals > threshold
        long_frames = np.flatnonzero(long_sync | long_stim)
        steps = np.array(
            [
                np.median(c[k + 1 : k + 1 + w]) - np.median(c[max(k + 1 - w, 0) : k + 1])
                for k in long_frames
            ]
        )
        unmatched = np.abs(steps) > max_step
        both = long_sync[long_frames] & long_stim[long_frames]
        minutes = (vsyncs - vsyncs[0]) / 60
        # time on sync minus time in the stim file, without the clock-rate difference
        drift = c[-1] / (vsyncs[-1] - vsyncs[0])
        detrended = (c - drift * (vsyncs - vsyncs[0])) * 1e3
        sub = max(1, len(c) // 50000)
        ax_c.plot(minutes[::sub], detrended[::sub], lw=0.5, color="0.4")
        ax_c.plot(minutes[long_frames[both]], detrended[long_frames[both]], "|", ms=6, color="tab:blue", label=f"long frame in both at the same index ({int(both.sum())})")
        ax_c.plot(minutes[long_frames[~both]], detrended[long_frames[~both]], "x", ms=5, color="tab:orange", label=f"long frame in one only: +-1 index or unpaired ({int((~both).sum())})")
        ax_c.plot(minutes[long_frames[unmatched]], detrended[long_frames[unmatched]], "o", ms=7, mfc="none", color="r", label=f"unmatched: step > {max_step * 1e3:.1f} ms ({int(unmatched.sum())})")
        ax_c.set_xlabel("minutes into block")
        ax_c.set_ylabel("frame time on sync minus\nin stim file (ms)")
        ax_c.set_title(
            f"vsync times vs cumulative stim-file frameIntervals (clock-rate difference {drift * 1e6:.1f} ppm removed): "
            "a refresh in one and not the other is a step of ~16.7 ms",
            fontsize=7,
        )
        ax_c.legend(fontsize=6, loc="upper left", ncol=3)
        ax_c.margins(y=0.25)
        lim = max(sync_intervals[long_frames].max(initial=threshold), stim_intervals[long_frames].max(initial=threshold)) * 1e3 * 1.1
        ax_scatter.plot([0, lim], [0, lim], "k:", lw=0.5)
        ax_scatter.plot(stim_intervals[long_frames][~unmatched] * 1e3, sync_intervals[long_frames][~unmatched] * 1e3, ".", ms=4, color="tab:blue")
        ax_scatter.plot(stim_intervals[long_frames][unmatched] * 1e3, sync_intervals[long_frames][unmatched] * 1e3, "o", ms=5, mfc="none", color="r")
        for v in (threshold * 1e3,):
            ax_scatter.axvline(v, color="0.6", lw=0.5, ls="--")
            ax_scatter.axhline(v, color="0.6", lw=0.5, ls="--")
        ax_scatter.set_xlabel("interval in stim file (ms)")
        ax_scatter.set_ylabel("interval between vsyncs (ms)")
        ax_scatter.set_title(f"the {len(long_frames)} long frames (> 1.5 refreshes, dashed, in either)", fontsize=7)
        if len(steps):
            span = max(np.abs(steps).max() * 1e3 * 1.1, max_step * 1e3 * 1.2)
            ax_step.hist(steps * 1e3, bins=np.linspace(-span, span, 61), color="0.5")
        for v in (-max_step * 1e3, max_step * 1e3):
            ax_step.axvline(v, color="r", lw=0.8, ls="--")
        ax_step.set_yscale("symlog", linthresh=1)
        ax_step.set_xlabel("step across the long frame, sync minus stim file (ms)")
        ax_step.set_ylabel("long frames")
        ax_step.set_title(f"time across each long frame: limit +-{max_step * 1e3:.1f} ms (red)", fontsize=7)
        for k, step, bad in zip(long_frames, steps, unmatched):
            if bad or not (long_sync[k] and long_stim[k]):
                listing.append(
                    {
                        "frame": int(k),
                        "minute": round(float(minutes[k]), 3),
                        "sync_interval_ms": round(float(sync_intervals[k] * 1e3), 2),
                        "stim_file_interval_ms": round(float(stim_intervals[k] * 1e3), 2),
                        "step_ms": round(float(step * 1e3), 2),
                        "unmatched": bool(bad),
                    }
                )
        bad_rows = [r for r in listing if r["unmatched"]]
        other_rows = [r for r in listing if not r["unmatched"]]

        def fmt(r: dict[str, Any]) -> str:
            return (
                f"  frame {r['frame']:>7} ({r['minute']:6.2f} min): sync {r['sync_interval_ms']:6.1f} ms, "
                f"stim file {r['stim_file_interval_ms']:6.1f} ms, step {r['step_ms']:+6.2f} ms"
            )

        lines.append(f"unmatched long frames (fail the check): {len(bad_rows)}")
        lines += [fmt(r) for r in bad_rows[:8]] + (["  ..."] if len(bad_rows) > 8 else [])
        lines.append(
            f"long in one only, time across them agrees (allowed: +-1 index, or a late vsync edge): {len(other_rows)}"
        )
        lines += [fmt(r) for r in other_rows[:6]] + (["  ..."] if len(other_rows) > 6 else [])
    else:
        for ax in (ax_c, ax_scatter, ax_step):
            ax.axis("off")
        lines.append(f"{len(vsyncs)} vsyncs, {len(stim_intervals)} frameIntervals in the stim file: not compared")
    ax_text.text(
        0, 1, "\n".join(_wrap(line, 160) for line in lines), va="top", ha="left",
        fontsize=6.5, family="monospace", transform=ax_text.transAxes,
    )
    passed = getattr(check, "passed", None)
    fig.suptitle(
        f"{session.id} {stem} | long (dropped) frames, vsyncs vs stim file: "
        + ("not checked" if passed is None else "PASSED" if passed else "FAILED"),
        fontsize=8,
    )
    fig.set_layout_engine("tight")
    return fig, {"frames_listed": listing[:200]}


# constant-lag blocks: validity record ---------------------------------------------------- #


def _validity_record(
    session: npc_sessions.DynamicRoutingSession,
    stem: str,
    block: npc_stim.BlockDisplayTimes,
    frame_check: Any | None,  # noqa: ANN401
    latency_result: Any | None,  # noqa: ANN401
    latency_error: str | None,
) -> dict[str, Any]:
    """The block's validity record as a dict of text: from `session.stim_validity` if this
    version of npc_sessions has it, otherwise the same two checks evaluated here."""
    validity = get_stim_validity(session, stem)
    record: dict[str, Any] = {
        "stim": stem,
        "display_time_source": block.source,
        "photodiode_not_used": block.reason,
        "frame_index_check": _summary(frame_check),
        "latency_check": _summary(latency_result) or f"not run: {latency_error}",
    }
    if validity is not None:
        record |= {
            "from": "session.stim_validity",
            "is_valid": bool(validity.is_valid),
            "reason": str(validity.reason),
            "display_time_source": str(getattr(validity, "display_time_source", block.source)),
            "summary": _summary(validity),
        }
        for name in ("frame_index_check", "latency_check", "reference_check", "diode_check"):
            if (check := getattr(validity, name, None)) is not None:
                record[name] = _summary(check)
            elif name not in record:
                record[name] = None
    else:
        a = getattr(frame_check, "passed", None)
        b = getattr(latency_result, "verdict", None)
        failed = []
        if a is not True:
            failed.append("frame index check " + ("not run" if a is None else "failed"))
        if b != "passed":
            failed.append(f"latency check {b or 'not run'}")
        record |= {
            "from": "evaluated by this plot from the frame index and latency checks "
            "(this version of npc_sessions has no session.stim_validity)",
            "is_valid": not failed,
            "reason": "; ".join(failed) or "frame index and latency checks passed",
        }
    return record


def _validity_figure(
    session: npc_sessions.DynamicRoutingSession, record: dict[str, Any]
) -> plt.Figure:
    valid = record["is_valid"]
    rows = [
        ("verdict", f"{'VALID' if valid else 'INVALID'}: {record['reason']}"),
        ("record from", record["from"]),
        ("display times", f"{record['display_time_source']} (photodiode not used: {record['photodiode_not_used']})"),
        ("(a) frame indices", record.get("frame_index_check")),
        ("(b) visual latency", record.get("latency_check")),
        ("reference block", record.get("reference_check")),
        ("photodiode", record.get("diode_check")),
    ]
    lines: list[str] = []
    for name, value in rows:
        wrapped = textwrap.wrap(str(value if value is not None else "-"), 118) or ["-"]
        lines.append(f"{name:<20}{wrapped[0]}")
        lines += [" " * 20 + w for w in wrapped[1:]]
        lines.append("")
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.115 * len(lines)))
    ax.axis("off")
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7, family="monospace", transform=ax.transAxes)
    fig.suptitle(
        f"{session.id} {record['stim']} | validity of visual stim times: {'VALID' if valid else 'INVALID'}",
        fontsize=9, color="tab:green" if valid else "tab:red", fontweight="bold",
    )
    fig.set_layout_engine("tight")
    return fig


def get_constant_lag_stems(session: npc_sessions.DynamicRoutingSession) -> list[str]:
    """Task and RFMapping stims whose display times are (or, for an excluded block, would
    be) vsync + constant lag."""
    stems = []
    for stem in get_visual_stim_stems(session):
        block = get_display_block(session, stem)
        if not isinstance(block, Exception) and block.source == "constant_lag":
            stems.append(stem)
    return stems


def plot_constant_lag_display_times(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[plt.Figure | dict, ...] | None:
    """Only for sessions with a task or RFMapping block whose photodiode couldn't be used,
    so that visual stim times are vsync + constant lag: what the photodiode did, and whether
    the block is valid. Works without a trials table (excluded blocks).

    Three figures per block, then the stats (dict):

    1. per minute of the block (shared time axis):
       - fraction of frames with a diode edge, unexplained diode edges, short vsync pulses
       - photodiode latency from npc_stim's estimator on 60-s chunks, where it works, vs the
         constant lag: should be within ~2 ms. Red bands: chunks with fewer diode flips than
         vsyncs (the display skipped frames: display times unreliable from there)
       - rising vs falling diode edges: opposite drift is a weakening diode signal, not a
         display delay
       - per-stim onset shift of the visual response vs diode noise: a step of 1-2 frames
         is a late display
    2. long (dropped) frames in the vsyncs vs the stim file's `frameIntervals` (acceptance
       criterion a), with the frames that don't match listed
    3. the validity record: every check's summary, the verdict and the reason

    The quarterly PSTHs (acceptance criterion b) are in `quartile_visual_latency`.
    """
    if not session.is_sync:
        return None
    stems = get_constant_lag_stems(session)
    if not stems:
        return None
    figs: list[plt.Figure | dict] = []
    stats: dict[str, dict[str, Any]] = {}
    for stem in stems:
        block = get_display_block(session, stem)
        assert isinstance(block, npc_stim.BlockDisplayTimes)
        latency_result, latency_error = None, None
        if session.is_sorted:
            try:
                latency_result, _ = get_quartile_latency(session, stem)
            except Exception as exc:  # noqa: BLE001
                latency_error = repr(exc)
        else:
            latency_error = "session isn't sorted"
        diode = get_diode_data(session, block)
        minutes = get_diode_minutes(diode)
        t = time.time()
        chunk_latency, chunks = get_chunk_latencies(diode, f"{session.id}_{stem}")
        chunk_seconds = time.time() - t
        logger.info(f"{session.id} {stem}: chunked photodiode estimate in {chunk_seconds:.0f} s")
        figs.append(
            _diode_figure(
                session, stem, block, diode, minutes, chunk_latency, chunks,
                latency_result, latency_error,
            )
        )
        frame_check = get_frame_index_check(block)
        frames: dict[str, Any] = {}
        try:
            fig, frames = _frame_index_figure(session, stem, block, frame_check)
            figs.append(fig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{session.id} {stem}: frame index figure failed: {exc!r}")
            frames = {"error": repr(exc)}
        record = _validity_record(session, stem, block, frame_check, latency_result, latency_error)
        figs.append(_validity_figure(session, record))
        validity = get_stim_validity(session, stem)
        stats[stem] = {
            "validity": record,
            "validity_record": (
                validity.to_dict() if validity is not None and hasattr(validity, "to_dict") else None
            ),
            "frame_index_check": (
                None
                if frame_check is None
                else {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in dataclasses.asdict(frame_check).items()
                }
                | frames
            ),
            "latency_check": (
                {"error": latency_error}
                if latency_result is None
                else {
                    k: latency_result.to_dict()[k]
                    for k in ("verdict", "reason", "vis_range_ms", "aud_range_ms", "frac_shifted")
                }
            ),
            "diode": {
                "frames_per_diode_flip": diode.frames_per_flip,
                "n_frames": int(len(diode.vsyncs)),
                "n_diode_edges": int(len(diode.edges)),
                "frac_frames_with_diode_edge": float(np.isfinite(diode.latency).mean()),
                "n_unexplained_edges": int((~diode.is_explained).sum()),
                "frac_short_vsync_pulses": float(
                    np.mean((diode.vsyncs - diode.vsync_rising) < SHORT_VSYNC_PULSE_S)
                ),
                "chunks": _chunk_summary(chunk_latency, chunks, chunk_seconds),
                "per_minute": {
                    k: [None if not np.isfinite(x) else round(float(x), 4) for x in v]
                    for k, v in minutes.items()
                },
            },
        }
    figs.append(
        {
            "session_id": str(session.id),
            "is_valid": {stem: s["validity"]["is_valid"] for stem, s in stats.items()},
            "blocks": stats,
        }
    )
    return tuple(figs)
