from __future__ import annotations

import contextlib
import datetime
import io
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import npc_mvr
import npc_sessions
import numpy as np
import numpy.typing as npt
import rich
import upath

if TYPE_CHECKING:
    pass

import npc_sessions_cache.plots.plot_utils as plot_utils


def _excluded_cameras(session: npc_sessions.DynamicRoutingSession) -> dict[str, str]:
    """Cameras npc_mvr leaves out of `video_paths` (npc_mvr>=0.1.20), with the reason."""
    return dict(getattr(session.mvr, "excluded_cameras", {}))


def _is_eye_tracking_available(session: npc_sessions.DynamicRoutingSession) -> bool:
    return session.is_video and "eye" in session.mvr.video_paths


def plot_camera_frame_counts(
    session: npc_sessions.DynamicRoutingSession,
) -> dict | None:
    """Frame counts per camera from npc_mvr (all cameras, including excluded ones):
    exposures on sync, frames in the video, lost frames, unaccounted exposures (neither in
    the video nor in `LostFrames`: frame times may drift), exposures inserted in unknown
    sync intervals, and why a camera is excluded (null if used)."""
    if not session.is_sync:
        return None
    mvr = session.mvr
    if not getattr(mvr, "all_video_paths", mvr.video_paths):
        return None
    camera_frame_times = getattr(mvr, "camera_frame_times", None)
    if camera_frame_times is None:  # npc_mvr<0.1.20
        return None
    excluded = _excluded_cameras(session)
    summary: dict[str, dict] = {}
    for camera_name, c in camera_frame_times.items():
        summary[camera_name] = {
            k: (list(map(list, v)) if k == "unknown_intervals" else v)
            for k, v in c.summary().items()
        } | {"excluded_reason": excluded.get(camera_name)}
    for camera_name, reason in excluded.items():
        summary.setdefault(camera_name, {"excluded_reason": reason})
    return summary


def _plot_video_info(
    session: npc_sessions.DynamicRoutingSession,
    capture_stdout: bool = True,
) -> str | None:
    "Not a plot: prints info to stdout"

    augmented_camera_info = session.mvr.augmented_camera_info

    captured_output = io.StringIO()
    with (
        contextlib.redirect_stdout(captured_output)
        if capture_stdout
        else contextlib.nullcontext()
    ):
        for camera, info in augmented_camera_info.items():
            rich.print(f"[bold]{camera} camera stats[bold]")
            if reason := info.get("excluded_reason"):
                rich.print(f"[red]Excluded: {reason}[/red]")

            frame_rate = info["FPS"]
            frame_rate_string = plot_utils.add_valence_to_string(
                f"Frame Rate: {frame_rate} \t",
                frame_rate,
                abs(frame_rate - 60) < 0.01,
                abs(frame_rate - 60) > 0.05,
            )

            lost_frame_percentage = (
                100 * info["FramesLostCount"] / info["FramesRecorded"]
            )
            lost_frame_string = plot_utils.add_valence_to_string(
                f"Lost frame percentage: {np.round(lost_frame_percentage, 3)} \t",
                lost_frame_percentage,
                lost_frame_percentage < 0.01,
                lost_frame_percentage > 0.1,
            )
            if "expected_minus_actual" not in info:
                rich.print(frame_rate_string + lost_frame_string)
                continue
            frame_diff_from_expected = info["expected_minus_actual"]
            frame_diff_string = plot_utils.add_valence_to_string(
                f"Frames expected minus actual: {frame_diff_from_expected}",
                frame_diff_from_expected,
                abs(frame_diff_from_expected) < 1,
                frame_diff_from_expected > 20,
            )

            rich.print(frame_rate_string + lost_frame_string + frame_diff_string)
    if capture_stdout:
        output = captured_output.getvalue()
        captured_output.close()
        return output
    else:
        return None


def plot_pupil_area_with_running(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure | None:
    if not _is_eye_tracking_available(session):
        return None
    np.array(session._eye_tracking.timestamps)[
        np.isnan(session._eye_tracking.pupil_area)
    ]
    valid_running_times = ~np.isnan(session._running_speed.timestamps) & ~np.isnan(
        session._running_speed.data
    )
    # for xlim in range(0, round(real_timestamps[-1]), 1000):
    plt.figure()
    plt.plot(
        session._running_speed.timestamps[valid_running_times],
        session._running_speed.data[valid_running_times]
        / np.nanmax(session._running_speed.data),
        lw=0.2,
    )
    plt.plot(
        session._eye_tracking.timestamps,
        (
            v := session._eye_tracking.pupil_area
            / np.nanmax(session._eye_tracking.pupil_area)
        )
        - np.nanmedian(v),
        lw=0.2,
    )
    plt.gca().legend(["running speed", "pupil area"])
    plt.gca().set_xlabel("time (s)")
    # plt.gca().set_aspect(100)
    plt.gcf().set_size_inches(15, 5)
    plt.gca().get_yaxis().set_visible(False)
    plt.title(f"{session.id}")
    return plt.gcf()


def plot_pupil_response(
    session: npc_sessions.DynamicRoutingSession,
) -> plt.Figure | None:
    if not session.is_task or not _is_eye_tracking_available(session):
        return None
    median_with_shaded_std = False
    dur = 2.5  # s
    pupil_area = session._eye_tracking.pupil_area
    real_timestamps = session._eye_tracking.timestamps

    query = "is_vis_nontarget & ~is_response"
    t0 = session.trials[:].query(query)["stim_start_time"].to_numpy()
    t1 = t0 + dur
    trial_pupil_size = [
        (
            pupil_area[slice(start, stop)]
            - np.nanmean(pupil_area[slice(baseline_0, start)])
            if 0 <= start < stop <= len(pupil_area)
            else []
        )
        for baseline_0, start, stop in np.searchsorted(
            real_timestamps, np.array([t0 - 0.5, t0, t1]).T
        )
    ]
    if median_with_shaded_std:
        trial_pupil_size: npt.NDArray = np.array([t for t in trial_pupil_size if t])
        x = np.arange(0, dur, dur / trial_pupil_size.shape[1])
        y = np.nanmedian(trial_pupil_size, 0)
        plt.fill_between(
            x,
            y - np.nanstd(trial_pupil_size, 0),
            y + np.nanstd(trial_pupil_size, 0),
            alpha=0.2,
        )
        plt.plot(x, y)
        # plt.show()
    else:
        for arr in trial_pupil_size:
            if len(arr) == 0:
                # stim outside video frames (e.g. a block of frames lost by the camera)
                continue
            x = np.arange(0, dur, dur / len(arr))
            y = arr
            plt.plot(x, y, lw=0.2, alpha=0.5, c="k")
        # number of camera frames per trial window varies by +-1: truncate to shortest
        valid = [t for t in trial_pupil_size if np.any(t)]
        if valid:
            n = min(len(t) for t in valid)
            plt.plot(
                np.arange(n) * (dur / n),
                np.nanmedian([t[:n] for t in valid], 0),
                lw=2,
            )

    plt.gca().set(
        xlabel="time from vis stim onset (s)",
        ylabel="pupil area minus baseline (pixels)",
    )
    plt.title(f"pupil response to stim\n{query}\n{session.id}", fontsize=8)
    plt.axhline(y=0, c="k", lw=0.5, ls="--")
    plt.axvline(x=0, c="k", lw=0.5, ls="--")
    plt.gca().set_ylim(-500, 500)
    plt.gca().set_xlim(-0.1, dur + 0.1)
    plt.gcf().set_size_inches(3, 4)
    return plt.gcf()


def plot_camera_frames(
    session: npc_sessions.DynamicRoutingSession,
    paths: Iterable[upath.UPath] | None = None,
    num_frames_to_grab: int = 5,
) -> plt.Figure | None:
    """Just plots evenly spaced frames, no concept of epochs.

    video frames across cameras aren't synced .
    """
    if paths is None:
        paths = session.video_paths

    paths = tuple(paths)
    if not paths:
        return None
    if len(paths) == 3:  # beh, eye, face
        paths = (paths[0], paths[-1], paths[-2])  # beh, face, eye
    fig = plt.figure(
        figsize=[10, 3 * len(paths)], constrained_layout=True, facecolor="0.5"
    )
    gs = gridspec.GridSpec(len(paths), num_frames_to_grab, figure=fig)
    gs.update(wspace=0.0, hspace=0.0)
    for idx, video_path in enumerate(paths):
        # get frames to plot

        v = npc_mvr.get_video_data(video_path)  # TODO open with upath from cloud

        frame_delta = np.ceil(v.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames_to_grab + 1)
        frames_of_interest = np.arange(
            v.get(cv2.CAP_PROP_FPS), v.get(cv2.CAP_PROP_FRAME_COUNT), frame_delta
        )

        for i, f in enumerate(frames_of_interest):
            v.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ret, frame = v.read()
            ax = fig.add_subplot(gs[idx, i])
            im = ax.imshow(frame)
            im.set_clim(0, 255)
            # ax.axis('off')
            ax.tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )
            ax.set_title(
                datetime.timedelta(seconds=f / v.get(cv2.CAP_PROP_FPS)), fontsize=10
            )
    return fig


def plot_lick_triggered_frames(
    session: npc_sessions.DynamicRoutingSession,
    trial_idx: int | None = None,
    lick_time: float | None = None,
) -> plt.Figure | None:
    if not (session.is_video and session.is_task):
        return None
    NUM_LICKS = 8 if (trial_idx is None and lick_time is None) else 1
    NUM_CAMERAS = 2  # 1 x face, 1 x body

    FRAMES_EITHER_SIDE_OF_LICK = 2
    """Symmetric around frame closest to lick"""
    FRAMES_PER_ROW = FRAMES_EITHER_SIDE_OF_LICK * 2 + 1

    ROWS_PER_VIDEO = 2  # 1 x camera frames, 1 x eventplots
    ROWS_PER_LICK = ROWS_PER_VIDEO * NUM_CAMERAS

    if lick_time is None:
        if "response_time" not in session.trials.colnames:
            # column is omitted when there are no responses (e.g. passive/opto sessions with no licks)
            return None
        response_times: npt.NDArray = (
            (session.trials[:].query("is_response").response_time.to_numpy())
            if trial_idx is None
            else (session.trials[trial_idx].response_time.to_numpy())
        )
        response_times = np.atleast_1d(response_times).astype(float)
        response_times = response_times[~np.isnan(response_times)]
        if not response_times.size:
            return None  # no lick responses
        lick_times = np.percentile(
            response_times,
            np.arange(0, 100, 100 / NUM_LICKS),
            method="closest_observation",
        )
    else:
        lick_times = [lick_time]

    fig = plt.figure(figsize=(12, 5 * NUM_LICKS), facecolor="0.5")
    # fig, axes = plt.subplots(NUM_LICKS * ROWS_PER_LICK, FRAMES_PER_ROW,)
    # constrained_layout=True,
    gs = gridspec.GridSpec(
        NUM_LICKS * ROWS_PER_LICK,
        FRAMES_PER_ROW,
        height_ratios=[1, 0.1] * NUM_CAMERAS * NUM_LICKS,
    )
    gs.update(wspace=0.0, hspace=0.0)

    # cameras npc_mvr excludes (untrustworthy frame times) aren't in `mvr.frame_times`
    camera_frame_times = session.mvr.frame_times
    # face + body cameras only (eye camera excluded; the order of paths isn't fixed)
    non_eye_videos = [
        (session.mvr.video_paths[camera_name], t)
        for camera_name, t in camera_frame_times.items()
        if camera_name != "eye"
    ][:NUM_CAMERAS]
    if not non_eye_videos:
        plt.close(fig)
        return None
    for vid_idx, (video_path, frame_times) in enumerate(non_eye_videos):
        for lick_idx, lick_time in enumerate(lick_times):
            v = npc_mvr.get_video_data(video_path)  # TODO open with upath from cloud

            closest_frame_index = np.nanargmin(np.abs(frame_times - lick_time))  # type: ignore[operator]
            frame_indices = np.arange(
                closest_frame_index - FRAMES_EITHER_SIDE_OF_LICK,
                closest_frame_index + FRAMES_EITHER_SIDE_OF_LICK + 1,
            )

            # video frames around lick time
            for frame_idx, frame_index in enumerate(frame_indices):
                v.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ret, frame = v.read()
                ax = fig.add_subplot(
                    gs[
                        (vid_idx * ROWS_PER_VIDEO) + (lick_idx * ROWS_PER_LICK),
                        frame_idx,
                    ]
                )
                if frame is None:
                    # e.g. lick time outside of video (camera stopped early)
                    ax.axis("off")
                    ax.text(0.5, 0.5, "no frame", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                    if frame_idx == FRAMES_EITHER_SIDE_OF_LICK:
                        ax.set_title(f"{video_path.stem}: no frame at {lick_time=:.1f} s", fontsize=8)
                    continue
                if "beh" in video_path.stem.lower():
                    x = slice(0, frame.shape[1] // 3)
                    y = slice(frame.shape[0] // 5, frame.shape[0] // 2)
                if "face" in video_path.stem.lower():
                    ymid, xmid = frame.shape[0] // 2, frame.shape[1] // 2
                    yspan, xspan = frame.shape[0] // 5, frame.shape[1] // 5
                    x = slice(xmid - xspan, xmid + xspan)
                    y = slice(ymid - yspan, ymid + yspan)
                im = ax.imshow(frame[y, x])
                im.set_clim(0, 255)
                ax.axis("off")
                if frame_index == closest_frame_index:
                    trial = (
                        np.searchsorted(  # type: ignore[call-overload]
                            session.trials[:].start_time, lick_time, "right"
                        )
                        - 1
                    )
                    ax.set_title(f"{trial=}, {lick_time=:.1f} s", fontsize=8)

            # markers for frame and lick times
            ax = fig.add_subplot(
                gs[
                    1 + (vid_idx * ROWS_PER_VIDEO) + (lick_idx * ROWS_PER_LICK),
                    :FRAMES_PER_ROW,
                ]
            )
            linekwargs = {"linewidths": 3}
            ax.eventplot(
                frame_times[frame_indices[(frame_indices >= 0) & (frame_indices < len(frame_times))]],
                label="frame",
                **linekwargs,
            )
            ax.eventplot([lick_time], color="red", label="lick", **linekwargs)
            ax.axis("off")
            if lick_idx == vid_idx == 0:
                ax.legend(fontsize=8, fancybox=True, ncol=2, loc="upper right")

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# lick-bout-triggered facemap SVs (side camera)
# --------------------------------------------------------------------------------------


def _lick_bout_onsets(lick_times: npt.ArrayLike, min_gap: float) -> npt.NDArray:
    """First lick of each bout: licks preceded by at least `min_gap` s without licks.

    >>> _lick_bout_onsets([0.0, 0.1, 0.2, 2.0, 2.1, 2.5, 4.0], min_gap=1.0).tolist()
    [0.0, 2.0, 4.0]
    """
    lick_times = np.sort(np.asarray(lick_times, dtype=float))
    return lick_times[np.diff(lick_times, prepend=-np.inf) >= min_gap]


def _camera_frame_loss(timestamps: npt.ArrayLike) -> dict[str, float]:
    """Frame loss estimated from a camera's sync-aligned frame times.

    - gap frames: frames missing from the video, inferred from inter-frame intervals longer
      than the median interval (round(dt / median_dt) - 1 per interval)
    - interior unsynced frames: frames in the video with no sync time (NaN), between the first
      and last synced frame
    - edge unsynced frames: NaN frame times before the first / after the last synced frame (e.g.
      frame 0, or video still recording after sync stopped) - reported, but not counted as loss

    >>> loss = _camera_frame_loss([np.nan, 0.0, 1.0, 2.0, 5.0, np.nan, 6.0, 7.0, np.nan])
    >>> loss['n_gap'], loss['n_nan_interior'], loss['n_nan_edge'], loss['n_expected']
    (2, 1, 2, 11)
    """
    ts = np.asarray(timestamps, dtype=float)
    finite = np.isfinite(ts)
    first, last = np.flatnonzero(finite)[[0, -1]]
    dt = np.diff(ts[finite])
    n_gap = int(np.clip(np.round(dt / np.median(dt)) - 1, 0, None).sum())
    n_nan_interior = int((~finite[first : last + 1]).sum())
    n_expected = len(ts) + n_gap
    return dict(
        n_gap=n_gap,
        n_nan_interior=n_nan_interior,
        n_nan_edge=int(first + (len(ts) - 1 - last)),
        n_expected=n_expected,
        pct_loss=100 * (n_gap + n_nan_interior) / n_expected,
        duration=float(ts[last] - ts[first]),
    )


def _event_triggered_rows(
    data: npt.ArrayLike,
    frame_times: npt.NDArray,
    event_times: npt.NDArray,
    window: tuple[float, float],
    n_columns: int,
    max_gap_rows: int = 600,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Sample the first `n_columns` of `data` (frames x columns) at the nearest synced frame to
    each event time + lag. Only contiguous blocks of rows around events are read, so `data` can
    be a lazy (e.g. zarr) array.

    Returns (lags, snippets [n_events x n_lags x n_columns], rows read [n_rows x n_columns]).
    Events whose window extends beyond the synced frames are dropped.
    """
    valid = np.flatnonzero(np.isfinite(frame_times))
    ts_valid = frame_times[valid]
    fps = 1 / np.median(np.diff(ts_valid))
    lags = np.arange(round(window[0] * fps), round(window[1] * fps) + 1) / fps
    event_times = event_times[
        (event_times + lags[0] >= ts_valid[0]) & (event_times + lags[-1] <= ts_valid[-1])
    ]
    if not len(event_times):
        return lags, np.empty((0, len(lags), n_columns)), np.empty((0, n_columns))

    sample_t = event_times[:, None] + lags[None, :]
    pos = np.clip(np.searchsorted(ts_valid, sample_t), 1, len(ts_valid) - 1)
    pos -= (sample_t - ts_valid[pos - 1]) < (ts_valid[pos] - sample_t)
    frame_idx = valid[pos]

    needed, inverse = np.unique(frame_idx, return_inverse=True)
    rows = np.empty((len(needed), n_columns), dtype=np.float32)
    breaks = np.flatnonzero(np.diff(needed) > max_gap_rows) + 1
    for block in np.split(np.arange(len(needed)), breaks):
        lo, hi = needed[block[0]], needed[block[-1]] + 1
        rows[block] = np.asarray(data[lo:hi, :n_columns])[needed[block] - lo]
    return lags, rows[inverse.reshape(frame_idx.shape)], rows


def plot_lick_bout_triggered_facemap_side(
    session: npc_sessions.DynamicRoutingSession,
    n_svs: int = 50,
    window: tuple[float, float] = (-1.0, 1.0),
    bout_gap: float = 1.0,
) -> plt.Figure | None:
    """Average of the top facemap motion SVs from the side (behavior) camera, aligned to the first
    lick of each lick bout (licks preceded by >= `bout_gap` s without licks; `is_likely_lick` only).

    - top: SVs 1-`n_svs`, change from pre-lick baseline in SD units
    - middle: SV1-5 (mean +/- SEM, baseline-subtracted)
    - bottom: mean |change| across all SVs
    Title reports side-camera frame loss (gap + interior unsynced frames as % of expected) and
    whether the side camera stopped early relative to the other cameras.
    """
    if not session.is_video or not session.is_task:
        return None
    licks = session._licks
    if licks is None or len(licks) == 0:
        return None
    lick_times = np.asarray(licks["timestamps"][:], dtype=float)
    n_licks = len(lick_times)
    if "is_likely_lick" in licks.colnames:
        lick_times = lick_times[np.asarray(licks["is_likely_lick"][:], dtype=bool)]
    onsets = _lick_bout_onsets(lick_times, bout_gap)

    camera_frame_times = session.mvr.frame_times
    if "behavior" not in camera_frame_times:
        return None
    frame_times = np.asarray(camera_frame_times["behavior"], dtype=float)
    try:
        facemap = npc_sessions.utils.get_facemap_output_from_s3(
            session.id, camera_name="behavior", array_name="motSVD"
        )
    except FileNotFoundError:
        return None  # facemap not run for this session
    if facemap.shape[0] != len(frame_times):
        raise ValueError(
            f"side camera facemap output has {facemap.shape[0]} frames, but there are {len(frame_times)} frame times"
        )

    loss = _camera_frame_loss(frame_times)
    longest = max(
        _camera_frame_loss(t)["duration"]
        for t in camera_frame_times.values()
        if np.isfinite(t).sum() > 1
    )
    pct_shorter = 100 * (1 - loss["duration"] / longest)

    n_svs = min(n_svs, facemap.shape[1])
    lags, snippets, rows = _event_triggered_rows(
        facemap, frame_times, onsets, window, n_svs
    )
    if len(snippets) == 0:
        return None  # no lick bouts within video
    mean = snippets.mean(axis=0)
    sem = (
        snippets.std(axis=0, ddof=1) / np.sqrt(len(snippets))
        if len(snippets) > 1
        else np.zeros_like(mean)
    )
    baseline = mean[lags < 0].mean(axis=0)
    sv_std = rows.std(axis=0)
    z = (mean - baseline) / np.where(sv_std > 0, sv_std, 1)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8, 10),
        sharex="col",
        gridspec_kw=dict(
            height_ratios=[3, 1.5, 1], width_ratios=[1, 0.025], hspace=0.08, wspace=0.03
        ),
    )
    (ax_h, cax), (ax_t, spare_t), (ax_m, spare_m) = axes
    spare_t.axis("off")
    spare_m.axis("off")

    vmax = np.nanpercentile(np.abs(z), 99) or 1
    im = ax_h.imshow(
        z.T,
        aspect="auto",
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(lags[0], lags[-1], n_svs + 0.5, 0.5),
        interpolation="nearest",
    )
    fig.colorbar(im, cax=cax, label="Δ from pre-lick baseline (SD units)")
    ax_h.set_ylabel("facemap SV #")

    loss_str = (
        f"side cam frame loss: {loss['pct_loss']:.3f}% "
        f"({loss['n_gap']} gap + {loss['n_nan_interior']} unsynced / {loss['n_expected']} expected"
        + (f"; {loss['n_nan_edge']} unsynced at start/end" if loss["n_nan_edge"] else "")
        + ")"
    )
    if pct_shorter > 1:
        loss_str += f"\nside cam stopped early: {pct_shorter:.0f}% shorter than longest camera"
    ax_h.set_title(
        f"{session.id}  |  side-camera facemap, lick-bout-onset-triggered average\n"
        f"{len(snippets)} bout onsets (≥{bout_gap:g} s since prev. lick; {len(onsets)} bouts, {n_licks} licks)\n"
        + loss_str,
        fontsize=9,
        color="r" if (loss["pct_loss"] > 0.1 or pct_shorter > 1) else "k",
    )

    for i in range(min(5, n_svs)):
        y = mean[:, i] - baseline[i]
        ax_t.plot(lags, y, lw=1.2, label=f"SV{i + 1}")
        ax_t.fill_between(lags, y - sem[:, i], y + sem[:, i], alpha=0.25)
    ax_t.set_ylabel("Δ SV (a.u.)")
    ax_t.legend(fontsize=7, ncol=5, loc="upper left", frameon=False)

    ax_m.plot(lags, np.abs(z).mean(axis=1), color="k")
    ax_m.set_ylabel(f"mean |Δ| SV1–{n_svs}\n(SD units)")
    ax_m.set_xlabel("time from first lick in bout (s)")
    ax_m.set_xlim(lags[0], lags[-1])
    for ax in (ax_h, ax_t, ax_m):
        ax.axvline(0, color="k", ls="--", lw=0.8)

    fig.subplots_adjust(left=0.12, right=0.9, top=0.91, bottom=0.06)
    return fig
