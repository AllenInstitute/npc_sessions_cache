from npc_sessions_cache.plots.sync import (
    plot_vsync_interval_dist,
    plot_vsync_intervals,
    plot_frametime_intervals,
    plot_diode_flip_intervals,
    plot_sync_timing_report,
)
from npc_sessions_cache.plots.timing import (
    plot_long_vsync_occurrences,
)
from npc_sessions_cache.plots.sync_patch import plot_trial_onset_frame_consistency
from npc_sessions_cache.plots.display_qc import (
    plot_constant_lag_display_times,
    plot_quartile_visual_latency,
)
from npc_sessions_cache.plots.long_frames import plot_task_long_frame_alignment
from npc_sessions_cache.plots.late_frame import plot_late_stim_frame_alignment
from npc_sessions_cache.plots.stim_frame_timing import plot_task_stim_frame_timing

instructions = {
    plot_vsync_interval_dist: """
    - for the Gaussian distribution with mean on 16.7ms, all points should be within Xms of the mean
    """,
    plot_vsync_intervals: "",
    plot_frametime_intervals: "",
    plot_diode_flip_intervals: "",
    plot_sync_timing_report: """
    - sessions with replayed sync intervals: every stim block should be clean or patched, not rejected
    - interpolated frames (red crosses in the frametime plot) should be a small fraction of frames
    """,
    plot_long_vsync_occurrences: "",
    plot_trial_onset_frame_consistency: """
    - red points (trials with an onset in an invalid interval) should be inside the grey band (median +- half a frame)
    - a red point near a dotted line (+-1 frame) is a 1-frame offset: fail
    """,
    plot_quartile_visual_latency: """
    - sorted sessions with a task or RFMapping block: one figure per block, stats in the json
    - the visual half-max latency should be the same in every quarter of the block (range <= 8 ms); the 4 PSTHs should overlap
    - a step of ~17 or ~33 ms in latency or in the per-stim shifts, with a stable auditory latency, is a late display: fail
    - json `review` is True if the visual range is > 8 ms, or > 10% of task stims are shifted by > 8 ms while the auditory latency is stable
    - blocks with few responsive units (< 10) have noisy per-stim shifts: go by the quarterly latencies
    - green/grey dashed lines (right axis): evoked and baseline rate of the visual units per quarter, relative to Q1. A timing error shifts the latency without changing the rates; a change of state (arousal, running, engagement) changes the rates, and the half-max latency with them (weaker responses peak later). In a photodiode block that passes the long-frame alignment, a latency change of < 1 frame that goes with a rate change and a stable auditory latency is not a timing error
    """,
    plot_late_stim_frame_alignment: """
    - task trials whose visual stim frame was displayed a refresh later than usual (photodiode): the response should follow the photodiode, not the vsync
    - left (aligned to stim_start_time): late and normal PSTHs overlap, shift ~0 ms; right (aligned to vsync): late trials ~17 ms later
    - fail: late trials respond ~17 ms earlier than normal trials relative to stim_start_time (their display times are a frame late)
    - untestable with < 5 late trials or < 5 visually responsive units
    """,
    plot_task_long_frame_alignment: """
    - block-level check of the task's frame indexing: long (dropped) frames in the stim file's frameIntervals vs in the display times. It can't tell whether a stim onset is right: use task_stim_frame_timing for that
    - FAIL: frame counts differ, or the global index offset (summary text) isn't 0 (every frame shifted), or a run of consecutive long frames all off by the same non-zero index
    - expected, not a fail: isolated orange/red dots. frameIntervals are flip-return times on the stim computer, not display times, and in box-1 blocks the frame between two measured diode edges is placed by rule, so a hold can be put on the neighbouring frame. Each such pair leaves at most one frame a refresh off, and never propagates: task_stim_frame_timing shows whether any of them is a stim onset/offset frame
    """,
    plot_task_stim_frame_timing: """
    - every visual stim onset and offset frame of the task: is its display time supported by that frame's own sync edges?
    - FAIL (json verdict 'failed'): any red 'unsupported' onset/offset frame = a stim_start_time/stim_stop_time a refresh or more off with nothing on sync to explain it
    - orange 'late falling edge' frames are fine: the script stalled after the flip, so the vsync falling edge is late (long frame-signal pulse) but the frame was displayed on time; they sit left of the grey band, above the dashed line, near the dotted diagonal
    - purple 'vsync + latency' frames were given vsync + median latency by npc_stim (frame_source 2): review, not a fail
    - review: an onset/offset frame inside a run of >= 2 off frames (grey dots in the top panel next to a coloured one)
    - top panel: grey dots (off frames that aren't stim events) are mostly late falling edges; a horizontal band of dots near +-17 ms would be a stretch of frames a refresh off (npc_stim rejects blocks with such a stretch longer than ~150 frames)
    """,
    plot_constant_lag_display_times: """
    - only sessions with a task or RFMapping block on vsync + constant lag (photodiode not usable)
    - photodiode latency on 60-s chunks (blue) should be within ~2 ms of the constant lag (red) wherever the diode can be read
    - red bands (fewer diode flips than vsyncs) in 3 or more consecutive minutes: the display skipped frames, expect the block to be invalid
    - rising and falling edges drifting apart is a weak diode signal, not a display delay
    - long-frame figure: no unmatched frames (red circles)
    - last figure per block: the validity record (every check, verdict, reason)
    """,
}
