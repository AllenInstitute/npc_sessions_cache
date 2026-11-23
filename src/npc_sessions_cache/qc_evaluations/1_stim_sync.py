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
