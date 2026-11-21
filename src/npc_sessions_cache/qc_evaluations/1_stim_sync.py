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
}
