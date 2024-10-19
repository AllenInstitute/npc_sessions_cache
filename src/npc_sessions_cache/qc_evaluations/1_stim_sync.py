from npc_sessions_cache.plots.sync import (
    plot_vsync_interval_dist,
    plot_vsync_intervals,
    plot_frametime_intervals,
    plot_diode_flip_intervals,
)
from npc_sessions_cache.plots.timing import (
    plot_long_vsync_occurrences,
)

instructions = {
    plot_vsync_interval_dist: """
    - for the Gaussian distribution with mean on 16.7ms, all points should be within Xms of the mean
    """,
    plot_vsync_intervals: "",
    plot_frametime_intervals: "",
    plot_diode_flip_intervals: "",
    plot_long_vsync_occurrences: "",
}
