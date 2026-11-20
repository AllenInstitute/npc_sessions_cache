import functools
from npc_sessions_cache.plots.sync import plot_barcode_intervals
from npc_sessions_cache.plots.ephys import plot_sensory_responses
from npc_sessions_cache.plots.responses import plot_fast_sensory_responses
from npc_sessions_cache.plots.sync_patch import plot_ephys_barcode_residuals

instructions = {
    plot_barcode_intervals: """
    - all points in the center panel should be very close to 30 seconds (deviation < 0.1s)
    - probes that failed timing QC are not plotted: they are listed (with a reason) in the printed output
    - all lines in the right panel should overlap almost perfectly (black line being longer is acceptable)
    """,
    plot_sensory_responses: """
    """,
    plot_fast_sensory_responses: """
    - rasters should show a sharp onset at a consistent latency in every trial
    - red rows (trials overlapping a replayed sync interval) should line up with the black rows
    - latency summary: no step (e.g. ~17 ms = 1 frame) in any trials, especially near red bands
    """,
    plot_ephys_barcode_residuals: """
    - every probe within +-0.5 ms (dashed; typically < 0.1 ms), flat across the session, including next to invalid intervals (red)
    - a step or outliers next to a device's dropped samples (dotted) or an invalid interval is a timing error
    """,
}
