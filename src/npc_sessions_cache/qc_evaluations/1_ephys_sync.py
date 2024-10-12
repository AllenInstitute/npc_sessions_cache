import functools
from npc_sessions_cache.plots.sync import plot_barcode_intervals
from npc_sessions_cache.plots.ephys import plot_sensory_responses

instructions = {
    plot_barcode_intervals: """
    - all points in the center panel should be very close to 30 seconds (deviation < 0.1s)
    - all lines in the right panel should overlap almost perfectly (black line being longer is acceptable)
    """,
    plot_sensory_responses: """
    """,
}