from npc_sessions_cache.plots.sync import plot_barcode_intervals
from npc_sessions_cache.plots.sync_patch import (
    plot_edges_near_invalid_intervals,
    plot_patched_line_alternation,
    plot_sync_patch_spans,
)

instructions = {
    plot_barcode_intervals: """
    - all points in the left panel should be very close to 30 seconds (deviation < 0.1s)
    - an interval of ~60s (a multiple of 30s) is a missing barcode, e.g. dropped near a replayed sync interval: acceptable
    """,
    plot_sync_patch_spans: """
    - only for sessions with a sync patch that has invalid intervals, faults or a timebase
    - red spans are lines whose events are unknown there: expected for some lines (e.g. camera readout), check the reason column
    - error_bound_ms should be well under 1 frame (16.7 ms); anchor_residual_ms well under 1 ms
    """,
    plot_edges_near_invalid_intervals: """
    - patched edges (bottom of each pair) should continue the rhythm of the measured edges on both sides of the shaded interval
    - fail on a gap, a doubled edge or a step of 1 frame in vsync_stim / stim_photodiode / camera lines
    """,
    plot_patched_line_alternation: """
    - every line should be "ok": a FAIL is a reconstructed span where rising and falling edges don't alternate
    """,
}
