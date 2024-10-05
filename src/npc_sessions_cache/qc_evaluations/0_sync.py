from npc_sessions_cache.plots.sync import plot_barcode_intervals

instructions = {
    plot_barcode_intervals: """
    - all points in the left panel should be very close to 30 seconds (deviation < 0.1s)
    """,
}
