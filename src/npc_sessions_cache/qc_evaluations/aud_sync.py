from npc_sessions_cache.plots.audio import plot_microphone_response

instructions = {
    plot_microphone_response: """
    - all values should be similar and far from zero (40 - 60 mV typical)
    - all values below 10 mV indicates an issues with the microphone or sync-alignment
    - a step change in values indicates an issue with the amplifier
    """
}