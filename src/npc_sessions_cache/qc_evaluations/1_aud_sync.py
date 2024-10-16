import functools

from npc_sessions_cache.plots.audio import plot_microphone_response, plot_audio_waveforms as _plot_audio_waveforms
plot_audio_target_waveforms = functools.partial(_plot_audio_waveforms, target_stim=True)
plot_audio_nontarget_waveforms = functools.partial(_plot_audio_waveforms, target_stim=False)

instructions = {
    plot_microphone_response: """
    - all values should be similar and far from zero (40 - 60 mV typical)
    - all values below 10 mV indicates an issues with the microphone or sync-alignment
    - a step change in values indicates an issue with the amplifier
    """,
    plot_audio_target_waveforms: """
    """,
    plot_audio_nontarget_waveforms: """
    """
}