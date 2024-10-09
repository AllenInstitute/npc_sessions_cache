import npc_sessions


def plot_unsorted_probes(session: npc_sessions.DynamicRoutingSession) -> str | None:
    if not session.is_ephys:
        return None
    return ''.join(sorted(session.probe_letters_skipped_by_sorting))

def plot_unsorted_surface_recording_probes(session: npc_sessions.DynamicRoutingSession) -> str | None:
    if not session.is_surface_channels:
        return None
    return ''.join(sorted(session.surface_recording.probe_letters_skipped_by_sorting))