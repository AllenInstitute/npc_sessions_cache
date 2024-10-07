import npc_sessions


def plot_unsorted_probes(session: npc_sessions.DynamicRoutingSession) -> str:
    return ''.join(sorted(session.probe_letters_skipped_by_sorting))