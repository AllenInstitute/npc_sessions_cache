import dataclasses
from typing import Any

import npc_sessions

import npc_sessions_cache.utils.session_records as session_records


def plot_session_metadata(session: npc_sessions.DynamicRoutingSession) -> dict[str, Any]:
    return dataclasses.asdict(session_records.get_session_record(session.session_id))
