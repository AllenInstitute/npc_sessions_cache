from __future__ import annotations

import npc_sessions
import upath
import npc_lims

def plot_lightning_pose_video(
    session: npc_sessions.DynamicRoutingSession,
) -> upath.UPath | None:
    path = npc_lims.S3_SCRATCH_ROOT / "LP_videos" / f"{session.id}_qc_LP_video.mp4"
    if not path.exists():
        return None
    return path