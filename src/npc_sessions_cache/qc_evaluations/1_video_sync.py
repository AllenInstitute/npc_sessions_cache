from npc_sessions_cache.plots.video import plot_lick_triggered_frames, plot_pupil_response, plot_pupil_area_with_running
from npc_sessions_cache.plots.sync_patch import plot_camera_intervals_near_invalid_intervals

instructions = {
    plot_lick_triggered_frames: """
    - tongue should be approaching or already touching the lick spout in the frame above the red tick (lick frame)
    - tongue should not be touching the lick spout in the frame preceding the lick frame
    """,
    plot_pupil_response: """
    - look for an increase in pupil area around 1.0 seconds
    - don't fail on this alone - no response may indicate poor eye-tracking data
    """,
    plot_pupil_area_with_running: """
    - look for correlations between the traces, especially at large changes in running speed
    - don't fail on this alone, but if you don't see correlations, check the lick plots again thoroughly
    """,
    plot_camera_intervals_near_invalid_intervals: """
    - only for sessions with invalid intervals: all intervals at 1 period (between dotted lines), no red points
    """,
}
