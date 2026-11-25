from npc_sessions_cache.plots.video import plot_camera_frame_counts, plot_lick_triggered_frames, plot_pupil_response, plot_pupil_area_with_running, plot_lick_bout_triggered_facemap_side
from npc_sessions_cache.plots.sync_patch import plot_camera_intervals_near_invalid_intervals

instructions = {
    plot_camera_frame_counts: """
    - cameras with `excluded_reason` are left out of the NWB (frame times can't be trusted)
    - `num_unaccounted_frames` > 0: exposures on sync that are neither in the video nor in `LostFrames`;
      frame times after them are early by up to that many frames
    - `num_exposures_inserted` > 0: exposures in unknown sync intervals (nan frame times there)
    """,
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
    plot_lick_bout_triggered_facemap_side: """
    - expect a flat pre-lick baseline, with face motion starting ~0.1-0.3 s before the first lick and
      rhythmic (~7-8 Hz) structure after it
    - motion that is offset in time from the lick (or no lick-locked motion) suggests a video-sync problem
    - title is red if side-camera frame loss is > 0.1% or the side camera stopped early: check the
      other video plots for this session
    """,
    plot_camera_intervals_near_invalid_intervals: """
    - only for sessions with invalid intervals: all intervals at 1 period (between dotted lines), no red points
    """,
}
