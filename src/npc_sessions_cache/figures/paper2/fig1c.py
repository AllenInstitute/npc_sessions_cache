# aligned blocks - standalone

import pathlib

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import npc_session
import numpy as np
import numpy.typing as npt
import polars as pl
import unit_utils


def plot(session_id: str) -> plt.Figure:
    try:
        session_id = npc_session.SessionRecord(session_id.id).id # in case session_id is an npc_sessions object
    except (AttributeError, TypeError):
        session_id = npc_session.SessionRecord(session_id).id

    licks_all_sessions = unit_utils.get_component_zarr('licks')
    trials_all_sessions = unit_utils.get_component_df('trials')
    all_sessions = unit_utils.get_component_df('session')
    performance_all_sessions = unit_utils.get_component_df('performance')

    performance = performance_all_sessions.filter(pl.col('session_id') == session_id)
    trials = trials_all_sessions.filter(pl.col('session_id') == session_id)
    lick_times: npt.NDArray = licks_all_sessions[session_id]['timestamps'][:]

    # add licks to trials:
    pad_start = 1 # seconds
    lick_times_by_trial = tuple(lick_times[slice(start_stop[0] - pad_start, start_stop[1])] for start_stop in np.searchsorted(lick_times, trials.select('start_time', 'stop_time')))
    trials_ = (
        trials
        .lazy()
        .with_columns(
            pl.Series(name="lick_times", values=lick_times_by_trial),
        )
        .with_row_index()
        .explode('lick_times')
        .with_columns(
            stim_centered_lick_times=(pl.col('lick_times') - pl.col('stim_start_time').alias('stim_centered_lick_times'))
        )
        .group_by(pl.all().exclude("lick_times", "stim_centered_lick_times"), maintain_order=True)
        .all()
    )

    # select VIStarget / AUDtarget trials
    trials_ = (
        trials_
        .filter(
            #! filter out autoreward trials triggered by 10 misses:
            # (pl.col('is_reward_scheduled').eq(True) & (pl.col('trial_index_in_block') < 5)) | pl.col('is_reward_scheduled').eq(False),
            pl.col('is_target'),
        )
    )

    # create dummy instruction trials for the non-rewarded stimulus for easier
    # alignment of blocks:
    trials_ = trials_.collect()
    for block_index in trials_['block_index'].unique():
        extra_df = (
            trials_
            .filter(
                pl.col('block_index') == block_index,
                pl.col('is_reward_scheduled'),
                pl.col('trial_index_in_block') < 10, # after 10 misses, an instruction trial is triggered: we don't want to duplicate these
            )
            .with_columns(
                # switch the stim name:
                stim_name=(
                    pl.when(pl.col('is_vis_context')).then(pl.lit('sound1')).otherwise(pl.lit('vis1'))
                ),
                # make sure there's no info that will trigger plotting:
                is_response=pl.lit(False),
                is_rewarded=pl.lit(False),
                stim_centered_lick_times=pl.lit([]),
            )
        )
        trials_ = pl.concat([trials_, extra_df])

    # add columns for easier parsing of block structure:
    trials_ = (
        trials_
        .sort('start_time')
        .with_columns(
            is_new_block=(pl.col('start_time') == pl.col('start_time').min().over('stim_name', 'block_index')),
            num_in_block=pl.col('is_reward_scheduled').sum().over('stim_name', 'block_index'),
            num_trials_in_block=pl.col('start_time').count().over('stim_name', 'block_index'),
        )
    )

    is_pass = len(
        pl.DataFrame(performance)
        .filter(
            pl.col('same_modal_dprime') > 1.0,
            pl.col('cross_modal_dprime') > 1.0,
        )
    ) > 3

    is_first_block_aud = trials_.filter(pl.col('block_index') == 0 )['is_aud_context'][0]

    scatter_params = dict(
        marker='|',
        s=20,
        color=[0.85] * 3,
        alpha=1,
        edgecolor='none',

    )
    line_params = dict(
        color='grey',
        lw=.3,
    )
    response_window_start_time = 0.1 # np.median(np.diff(trials.select('stim_start_time', 'response_window_start_time')))
    response_window_stop_time = 1 # np.median(np.diff(trials.select('stim_start_time', 'response_window_stop_time')))
    xlim_0 = -1
    block_size = 20
    fig, axes = plt.subplots(1,2, figsize=(3, 6), sharey=True)
    for ax, stim in zip(axes, ("sound1", "vis1")):
        ax: plt.Axes

        stim_trials = (
            trials_
            .filter(
                pl.col('stim_name') == stim,
                pl.col('is_target')
            )
        )

        idx_in_block = 0
        for idx, trial in enumerate(stim_trials.iter_rows(named=True)):

            num_instructed_trials = max(len(
                trials # check original trials, not modified ones with dummy instruction trials
                .filter(
                    pl.col('block_index') == trial['block_index'],
                    pl.col(f'is_{c}_context'),
                    pl.col('is_reward_scheduled'),
                    pl.col('trial_index_in_block') < 14,
                )
            ) for c in ("aud", "vis"))

            is_vis_block: bool = "vis" in trial["context_name"]
            is_vis_stim: bool = "vis" in trial["stim_name"]
            is_rewarded_stim: bool = is_vis_stim == is_vis_block


            if trial['is_new_block']:
                idx_in_block = 0
                block_df = stim_trials.filter(pl.col('block_index') == trial['block_index'])
                ypositions = np.linspace(0, block_size, len(block_df), endpoint=False) + trial['block_index'] * block_size
                halfline = 0.5 * np.diff(ypositions).mean()
            ypos = ypositions[idx_in_block]

            idx_in_block += 1 # updated for next trial - don't use after this point

            if trial['is_new_block']:
                if is_rewarded_stim:
                    assert num_instructed_trials == (x := len(block_df.filter((pl.col('trial_index_in_block') < 10) & (pl.col('is_reward_scheduled'))))), f"{x} != {num_instructed_trials=}"

                if ax is axes[0]:
                    # block label
                    rotation = 0
                    ax.text(
                        x=xlim_0 - 0.4,
                        y=ypositions[0] + block_size // 2,
                        s=str(trial["block_index"] + 1),
                        fontsize=8, ha='center', va='center', color='k', rotation=rotation,
                    )

                # block switch horizontal lines
                if trial['block_index'] > 0:
                    ax.axhline(
                        y=ypos - halfline,
                        **line_params, zorder=99,
                        )

                if is_vis_block == is_vis_stim:
                    # autoreward trials green patch
                    green_patch_params = dict(color=[.9, .95, .9], lw=0, zorder=-1)
                    ax.axhspan(
                        ymin=max(ypos, 0) - halfline,
                        ymax=ypositions[num_instructed_trials - 1] + halfline,
                        **green_patch_params,
                        )

                if trial['is_vis_context']:
                    # vis block grey patch
                    ax.axhspan(
                        ymin=ypositions[num_instructed_trials] - halfline,
                        ymax=ypositions[-1] + halfline,
                        color=[.95]*3, lw=0, zorder=-1,
                        )

                # response window cyan patch
                rect = patches.Rectangle(
                    xy=(0.1, (y := max(0, (ypos if is_rewarded_stim else ypositions[num_instructed_trials])) - halfline)),
                    width=.9,
                    height=(ypositions[-1] + halfline) - y,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=[0.85, 0.95, 1, 0.5],
                    zorder=20,
                )
                ax.add_patch(rect)

            # green patch for instruction trials triggered after 10 consecutive misses
            if trial['is_reward_scheduled'] and trial['trial_index_in_block'] > 10:
                ax.axhspan(ypos - halfline, ypos + halfline, **green_patch_params)

            # licks
            lick_times = np.array(trial['stim_centered_lick_times'])
            eventplot_params = dict(lineoffsets=ypos, linewidths=.3, linelengths=.8,
                                    color=[.6]*3, zorder=99)
            ax.eventplot(positions=lick_times, **eventplot_params)

            # times of interest
            override_params = dict(alpha=1)
            if trial['is_rewarded']:
                time_of_interest  = trial['reward_time'] - trial['stim_start_time']
                override_params |= dict(marker='.', color='c', edgecolor='none')
            elif trial['is_false_alarm']:
                time_of_interest  = lick_times[lick_times > 0][0]
                false_alarm_line = True # set False to draw a dot instead of a line
                if false_alarm_line:
                    ax.eventplot(positions=[time_of_interest], **eventplot_params | dict(color='r'))
                    continue
                else:
                    override_params |= dict(marker='.', color='r', edgecolor='none')
            else:
                continue
            ax.scatter(time_of_interest, ypos, **scatter_params | override_params, zorder=99)

        # stim onset vertical line
        ax.axvline(x=0, **line_params)

        ax.set_xlim(xlim_0, 2.0)
        ax.set_ylim(-0.5, ypos + 0.5)
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_xticklabels("" if v%2 else str(v) for v in ax.get_xticks())
        ax.set_yticks([])
        if ax is axes[0]:
            ax.set_ylabel("← trials")
            ax.yaxis.set_label_coords(x=-0.3, y=0.5)
            ax.text(x=xlim_0 - 0.4, y=-0, s="block", fontsize=8, ha='center', va='center', color='k', rotation=0)
        ax.set_xlabel("time rel. to\nstim onset (s)")
        ax.invert_yaxis()
        ax.set_aspect(0.1)
        ax.set_title("VIS+" if is_vis_stim else "AUD+", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_zorder(99)

    fig.suptitle(f"{'pass' if is_pass else 'fail'}\n{session_id}") #! update to session.id
    return fig

if __name__ == "__main__":

    session_id = '681532_2023-10-18_0' # VIS first, attractor-like clusterings of FAs
    session_id = '715710_2024-07-17_0' # VIS first, many misses in last block
    session_id = '714753_2024-07-02_0' # AUD first, low FA rate, some blocks apparently don't need instruction trial
    # for session_id in ['714748_2024-06-24','664851_2023-11-16','666986_2023-08-16',
    #                     '667252_2023-09-28','674562_2023-10-03','681532_2023-10-18',
    #                     '708016_2024-04-29','714753_2024-07-02','644866_2023-02-10']:
    # session_id = '620263_2022-07-26' #< session with 10 autorewards

    fig = plot(session_id)

    pyfile_path = pathlib.Path(__file__)
    figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}_{session_id}")
    fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches='tight')

    # make sure text is editable in illustrator before saving pdf:
    plt.rcParams['pdf.fonttype'] = 42
    fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches='tight')
