# aligned blocks - standalone

import pathlib

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import npc_session
import numpy as np
import numpy.typing as npt
import polars as pl

import npc_sessions_cache.figures.paper2.utils as utils

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["pdf.fonttype"] = 42


def plot(
    session_id: str,
    stim_names=("vis1", "sound1", "vis2", "sound2"),
    use_session_obj: bool = False,
    session = None,
) -> plt.Figure:
    try:
        session_id = npc_session.SessionRecord(
            session_id.id
        ).id  # in case session_id is an npc_sessions object
    except (AttributeError, TypeError):
        session_id = npc_session.SessionRecord(session_id).id
        
    if use_session_obj or session is not None:
        if session is not None:
            obj = session
        else:
            import npc_sessions
            obj = npc_sessions.Session(session_id)
        trials = pl.DataFrame(obj.trials[:]).drop('index', strict=False)
        performance: pl.DataFrame = pl.DataFrame(obj.intervals['performance'][:])
        lick_times: npt.NDArray = obj._all_licks[0].timestamps
    
    else:
        licks_all_sessions = utils.get_component_zarr("licks")
        trials_all_sessions = utils.get_component_df("trials")
        performance_all_sessions = utils.get_component_df("performance")

        performance = performance_all_sessions.filter(pl.col("session_id") == session_id)
        trials = trials_all_sessions.filter(pl.col("session_id") == session_id)
        lick_times: npt.NDArray = licks_all_sessions[session_id]["timestamps"][:]

    modality_to_rewarded_stim = {"aud": "sound1", "vis": "vis1"}

    # add licks to trials:
    pad_start = 1.5  # seconds
    lick_times_by_trial = tuple(
        lick_times[slice(start, stop)]
        if 0 <= start < stop < len(lick_times)
        else []
        for start, stop in np.searchsorted(
            lick_times, trials.select(pl.col("start_time") - pad_start, "stop_time")
        )
    )
    trials_ = (
        trials.lazy()
        .with_columns(
            pl.Series(name="lick_times", values=lick_times_by_trial),
        )
        .with_row_index()
        .explode("lick_times")
        .with_columns(
            stim_centered_lick_times=(
                pl.col("lick_times")
                - pl.col("stim_start_time").alias("stim_centered_lick_times")
            )
        )
        .group_by(
            pl.all().exclude("lick_times", "stim_centered_lick_times"),
            maintain_order=True,
        )
        .all()
        .drop('lick_times')
    )

    # select VIStarget / AUDtarget trials
    trials_: pl.LazyFrame = trials_.filter(
        #! filter out autoreward trials triggered by 10 misses:
        # (pl.col('is_reward_scheduled').eq(True) & (pl.col('trial_index_in_block') < 5)) | pl.col('is_reward_scheduled').eq(False),
        pl.col("stim_name").is_in(stim_names),
    )

    # create dummy instruction trials for the non-rewarded stimuli for easier
    # alignment of blocks:
    trials_: pl.DataFrame = trials_.collect()
    for block_index in trials_["block_index"].unique():
        context_name = trials_.filter(pl.col("block_index") == block_index)[
            "context_name"
        ][0]
        autorewarded_stim = modality_to_rewarded_stim[context_name]
        for stim_name in stim_names:
            if autorewarded_stim == stim_name:
                continue
            extra_df = trials.filter(
                # filter original trials, not modified ones with dummy instruction trials
                # need to make sure same set of columns for both though                      
                pl.col("block_index") == block_index,
                pl.col("is_reward_scheduled"),
                pl.col("trial_index_in_block")
                <= 5,  # after 10 misses, an instruction trial is triggered: we don't want to duplicate these
            ).with_columns(
                # switch the stim name:
                stim_name=pl.lit(stim_name),
                # make sure there's no info that will trigger plotting:
                is_response=pl.lit(False),
                is_rewarded=pl.lit(False),
                stim_centered_lick_times=pl.lit([]),
            )
            trials_ = trials_.drop('index', strict=False)
            extra_df = extra_df.drop('index', strict=False)
            assert not (diff := set(trials_.columns) ^ set(extra_df.columns)), f"difference in columns: {diff}"
            trials_ = pl.concat([trials_, extra_df], how="vertical_relaxed")

    # add columns for easier parsing of block structure:
    trials_ = trials_.sort("start_time").with_columns(
        is_new_block=(
            pl.col("start_time")
            == pl.col("start_time").min().over("stim_name", "block_index")
        ),
        num_trials_in_block=pl.col("start_time")
        .count()
        .over("stim_name", "block_index"),
    )


    scatter_params = dict(
        marker="|",
        s=20,
        color=[0.85] * 3,
        alpha=1,
        edgecolor="none",
    )
    line_params = dict(
        color="grey",
        lw=0.3,
    )
    response_window_start_time = 0.1  # np.median(np.diff(trials.select('stim_start_time', 'response_window_start_time')))
    response_window_stop_time = 1  # np.median(np.diff(trials.select('stim_start_time', 'response_window_stop_time')))
    xlim_0 = -1
    block_height_on_page = 120 / trials_.n_unique(
        "block_index"
    )  # height of each row will be this value / len(block_df)
    fig, axes = plt.subplots(
        1, len(stim_names), figsize=(1.5 * len(stim_names), 6), sharey=True
    )
    last_ypos: list[float] = []
    for ax, stim in zip(axes, stim_names):
        ax: plt.Axes

        stim_trials = trials_.filter(pl.col("stim_name") == stim)
        idx_in_block = 0
        for idx, trial in enumerate(stim_trials.iter_rows(named=True)):

            num_instructed_trials = max(
                len(
                    trials.filter(  # check original trials, not modified ones with dummy instruction trials
                        pl.col("block_index") == trial["block_index"],
                        pl.col(f"is_{c}_context"),
                        pl.col("is_reward_scheduled"),
                        pl.col("trial_index_in_block") < 14,
                    )
                )
                for c in ("aud", "vis")
            )

            is_vis_block: bool = "vis" in trial["context_name"]
            is_vis_target: bool = "vis1" in trial["stim_name"]
            is_aud_target: bool = "sound1" in trial["stim_name"]
            is_rewarded_stim: bool = (is_vis_target and is_vis_block) or (
                is_aud_target and not is_vis_block
            )

            if trial["is_new_block"]:
                idx_in_block = 0
                block_df = stim_trials.filter(
                    pl.col("block_index") == trial["block_index"]
                )
                ypositions = (
                    np.linspace(0, block_height_on_page, len(block_df), endpoint=False)
                    + trial["block_index"] * block_height_on_page
                )
                halfline = 0.5 * np.diff(ypositions).mean()
            ypos = ypositions[idx_in_block]

            idx_in_block += 1  # updated for next trial - don't use after this point

            if trial["is_new_block"]:
                if is_rewarded_stim:
                    assert num_instructed_trials == (
                        x := len(
                            block_df.filter(
                                (pl.col("trial_index_in_block") < 10)
                                & (pl.col("is_reward_scheduled"))
                            )
                        )
                    ), f"{x} != {num_instructed_trials=}"

                if ax is axes[0]:
                    # block label
                    rotation = 0
                    ax.text(
                        x=xlim_0 - 0.6,
                        y=ypositions[0] + block_height_on_page // 2,
                        s=str(trial["block_index"] + 1),
                        fontsize=8,
                        ha="center",
                        va="center",
                        color="k",
                        rotation=rotation,
                    )

                # block switch horizontal lines
                if trial["block_index"] > 0:
                    ax.axhline(
                        y=ypos - halfline,
                        **line_params,
                        zorder=99,
                    )

                if is_rewarded_stim:
                    # autoreward trials green patch
                    green_patch_params = dict(color=[0.9, 0.95, 0.9], lw=0, zorder=-1)
                    ax.axhspan(
                        ymin=max(ypos, 0) - halfline,
                        ymax=ypositions[num_instructed_trials - 1] + halfline,
                        **green_patch_params,
                    )

                if trial["is_vis_context"] and len(block_df) > num_instructed_trials:
                    # vis block grey patch
                    ax.axhspan(
                        ymin=ypositions[num_instructed_trials] - halfline,
                        ymax=ypositions[-1] + halfline,
                        color=[0.95] * 3,
                        lw=0,
                        zorder=-1,
                    )

                # response window cyan patch
                rect = patches.Rectangle(
                    xy=(
                        response_window_start_time,
                        (
                            y := max(
                                0,
                                (
                                    ypos
                                    if is_rewarded_stim
                                    else ypositions[
                                        min(num_instructed_trials, len(block_df) - 1)
                                    ]
                                ),
                            )
                            - halfline
                        ),
                    ),
                    width=response_window_stop_time - response_window_start_time,
                    height=(ypositions[-1] + halfline) - y,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=[0.85, 0.95, 1, 0.5],
                    zorder=20,
                )
                ax.add_patch(rect)

            # green patch for instruction trials triggered after 10 consecutive misses
            if trial["is_reward_scheduled"] and trial["trial_index_in_block"] > 10:
                ax.axhspan(ypos - halfline, ypos + halfline, **green_patch_params)

            # licks
            lick_times = np.array(trial["stim_centered_lick_times"])
            eventplot_params = dict(
                lineoffsets=ypos,
                linewidths=0.3,
                linelengths=0.8,
                color=[0.4] * 3,
                zorder=99,
            )
            if lick_times.size == 1 and lick_times[0] is None:
                continue
            ax.eventplot(positions=lick_times, **eventplot_params)

            # times of interest
            override_params = dict(alpha=1)
            if trial["is_rewarded"]:
                time_of_interest = trial["reward_time"] - trial["stim_start_time"]
                override_params |= dict(marker=".", color="c", edgecolor="none")
            elif trial["is_false_alarm"]:
                time_of_interest = lick_times[lick_times > 0][0]
                false_alarm_line = True  # set False to draw a dot instead of a line
                if false_alarm_line:
                    ax.eventplot(
                        positions=[time_of_interest],
                        **eventplot_params | dict(color="r"),
                    )
                    continue
                else:
                    override_params |= dict(marker=".", color="r", edgecolor="none")
            else:
                continue
            ax.scatter(
                time_of_interest,
                ypos,
                **scatter_params | override_params,
                zorder=99,
                clip_on=False,
            )
        last_ypos.append(ypos)
        
        # stim onset vertical line
        ax.axvline(x=0, **line_params)

        ax.set_xlim(xlim_0, 2.0)
        ax.set_ylim(-0.5, max(ypos, *last_ypos) + 0.5)
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_xticklabels("" if v % 2 else str(v) for v in ax.get_xticks())
        ax.set_yticks([])
        if ax is axes[0]:
            ax.set_ylabel("← Trials")
            ax.yaxis.set_label_coords(x=-0.3, y=0.5)
            ax.text(
                x=xlim_0 - 0.6,
                y=-0,
                s="Block #",
                fontsize=8,
                ha="center",
                va="center",
                color="k",
                rotation=0,
            )
        ax.set_xlabel("Time after\nstimulus onset(s)")
        ax.invert_yaxis()
        ax.set_aspect(0.1)
        stim_to_label = {
            "vis1": "VIS+",
            "vis2": "VIS-",
            "sound1": "AUD+",
            "sound2": "AUD-",
        }
        ax.set_title(stim_to_label[stim], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_zorder(199)
         
    is_pass = (
        len(
            pl.DataFrame(performance).filter(
                pl.col("same_modal_dprime") > 1.0,
                pl.col("cross_modal_dprime") > 1.0,
            )
        )
        > 3
    )
    fig.suptitle(
        f"{'pass' if is_pass else 'fail'}\n{session_id}"
    )  #! update to session.id
    return fig


if __name__ == "__main__":

    a = "715710_2024-07-17_0"  # VIS first, many misses in last block
    b = "681532_2023-10-18_0"  # VIS first, attractor-like clusterings of FAs
    c = "714753_2024-07-02_0"  # AUD first, low FA rate, some blocks apparently don't need instruction trial
    # for session_id in ['714748_2024-06-24','664851_2023-11-16','666986_2023-08-16',
    #                     '667252_2023-09-28','674562_2023-10-03','681532_2023-10-18',
    #                     '708016_2024-04-29','714753_2024-07-02','644866_2023-02-10']:
    # session_id = '620263_2022-07-26' #< session with 10 autorewards

    stim_names = ("vis1", "vis2", "sound1", "sound2")
    for session_id, stim_names in zip(
        (b, c), (stim_names, ("sound1", "vis1"))
    ):
        pyfile_path = pathlib.Path(__file__)
        print(f"plotting {pyfile_path.stem} for {session_id}")
        fig = plot(session_id, stim_names)

        figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}_{session_id}")
        fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches="tight")

        # make sure text is editable in illustrator before saving pdf:
        fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches="tight")
