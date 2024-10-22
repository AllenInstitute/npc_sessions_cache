# aligned blocks - standalone

import logging
import pathlib

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import npc_session
import numpy as np
import numpy.typing as npt
import polars as pl

import npc_sessions_cache.figures.paper2.utils as utils

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["pdf.fonttype"] = 42


def plot(
    unit_id: str,
    stim_names=("vis1", "sound1", "vis2", "sound2"),
    with_instruction_trial_whitespace: bool = False,
    max_psth_spike_rate: float = 60, # Hz
    use_session_obj: bool = False,
    session = None,
    xlim_0 = -1.0, # seconds before stim onset
    xlim_1 = 2.0, # seconds after stim onset
) -> plt.Figure:
    # in case unit_id is an npc_sessions object
    try:
        session_id = npc_session.SessionRecord(
            unit_id.id
        ).id  # in case unit_id is an npc_sessions object
    except (AttributeError, TypeError):
        session_id = npc_session.SessionRecord(unit_id).id
    
    session_obj = session
    if use_session_obj or session_obj is not None:
        if session_obj is not None:
            obj = session_obj
        else:
            import npc_sessions
            obj = npc_sessions.Session(session_id)
        trials = pl.DataFrame(obj.trials[:])
        try:
            units = pl.DataFrame(obj.units[:][['spike_times', 'location', 'structure', 'unit_id']])
        except KeyError:
            units = pl.DataFrame(obj.units[:][['spike_times', 'unit_id']])
        unit = units.filter(pl.col("unit_id") == unit_id)
        unit_spike_times = unit['spike_times'].to_numpy()[0]
        performance: pl.DataFrame = pl.DataFrame(obj.intervals['performance'][:])
    else:
        units_all_sessions = utils.get_component_df("units")
        spike_times_all_sessions = utils.get_component_zarr("spike_times")
        trials_all_sessions = utils.get_component_df("trials")

        trials = trials_all_sessions.filter(pl.col("session_id") == session_id)

        unit = units_all_sessions.filter(pl.col("unit_id") == unit_id)

        #! session id is without idx for spike times
        spike_times_session_id = "_".join(unit_id.split("_")[:2])
        unit_spike_times: npt.NDArray = spike_times_all_sessions[spike_times_session_id][unit_id][:]
        performance_all_sessions = utils.get_component_df("performance")
        performance = performance_all_sessions.filter(pl.col("session_id") == session_id)
    
    pad_start = -xlim_0 + 0.5  # seconds before stim onset 
    if trials.is_empty():
        raise ValueError(f"No trials found for {session_id}")
    if not unit_spike_times.size:
        raise ValueError(f"No spike times found for {unit_id}")
    modality_to_rewarded_stim = {"aud": "sound1", "vis": "vis1"}
    # add spikes to trials:
    spike_times_by_trial = tuple(
        unit_spike_times[slice(start, stop)] if 0 <= start < stop <= len(unit_spike_times) else []
        for start, stop in np.searchsorted(
            unit_spike_times, trials.select(pl.col("start_time") - pad_start, "stop_time")
        )
    )
    if not spike_times_by_trial or not any(np.array(a).any() for a in spike_times_by_trial):
        raise ValueError(f"No spike times found matching trial times {unit} - either no task presented or major timing issue")
    trials = (
        trials
        .with_columns(
            pl.Series(name="spike_times", values=spike_times_by_trial, dtype=pl.List(pl.Float64)), # doesn't handle empty entries well without explicit dtype
        )
        .with_row_index()
        .explode("spike_times")
        .with_columns(
            stim_centered_spike_times=(
                pl.col("spike_times")
                - pl.col("stim_start_time").alias("stim_centered_spike_times")
            )
        )
        .group_by(
            pl.all().exclude("spike_times", "stim_centered_spike_times"),
            maintain_order=True,
        )
        .all()
        .filter(
            pl.col("stim_name").is_in(stim_names),
            #! filter out autoreward trials triggered by 10 misses:
            # (pl.col('is_reward_scheduled').eq(True) & (pl.col('trial_index_in_block') < 5)) | pl.col('is_reward_scheduled').eq(False),
        )
    )

    # create dummy instruction trials for the non-rewarded stimuli for easier
    # alignment of blocks:
    trials_: pl.DataFrame = trials
    if with_instruction_trial_whitespace:
        for block_index in trials_["block_index"].unique():
            context_name = trials_.filter(pl.col("block_index") == block_index)[
                "context_name"
            ][0]
            autorewarded_stim = modality_to_rewarded_stim[context_name]
            for stim_name in stim_names:
                if autorewarded_stim == stim_name:
                    continue
                extra_df = trials.filter( # filter original trials, not modified ones with dummy instruction trials
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
                    stim_centered_spike_times=pl.lit([]),
                )
                trials_ = pl.concat([trials_, extra_df])

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

    line_params = dict(
        color="grey",
        lw=0.3,
    )
    response_window_start_time = 0.1  # np.median(np.diff(trials.select('stim_start_time', 'response_window_start_time')))
    response_window_stop_time = 1  # np.median(np.diff(trials.select('stim_start_time', 'response_window_stop_time')))
    aud_block_color = 'orange'
    add_psth = True
    nominal_rows_per_block = 20
    block_height_on_page = (
        (6 + add_psth) * nominal_rows_per_block / trials_.n_unique("block_index")
    )  # height of each row will be this value / len(block_df)
    fig, axes = plt.subplots(
        1, len(stim_names), figsize=(1.5 * len(stim_names), 6 + add_psth), sharey=True
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
                        color="grey" if is_vis_block else aud_block_color,
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
                        ymin=(
                            ypositions[num_instructed_trials] - halfline
                            if is_rewarded_stim or with_instruction_trial_whitespace
                            else ypositions[0] - halfline
                        ),
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
                            y0 := (
                                (
                                    ypos
                                    if is_rewarded_stim or not with_instruction_trial_whitespace
                                    else ypositions[
                                        min(num_instructed_trials, len(block_df) - 1)
                                    ]
                                )
                            )
                            - halfline
                        ),
                    ),
                    width=response_window_stop_time - response_window_start_time,
                    height=(ypositions[-1] + halfline) - y0,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=[0.85, 0.95, 1, 0.5],
                    zorder=20,
                )
                ax.add_patch(rect)

            # green patch for instruction trials triggered after 10 consecutive misses
            if trial["is_reward_scheduled"] and trial["trial_index_in_block"] > 10:
                ax.axhspan(ypos - halfline, ypos + halfline, **green_patch_params)

            # spikes
            trial_spike_times = np.array(trial["stim_centered_spike_times"])
            eventplot_params = dict(
                lineoffsets=ypos,
                linewidths=0.3,
                linelengths=0.8,
                color=[0.6] * 3,
                zorder=99,
            )
            if trial_spike_times.size == 1 and trial_spike_times[0] is None:
                pass
            else:
                ax.eventplot(positions=trial_spike_times, **eventplot_params)

            # times of interest
            override_params = dict(alpha=1)
            if trial["is_rewarded"]:
                time_of_interest = trial["reward_time"] - trial["stim_start_time"]
                override_params |= dict(marker=".", color="c", edgecolor="none")
                ax.eventplot(
                    positions=[time_of_interest],
                    **eventplot_params | dict(color="c"),
                )
                continue
            elif trial["is_false_alarm"]:
                if trial["response_time"] is None:
                    assert (
                        trial["task_control_response_time"] is not None
                    ), "false alarm without response time"
                    continue
                time_of_interest = trial["response_time"] - trial["stim_start_time"]
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
        last_ypos.append(ypos)
    # format axes and add PSTH
    for ax, stim in zip(axes, stim_names):
        ax: plt.Axes
        if add_psth:
            average_block_psth = True # plot PSTHs for individual blocks, and their average
            bin_size_s = 25 / 1000
            scale_bar_len = max(1, int(max_psth_spike_rate / 4)) # Hz
            ypad = 5
            ymin = max(last_ypos) + ypad
            ymax = ymin + nominal_rows_per_block
            ypos = ymax + 0.5
            
            def add_psth_plot(hist, bin_edges, **plot_kwargs):
                # need to plot upside down, scaled
                norm_spike_rate = (hist / max_psth_spike_rate) * (ymax - ymin)
                ax.plot(
                    bin_edges + np.diff(bin_edges)[0] / 2,
                    ymax - norm_spike_rate,
                    **plot_kwargs,
                )
                
            for context_name, color in zip(("aud", "vis"), ('orange', 'grey')):
                
                if average_block_psth:
                    hist_results = []
                    for _, block_trials in trials.group_by("block_index"):
                        df = (
                            block_trials.filter(
                                pl.col(f"is_{context_name}_context"),
                                pl.col("stim_name") == stim,
                            )
                        )
                        a = df["stim_centered_spike_times"].to_numpy()
                        if not a.size:
                            continue
                        hist, bin_edges = utils.makePSTH_numba(
                            spikes=np.sort(unit_spike_times),
                            startTimes=np.array(df["stim_start_time"] - pad_start),
                            windowDur=pad_start + xlim_1, binSize=bin_size_s,
                        )
                        bin_edges = bin_edges - pad_start
                        # hist, bin_edges = hist_(a)
                        hist_results.append(hist)
                        add_psth_plot(hist, bin_edges, lw=.3, c=color, alpha=.3)
                    if not hist_results:
                        continue
                    add_psth_plot(np.mean(hist_results, axis=0), bin_edges, lw=.75, c=color)
                else:
                    df = (
                        trials.filter(
                            pl.col(f"is_{context_name}_context"),
                            pl.col("stim_name") == stim,
                        )
                    )
                    hist, bin_edges = utils.makePSTH_numba(
                        spikes=np.sort(unit_spike_times),
                        startTimes=np.array(df["stim_start_time"] - pad_start),
                        windowDur=pad_start + xlim_1, binSize=bin_size_s,
                    )
                    bin_edges = bin_edges - pad_start
                    add_psth_plot(hist, bin_edges, lw=.5, c=color)
                    
            # response window cyan patch in PSTH
            rect = patches.Rectangle(
                xy=(response_window_start_time, ymin),
                width=response_window_stop_time - response_window_start_time,
                height=ymax - ymin + .5,
                linewidth=0,
                edgecolor="none",
                facecolor=[0.85, 0.95, 1, 0.5],
                zorder=-1,
            )
            ax.add_patch(rect)
            if ax is axes[0]:
                # add a scale bar
                length = (ymax - ymin) * scale_bar_len / max_psth_spike_rate
                ax.plot(
                    [xlim_0 - .1, xlim_0 - .1],
                    [ymax - length, ymax],
                    c="k",
                    lw=1,
                    clip_on=False,
                )
                ax.text(
                    x=xlim_0 - 0.6,
                    y=ymax - (length / 2),
                    s=f"{scale_bar_len} Hz",
                    fontsize=6,
                    ha="center",
                    va="center",
                    color="k",
                    rotation=0,
                )
                
        # stim onset vertical line
        ax.axvline(x=0, **line_params)

        ax.set_xlim(xlim_0, xlim_1)
        ax.set_ylim(-0.5, max(ypos, *last_ypos) + 0.5)
        ax.set_xticks(sorted(set([min(xlim_0, 0), 0, xlim_1])))
        # ax.set_xticklabels("" if v % 2 else str(v) for v in ax.get_xticks())
        ax.xaxis.set_tick_params(labelsize=6)
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
        ax.set_xlabel("Time after\nstimulus onset (s)")
        ax.invert_yaxis()
        ax.set_aspect(0.1 * (xlim_1 - xlim_0) / 3)
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
    if unit.is_empty():
        location = "not in units df"
    else:
        if "location" in unit.columns:
            location = unit["location"][0]
        else:
            location = "no CCF location (unannotated)"
    fig.suptitle(
        f"{'behavior pass' if is_pass else 'behavior fail'}\n{unit_id}\n{location}"
    )
    fig.set_dpi(300)
    return fig

def get_unit_ids_shailaja_pkl() -> tuple[str, ...]:
    import pickle
    f = pathlib.Path("c:/users/ben.hardcastle/downloads/list_of_context_units.pkl")
    p = pickle.loads(f.read_bytes())
    unit_ids = []
    for k, v in p.items():
        unit_ids.extend(v)
    return tuple(sorted(unit_ids))

def get_unit_ids_shawn_session_list() -> pl.Series:
    sessions = [f"{i}_0" for i in ('666986_2023-08-16', '714748_2024-06-24','664851_2023-11-16','666986_2023-08-16',
                    '667252_2023-09-28','674562_2023-10-03','681532_2023-10-18',
                    '708016_2024-04-29','714753_2024-07-02','644866_2023-02-10',
'674562_2023-10-05','712141_2024-06-06')]
    top_k = 20
    units = (
        pl.read_csv("c:/users/ben.hardcastle/downloads/context_encoding_units.csv")
        .group_by('session_id')
        .agg(
            pl.col('unit_id').top_k_by(by=pl.col('context drop score'), k=top_k)
        )
        .explode('unit_id')
    ).get_column('unit_id')
    if len(units) < top_k * len(sessions):
        print(f"Expected {top_k * len(sessions)} units, got {len(units)}")
    return units

def get_specific_unit_ids() -> list[str]:
    return [
        '666986_2023-08-16_F-213',
        # '667252_2023-09-26_C-233',
    ]

def get_grouped_baseline_psth_parquet():
    u = (
        pl.read_parquet(r"C:\Users\ben.hardcastle\github\npc_sessions_cache\src\npc_sessions_cache\figures\paper2\baseline_psth.parquet")
        .join(utils.get_component_df("units"), on='unit_id', )
    )
    k = 10
    dfs = []
    for stim in ("vis", "aud"):
        for target in ("target",):
            col = pl.col(f"{stim}_{target}_context_selectivity_index").abs()
            dfs.append(
                u
                .group_by(
                    'structure'
                )
                .agg(
                    [
                        col.top_k(k).alias('abs_context_index'),
                        pl.col('unit_id').top_k_by(col, k),
                    ]
                )
                .explode(pl.all().exclude('structure'))
            )
                
    return pl.concat(dfs).sort('abs_context_index', descending=True)

def get_unit_ids_baseline_psth_parquet():
    return get_grouped_baseline_psth_parquet()['unit_id']

if __name__ == "__main__":

    all_stim_names = ("sound1", "vis1", "sound2", "vis2")
    target_stim_names = ("sound1", "vis1")
    pyfile_path = pathlib.Path(__file__)
    
    raise_on_error = True
    skip_existing = False
    print(f"{skip_existing=}, {raise_on_error=}")
    get_unit_id_func = get_specific_unit_ids
    for unit_id in get_unit_id_func():
        
        figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}_{unit_id}")
        if get_unit_id_func in (
            get_unit_ids_shawn_session_list,
            get_unit_ids_baseline_psth_parquet
        ):
            materials_path = pathlib.Path("C:/Users/ben.hardcastle/OneDrive - Allen Institute/Shared Documents - Dynamic Routing/DR Manuscripts/DR Paper 2 - context representations/Figures/Figure 3/materials")
            if get_unit_id_func is get_unit_ids_shawn_session_list:
                requested_path = materials_path / "top_context_units_by_session_v0.0.235"
            elif get_unit_id_func is get_unit_ids_baseline_psth_parquet:
                area = get_grouped_baseline_psth_parquet().filter(pl.col('unit_id') == unit_id)['structure'][0]
                requested_path = materials_path / "top_context_units_by_area_v0.0.235" / area
            requested_path.mkdir(exist_ok=True, parents=True)
            figsave_path = requested_path / figsave_path.name
        if skip_existing and figsave_path.with_suffix(".png").exists():
            print(f"skipping {pyfile_path.stem} for {unit_id} - already exists")
            continue        
        
        print(f"plotting {pyfile_path.stem} for {unit_id}")
        try:
            fig = plot(
                unit_id=unit_id,
                stim_names=all_stim_names,
                with_instruction_trial_whitespace=False,
                use_session_obj=False,
            )
        except Exception as exc:
            if raise_on_error:
                raise
            print(f"failed: {exc!r}")
            continue
        fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        

# %%
