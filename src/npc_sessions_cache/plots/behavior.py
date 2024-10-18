from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import npc_sessions
import numpy as np
from matplotlib.patches import Rectangle
import polars as pl

if TYPE_CHECKING:
    import pandas as pd

import npc_sessions_cache.plots.plot_utils as plot_utils


def plot_performance_by_block(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    task_performance_by_block_df: pd.DataFrame = session.performance[:]

    dprime_threshold = 1.5 if session.is_training else 1.0
    n_passing_blocks = np.sum(task_performance_by_block_df["cross_modal_dprime"] >= dprime_threshold)
    failed_block_ind = task_performance_by_block_df["cross_modal_dprime"] < dprime_threshold

    # blockwise behavioral performance
    xvect = task_performance_by_block_df.index.values
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(
        xvect,
        task_performance_by_block_df["signed_cross_modal_dprime"],
        "ko-",
        label="cross-modal",
    )
    ax[0].plot(
        xvect[failed_block_ind],
        task_performance_by_block_df["signed_cross_modal_dprime"][failed_block_ind],
        "ro",
        label="failed",
    )
    ax[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    max_dprime = np.nanmax(np.abs(task_performance_by_block_df["signed_cross_modal_dprime"]))
    ax[0].set_ylim([-2, 2] if max_dprime < 2 else [-max_dprime - 0.2, max_dprime + 0.2])
    ax[0].set_title(
        "cross-modal dprime: "
        + str(n_passing_blocks)
        + "/"
        + str(len(task_performance_by_block_df))
        + " blocks passed"
        + f" (threshold={dprime_threshold})"
    )
    ax[0].set_ylabel("aud <- dprime -> vis")

    ax[1].plot(
        xvect, task_performance_by_block_df["vis_intra_dprime"], "go-", label="vis"
    )
    ax[1].plot(
        xvect, task_performance_by_block_df["aud_intra_dprime"], "bo-", label="aud"
    )
    max_dprime =np.nanmax(
        np.abs(
            np.concatenate(
                [
                    task_performance_by_block_df["vis_intra_dprime"],
                    task_performance_by_block_df["aud_intra_dprime"],
                ]
            )
        )
    )
    ax[1].set_ylim([0, 2 if max_dprime < 2 else max_dprime + 0.2])
    ax[1].set_title("intra-modal dprime")
    ax[1].legend(["vis", "aud"])
    ax[1].set_xlabel("block index")
    ax[1].set_ylabel("dprime")

    fig.suptitle(session.id)
    fig.tight_layout()

    return fig


def plot_first_lick_latency_hist(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    # first lick latency histogram

    trials: pd.DataFrame = session.trials[:]

    xbins = np.arange(0, 1, 0.05)
    fig, ax = plt.subplots(1, 1)
    ax.hist(
        trials.query("is_vis_stim==True")["response_time"]
        - trials.query("is_vis_stim==True")["stim_start_time"],
        bins=xbins,
        alpha=0.5,
    )

    ax.hist(
        trials.query("is_aud_stim==True")["response_time"]
        - trials.query("is_aud_stim==True")["stim_start_time"],
        bins=xbins,
        alpha=0.5,
    )

    ax.legend(["vis stim", "aud stim"])
    ax.set_xlabel("lick latency (s)")
    ax.set_ylabel("trial count")
    ax.set_title("lick latency: " + session.id)

    return fig


def plot_lick_raster(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    from npc_sessions_cache.figures.paper2.fig1c import plot
    return plot(session_id=session.id, session=session)


def plot_running(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:
    timeseries = session.processing["behavior"]["running_speed"]
    epochs: pd.DataFrame = session.epochs[:]
    licks = session.processing["behavior"]["licks"]
    plt.style.use("seaborn-v0_8-notebook")

    fig, ax = plt.subplots()

    for _, epoch in epochs.iterrows():
        epoch_indices = (timeseries.timestamps >= epoch["start_time"]) & (
            timeseries.timestamps <= epoch["stop_time"]
        )
        if len(epoch_indices) > 0:
            ax.plot(
                timeseries.timestamps[epoch_indices],
                timeseries.data[epoch_indices],
                linewidth=0.1,
                alpha=1,
                color="k",
                label="speed",
                zorder=30,
            )
    k = 100 if "cm" in timeseries.unit else 1
    ymax = 0.8 * k
    ax.set_ylim([-0.05 * k, ymax])
    ax.vlines(
        licks.timestamps,
        *ax.get_ylim(),
        color="lime",
        linestyle="-",
        linewidth=0.05,
        zorder=10,
    )
    ax.hlines(
        0,
        0,
        max(timeseries.timestamps),
        color="k",
        linestyle="--",
        linewidth=0.5,
        zorder=20,
    )
    plot_utils.add_epoch_color_bars(ax, epochs, rotation=90, y=ymax, va="top")
    ax.margins(0)
    ax.set_frame_on(False)
    ax.set_ylabel(timeseries.unit)
    ax.set_xlabel(timeseries.timestamps_unit)
    title = timeseries.description
    if max(timeseries.data) > ax.get_ylim()[1]:
        title += f"\ndata clipped: {round(max(timeseries.data)) = } {timeseries.unit} at {timeseries.timestamps[np.argmax(timeseries.data)]:.0f} {timeseries.timestamps_unit}"
    ax.set_title(title, fontsize=8)
    fig.suptitle(session.id, fontsize=10)
    fig.set_size_inches(10, 4)
    fig.set_layout_engine("tight")
    return fig


def plot_response_rate_by_stimulus_type(
    session: npc_sessions.DynamicRoutingSession,
) -> matplotlib.figure.Figure:

    trials = session.trials[:]
    start_time = trials.iloc[0]['start_time']
    end_time = trials.iloc[-1]['stop_time']

    switch_times = trials[trials['is_context_switch']]['start_time']
    switch_starts = np.insert(switch_times, 0, start_time)
    switch_ends = np.append(switch_times, end_time)
    switch_durations = switch_ends - switch_starts

    window_size = 120 #seconds

    time = np.arange(start_time,end_time,window_size)

    stim_types = ['vis_target', 'aud_target', 'vis_nontarget', 'aud_nontarget']
    rate_dict = {stim_type:[] for stim_type in stim_types}
    for stim_type in stim_types:

        response_times = trials[trials[f'is_{stim_type}'] & trials['is_response']]['stim_start_time'].values
        trial_type_times = trials[trials[f'is_{stim_type}']]['stim_start_time'].values

        rt_hist, _ = np.histogram(response_times, bins = time)
        tt_hist, _ = np.histogram(trial_type_times, bins = time)

        rate_dict[stim_type] = rt_hist/tt_hist

    aud_block_inds = np.arange(0, len(switch_starts), 2) + trials.iloc[0]['is_vis_context']

    fig, ax = plt.subplots()
    for stim_type in stim_types:
        ax.plot(time[:-1], rate_dict[stim_type])

    ax.set_ylabel('Response Rate')
    ax.set_xlabel('Session Time (s)')

    for aud_block in aud_block_inds:
        rectangle = Rectangle((switch_starts[aud_block], 0), switch_durations[aud_block], 1, color='k', alpha=0.2)
        ax.add_artist(rectangle)

    ax.legend(stim_types + ['aud_block'])
    return fig


def plot_licks_by_block(session: npc_sessions.DynamicRoutingSession) -> plt.Figure:
    """plot licks per trial, separated by block, go/nogo, and vis/aud

    loads multiple blocks (assumes vis and sound present)
    """

    f = session.task_data
    trialPreStimFrames=f['trialPreStimFrames'][:]
    preStimFramesFixed=f['preStimFramesFixed'][()]
    trialStimStartFrame=f['trialStimStartFrame'][:]
    trialEndFrame=f['trialEndFrame'][:]
    trialStimID=f['trialStim'][:].astype('str')
    lickFrames=f['lickFrames'][:]
    rewardFrames=f['rewardFrames'][:]
    trialAutoRewarded=f['trialAutoRewarded'][:]
    responseWindow=f['responseWindow'][:]
    manualRewardFrames=f['manualRewardFrames'][:]
    trialResponseFrame=f['trialResponseFrame'][:]
    quiescentViolationFrames=f['quiescentViolationFrames'][:]
    blockStim=f['blockStim'][:].astype('str')
    trialBlock=f['trialBlock'][:]
    blockStimRewarded=f['blockStimRewarded'][:].astype('str')
    rigName=f['rigName'][()]

    subjectName=f['subjectName'][()].decode()
    startTime=f['startTime'][()].decode()
    taskVersion=f['taskVersion'][()].decode()

    if 'trialOptoVoltage' in list(f.keys()):
        trialOptoVoltage=f['trialOptoVoltage'][:]
    else:
        trialOptoVoltage = []

    if type(blockStim)==str:
        blockStim=ast.literal_eval(blockStim)

    for xx in range(0,len(blockStim)):
        if type(blockStim[xx])==str:
            blockStim[xx]=ast.literal_eval(blockStim[xx])

    unique_blocks=np.unique(trialBlock)

    fig,ax=plt.subplots(2,3,figsize=(15,10))
    ax=ax.flatten()
    for aa in range(0,len(ax)):
        ax[aa].axvline(responseWindow[0],linewidth=0.25)
        ax[aa].axvline(responseWindow[1],linewidth=0.25)


    lickFrames=lickFrames[np.where(np.diff(lickFrames)>1)[0]]

    #specify vis go/nogo, aud go/nogo
    vis_go_trials=[]
    vis_nogo_trials=[]
    vis_hit_trials=[]
    vis_false_alarm_trials=[]
    vis_miss_trials=[]
    vis_correct_reject_trials=[]
    vis_autoreward_trials=[]
    vis_manualreward_trials=[]

    aud_go_trials=[]
    aud_nogo_trials=[]
    aud_hit_trials=[]
    aud_false_alarm_trials=[]
    aud_miss_trials=[]
    aud_correct_reject_trials=[]
    aud_autoreward_trials=[]
    aud_manualreward_trials=[]

    catch_trials=[]
    catch_resp_trials=[]

    vis_hit_title='vis hit'
    vis_fa_title='vis fa'
    aud_hit_title='aud hit'
    aud_fa_title='aud fa'
    catch_title='catch'

    block=[]
    stim_rewarded=[]
    start_time=[]

    vis_go_count=0
    vis_go_non_auto_reward=0
    vis_nogo_count=0
    vis_hit_count=0
    vis_fa_count=0

    aud_go_count=0
    aud_go_non_auto_reward=0
    aud_nogo_count=0
    aud_hit_count=0
    aud_fa_count=0

    catch_count=0
    catch_resp_count=0

    for bb in range(0,len(unique_blocks)):

        blockTrialStart=np.where(trialBlock==unique_blocks[bb])[0][0]
        blockTrialEnd=np.where(trialBlock==unique_blocks[bb])[0][-1]+1

        for tt in range(blockTrialStart,blockTrialEnd):
            if (tt>=len(trialEndFrame)):
                break
            temp_start_frame=trialStimStartFrame[tt]-250#preStimFramesFixed
            temp_end_frame=trialEndFrame[tt]

            temp_licks=[]
            temp_reward=[]
            temp_manual_reward=[]
            temp_quiescent_viol=[]

            temp_ax=[]
            temp_count=[]
            reward_color='y'

            if len(lickFrames)>0:
                temp_licks=np.copy(lickFrames)
                temp_licks=temp_licks[(temp_licks>temp_start_frame)&(temp_licks<temp_end_frame)]-trialStimStartFrame[tt]

            if len(rewardFrames)>0:
                temp_reward=np.copy(rewardFrames)
                temp_reward=temp_reward[(temp_reward>temp_start_frame)&(temp_reward<temp_end_frame)]-trialStimStartFrame[tt]

            if len(manualRewardFrames)>0:
                temp_manual_reward=np.copy(manualRewardFrames)
                temp_manual_reward=temp_manual_reward[(temp_manual_reward>temp_start_frame)&
                                                    (temp_manual_reward<temp_end_frame)]-trialStimStartFrame[tt]
            if len(quiescentViolationFrames)>0:
                temp_quiescent_viol=np.copy(quiescentViolationFrames)
                temp_quiescent_viol=temp_quiescent_viol[(temp_quiescent_viol>temp_start_frame)&
                                                    (temp_quiescent_viol<temp_end_frame)]-trialStimStartFrame[tt]

            temp_RW_lick=0
            for ii in temp_licks:
                if (ii>=responseWindow[0])&(ii<=responseWindow[1]):
                    temp_RW_lick=1

            #visual-go block
            if 'vis' in blockStimRewarded[bb]:
                if (trialStimID[tt] == 'vis1'):
                    vis_go_count+=1
                    temp_ax=0
                    temp_count=vis_go_count
                    if ~trialAutoRewarded[tt]:
                        vis_go_non_auto_reward+=1
                        reward_color=[0,1,0]
                        if temp_RW_lick:
                            vis_hit_count+=1
                    else:
                        reward_color=[1,0,0]

                elif (trialStimID[tt] == 'vis2'):
                    vis_nogo_count+=1
                    temp_ax=1
                    temp_count=vis_nogo_count
                    if temp_RW_lick:
                        vis_fa_count+=1

                elif trialStimID[tt] == 'catch':
                    catch_count+=1
                    temp_ax=2
                    temp_count=catch_count
                    if temp_RW_lick:
                        catch_resp_count+=1

                elif ('sound1' in trialStimID[tt]):
                    aud_go_count+=1
                    temp_ax=3
                    temp_count=aud_go_count
                    if ~trialAutoRewarded[tt]:
                        aud_go_non_auto_reward+=1
                        if temp_RW_lick:
                            aud_hit_count+=1

                elif ('sound2' in trialStimID[tt]):
                    aud_nogo_count+=1
                    temp_ax=4
                    temp_count=aud_nogo_count
                    if temp_RW_lick:
                        aud_fa_count+=1


            #sound-go block
            elif 'sound' in blockStimRewarded[bb]:
                if (trialStimID[tt] == 'sound1'):
                    aud_go_count+=1
                    temp_ax=3
                    temp_count=aud_go_count
                    if ~trialAutoRewarded[tt]:
                        aud_go_non_auto_reward+=1
                        reward_color=[0,1,0]
                        if temp_RW_lick:
                            aud_hit_count+=1
                    else:
                        reward_color=[1,0,0]
                    ax[3].vlines(temp_manual_reward,ymin=aud_go_count,ymax=aud_go_count+1,color=[0,0,1],linewidth=2)

                elif (trialStimID[tt] == 'sound2'):
                    aud_nogo_count+=1
                    temp_ax=4
                    temp_count=aud_nogo_count
                    if temp_RW_lick:
                        aud_fa_count+=1

                elif trialStimID[tt] == 'catch':
                    catch_count+=1
                    temp_ax=2
                    temp_count=catch_count
                    if temp_RW_lick:
                        catch_resp_count+=1

                elif ('vis1' in trialStimID[tt]):
                    vis_go_count+=1
                    temp_ax=0
                    temp_count=vis_go_count
                    if ~trialAutoRewarded[tt]:
                        vis_go_non_auto_reward+=1
                        if temp_RW_lick:
                            vis_hit_count+=1

                elif ('vis2' in trialStimID[tt]):
                    vis_nogo_count+=1
                    temp_ax=1
                    temp_count=vis_nogo_count
                    if temp_RW_lick:
                        vis_fa_count+=1

            if (len(trialOptoVoltage)>0):
                if ~np.isnan(trialOptoVoltage[tt]):
                    #ax[temp_ax].axhline(temp_count+0.5,color='b')
                    temp_patch=matplotlib.patches.Rectangle([0,temp_count],60,1,
                                            color=[0.2,0.2,0.8],alpha=0.15)
                    ax[temp_ax].add_patch(temp_patch)

            ax[temp_ax].vlines(temp_licks,ymin=temp_count,ymax=temp_count+1,color='grey')
            ax[temp_ax].vlines(temp_reward,ymin=temp_count,ymax=temp_count+1,color=reward_color,linewidth=2)
            ax[temp_ax].vlines(temp_manual_reward,ymin=temp_count,ymax=temp_count+1,color=[0,0,1],linewidth=2)

            ax[temp_ax].plot(temp_quiescent_viol,np.ones(len(temp_quiescent_viol))*temp_count+0.5,'m*')

            ax[temp_ax].set_xlim([-30,120])

            if ('vis1' in blockStim[bb])&('sound1' in blockStim[bb]):
                if aud_go_non_auto_reward==0:
                    aud_go_non_auto_reward=1
                if vis_go_non_auto_reward==0:
                    vis_go_non_auto_reward=1

        if bb<len(unique_blocks)-1:
            ax[0].axhline(vis_go_count+1,color='k',linestyle='--')
            ax[1].axhline(vis_nogo_count+1,color='k',linestyle='--')
            ax[2].axhline(catch_count+1,color='k',linestyle='--')
            ax[3].axhline(aud_go_count+1,color='k',linestyle='--')
            ax[4].axhline(aud_nogo_count+1,color='k',linestyle='--')

        vis_go_trials.append(vis_go_non_auto_reward-np.sum(vis_go_trials))
        vis_nogo_trials.append(vis_nogo_count-np.sum(vis_nogo_trials))
        vis_hit_trials.append(vis_hit_count-np.sum(vis_hit_trials))
        vis_false_alarm_trials.append(vis_fa_count-np.sum(vis_false_alarm_trials))
        vis_miss_trials.append((vis_go_non_auto_reward-vis_hit_count)-np.sum(vis_miss_trials))
        vis_correct_reject_trials.append((vis_nogo_count-vis_fa_count)-np.sum(vis_correct_reject_trials))
        vis_autoreward_trials.append((vis_go_count-vis_go_non_auto_reward)-np.sum(vis_autoreward_trials))

        aud_go_trials.append(aud_go_non_auto_reward-np.sum(aud_go_trials))
        aud_nogo_trials.append(aud_nogo_count-np.sum(aud_nogo_trials))
        aud_hit_trials.append(aud_hit_count-np.sum(aud_hit_trials))
        aud_false_alarm_trials.append(aud_fa_count-np.sum(aud_false_alarm_trials))
        aud_miss_trials.append((aud_go_non_auto_reward-aud_hit_count)-np.sum(aud_miss_trials))
        aud_correct_reject_trials.append((aud_nogo_count-aud_fa_count)-np.sum(aud_correct_reject_trials))
        aud_autoreward_trials.append((aud_go_count-aud_go_non_auto_reward)-np.sum(aud_autoreward_trials))

        catch_trials.append(catch_count-np.sum(catch_trials))
        catch_resp_trials.append(catch_resp_count-np.sum(catch_resp_trials))

        block.append(bb)
        stim_rewarded.append(blockStimRewarded[bb])
        start_time.append(startTime)

        ###make block titles more generalized like this
        temp_vis_hit_title=(' '+str(bb+1)+f':{(vis_hit_trials[bb]/vis_go_trials[bb])*100:.1f}%')
        vis_hit_title=vis_hit_title+temp_vis_hit_title

        temp_vis_fa_title=(' '+str(bb+1)+f':{(vis_false_alarm_trials[bb]/vis_nogo_trials[bb])*100:.1f}%')
        vis_fa_title=vis_fa_title+temp_vis_fa_title

        temp_catch_title=(' '+str(bb+1)+f':{(catch_resp_trials[bb]/catch_trials[bb])*100:.1f}%')
        catch_title=catch_title+temp_catch_title

        temp_aud_hit_title=(' '+str(bb+1)+f':{(aud_hit_trials[bb]/aud_go_trials[bb])*100:.1f}%')
        aud_hit_title=aud_hit_title+temp_aud_hit_title

        temp_aud_fa_title=(' '+str(bb+1)+f':{(aud_false_alarm_trials[bb]/aud_nogo_trials[bb])*100:.1f}%')
        aud_fa_title=aud_fa_title+temp_aud_fa_title


    ax[0].set_title(vis_hit_title,fontsize=10)
    ax[0].set_ylabel('trial')
    ax[0].set_xlabel('frames')

    ax[1].set_title(vis_fa_title,fontsize=10)
    ax[1].set_xlabel('frames')

    ax[2].set_title(catch_title,fontsize=10)
    ax[2].set_xlabel('frames')

    ax[3].set_title(aud_hit_title,fontsize=10)
    ax[3].set_xlabel('frames')

    ax[4].set_title(aud_fa_title,fontsize=10)
    ax[4].set_xlabel('frames')

    fig.suptitle(subjectName+' '+startTime+' '+taskVersion)

    fig.tight_layout()
    return fig

def plot_lick_raster_by_block(session: npc_sessions.DynamicRoutingSession) -> matplotlib.figure.Figure:
    lick_times = session.processing['behavior']['licks'].timestamps
    trials = pl.DataFrame(session.trials[:])
    lick_times_by_trial = tuple(lick_times[slice(*start_stop)] for start_stop in np.searchsorted(lick_times, trials.select('start_time', 'stop_time')))
    trials = (
        trials
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
    is_pass = len(
        pl.DataFrame(session.intervals["performance"][:])
        .filter(
            pl.col('same_modal_dprime') > 1.0,
            pl.col('cross_modal_dprime') > 1.0,
        )
    ) > 3

    scatter_params = dict(
        marker='.',
        s=15,
        color=[0.85] * 3,
        alpha=1,
        edgecolor='none',
    )
    line_params = dict(
        color='grey',
        lw=.5,
    )
    response_window_start_time = np.median(np.diff(trials.select('stim_start_time', 'response_window_start_time')))
    response_window_stop_time = np.median(np.diff(trials.select('stim_start_time', 'response_window_stop_time')))

    fig, axes = plt.subplots(1,2, figsize=(3, 6))
    for ax, stim_name in zip(axes, ("vis1", "sound1")):
        ax: plt.Axes
        
        idx_in_block = 0
        previous_trial_idx = -1
        stim_trials = trials.filter(pl.col('stim_name') == stim_name)
        for idx, trial in enumerate(stim_trials.iter_rows(named=True)):
        
            is_vis_block: bool = "vis" in trial["context_name"]
            is_vis_stim: bool = "vis" in trial["stim_name"]
            
            # block label
            if idx_in_block == (len(stim_trials.filter(pl.col('block_index') == trial['block_index'])) // 2):
                text_color = 'black' if is_vis_block == is_vis_stim else 'grey'
                rotation = 0
                label = "V" if "vis" in trial["context_name"] else "A"
                ax.text(-0.4, idx, label, fontsize=8, ha='center', va='center', color=text_color, rotation=rotation)
            
            # block switch horizontal lines
            if trial['trial_index_in_block'] < previous_trial_idx:
                ax.axhline(idx - .5, **line_params)
                idx_in_block = 0
            previous_trial_idx = trial['trial_index_in_block']
            idx_in_block += 1

            # response window vertical lines
            ax.axvline(response_window_start_time, **line_params)
            ax.axvline(response_window_stop_time, **line_params)
            times = trial['stim_centered_lick_times']
            
            # licks
            ax.scatter(times, np.full_like(times, idx), **scatter_params)
            
            # first licks of interest
            if trial['is_hit']:
                first_lick_color = 'g'
            elif trial['is_false_alarm']:
                first_lick_color = 'm'
            elif not trial['is_contingent_reward'] and trial['is_rewarded']:
                first_lick_color = 'c'
            else:
                continue
            colored_time = times[0]
            ax.scatter(colored_time, idx, **scatter_params | dict(color=first_lick_color, alpha=1))
            
        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(-0.5, idx + 0.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels("" if v%2 else str(v) for v in ax.get_xticks())
        ax.set_yticks([])
        if ax is axes[0]:
            ax.set_ylabel("← trials (non-consecutive)")
            ax.yaxis.set_label_coords(-0.3, 0.91) 
            ax.text(-0.4, -0.5, "← context", fontsize=8, ha='center', va='center', color='k', rotation=90)

        ax.set_xlabel("time rel. to\nstim onset (s)")
        ax.invert_yaxis()
        ax.set_aspect(0.1)
        ax.set_title("VIS+" if is_vis_stim else "AUD+", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    fig.suptitle(f"{'pass' if is_pass else 'fail'}\n{session.id}")
    return fig