# aligned blocks - standalone

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import npc_session
import numpy.typing as npt
import polars as pl

import utils 


def get_rate_expr(stim: str, is_target: bool):
    stim_col = pl.col(f'is_{stim}_{"non" if not is_target else ""}target')
    response_trials = (stim_col & pl.col('is_response') & ~pl.col('is_reward_scheduled')).sum()
    total_trials = stim_col.sum()
    return (response_trials / total_trials).over(['session_id', 'block_index'])

def plot() -> plt.Figure:

    df = (
        utils.get_prod_trials(cross_modal_dprime_threshold=1.5)
        # calculate response rates in each block:
        .with_columns(
            get_rate_expr(stim='vis', is_target=True).alias(a := 'vis_target_response_rate'),
            get_rate_expr(stim='vis', is_target=False).alias(b := 'vis_nontarget_response_rate'),
            get_rate_expr(stim='aud', is_target=True).alias(c := 'aud_target_response_rate'),
            get_rate_expr(stim='aud', is_target=False).alias(d := 'aud_nontarget_response_rate'),
        )
        # don't calculate median by block yet - we need to filter sessions that start
        # with vis or aud block first 
        .group_by('session_id', 'block_index')
        .agg(
            pl.col('subject_id').first(),
            pl.col('is_first_block_aud').first(),
            pl.col(a, b, c, d).first(),
        )
    )
    print(df.sort('session_id'), df.describe())
    print(df.unique('session_id').select(pl.col('is_first_block_aud').first().over('session_id'))['is_first_block_aud'].value_counts())
    
    
    is_first_block_aud = False
    is_boxplot = False
    is_ci_lines = True
    is_median_marker = False
    is_median_line = True

    fig, axes = plt.subplots(1, df.n_unique('block_index'), figsize=(df.n_unique('block_index') * 2, 3), sharey=True)

    modalitites = ('aud', 'vis') if is_first_block_aud else ('vis', 'aud') 
    xpos = (0, 1)

    common_line_params = dict(alpha=1, lw=.3)
    line_params = {
        'target': common_line_params.copy(),
        'nontarget': common_line_params.copy(),
    }
    line_params['target'] |= dict(c=[0.5]*3)
    line_params['nontarget'] |= dict(c=[0.7]*3, ls=':')

    for block_index in df['block_index'].unique().sort():
        block_df = (
            df
            .filter(
                pl.col('block_index') == block_index,
                pl.col('is_first_block_aud') == is_first_block_aud,
            )
            .group_by('subject_id')
            .agg(
                pl.selectors.ends_with('_response_rate').drop_nans().mean(),
                pl.col('block_index').first(),
            )
        )
        ax: plt.Axes = axes[block_index]
        box_data = []
        for target in ('nontarget', 'target'):
            y = block_df.select(*[f"{modality}_{target}_response_rate" for modality in modalitites]).to_numpy().T
            ax.plot(xpos, y, **line_params[target])
            if is_median_marker:
                ax.plot(xpos, np.median(y, axis=-1), '+', c=-0.1 + np.array(line_params[target]['c']), ms=6, zorder=99)
            if is_median_line:
                ax.plot(xpos, np.median(y, axis=-1), c=-0.1 + np.array(line_params[target]['c']), lw=1.5, zorder=99)
            
            box_data.extend([y[0, :].flatten(), y[1, :].flatten()])
            
        if is_boxplot:
            lw = .5
            boxplot_objects = ax.boxplot(
                box_data,
                positions=[0, 1, 0, 1],
                widths=0.05,
                patch_artist=True,
                bootstrap=10_000,
                notch=True,
                showfliers=False,
                boxprops=dict(linewidth=lw, facecolor='none'),
                medianprops=dict(linewidth=lw),
                whiskerprops=dict(linewidth=0),
                capprops=dict(linewidth=0),
            )
            for box in boxplot_objects['boxes']:
                box.set_edgecolor([0.4]*3)
                
        if is_ci_lines:
            import matplotlib.cbook
            stats = matplotlib.cbook.boxplot_stats(box_data, bootstrap=10_000)
            for i, stat in enumerate(stats):
                ax.plot([xpos[i % 2]]*2, [stat['cilo'], stat['cihi']], c=[0.4]*3, lw=1.5, zorder=98)
                #TODO save stats
                
        context = ('vis', 'aud')[(block_index + is_first_block_aud) % 2]
        ax.set_title('V' if context == 'vis' else 'A')
        ax.set_aspect(1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        if block_index == 0:
            ax.set_ylabel("response rate")
            ax.set_yticks([0, 0.5, 1])
            ax.spines['left'].set_visible(True)
        else:
            ax.yaxis.set_visible(False)

        x_pad = 0.05
        ax.set_xlim(xpos[0] - x_pad, xpos[1] + x_pad)
        ax.set_xticks(xpos)
        ax.set_xticklabels(modalitites)
    fig.tight_layout()
        
    return fig

if __name__ == "__main__":
    fig = plot()
    
        
    pyfile_path = pathlib.Path(__file__)
    figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}")
    fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches='tight')
    
    # make sure text is editable in illustrator before saving pdf:
    plt.rcParams['pdf.fonttype'] = 42 
    fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches='tight')