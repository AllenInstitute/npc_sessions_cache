# %%
# import matplotlib.pyplot as plt
import pathlib
import numpy as np
import polars as pl
import utils
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["pdf.fonttype"] = 42


def get_filtered_performance() -> pl.DataFrame:
    return (
        utils.get_component_df("performance")
        .join(
            other=(
                utils.get_prod_trials(
                    cross_modal_dprime_threshold=1.5,
                    late_autorewards=True,
                )
            ),
            on="session_id",
            how="semi",
        )
        .filter(pl.col("is_first_block_aud"))
    )


# %%


def plot():
    # %%
    perf = get_filtered_performance()
    for modality, color in zip(("aud", "vis"), "mg"):
        fig, ax = plt.subplots(figsize=(2,2))
        df = (
            perf
            .pivot(
                values=f"{modality}_target_response_rate",
                index="subject_id",
                columns="block_index",
                aggregate_function='mean',
                sort_columns=True,
            )
            .drop('subject_id')
        )
        x = np.arange(1,7)
        y = df.to_numpy()
        ax.plot(
            x, y.T,
            c=[0.6] * 3,
            alpha=0.5,
            lw=0.5,
        )
        ax.plot(
            x, df.median().to_numpy().squeeze(),
            c=color,
            lw=.8,
        )
        ax.scatter(
            x, df.median().to_numpy().squeeze(),
            c=list('rcrcrc') if modality == 'aud' else list('crcrcr'),
            s=6,
            edgecolor='none',
            zorder=99,
        )
        lower = np.full(len(x), np.nan)
        upper = np.full(len(x), np.nan)
        for i in range(len(x)):
            ys = y[~np.isnan(y[:, i]), i]
            lower[i], upper[i] = np.percentile(
                [
                    np.nanmedian(
                        np.random.choice(ys, size=ys.size, replace=True)
                    )
                    for _ in range(1000)
                ],
                (5, 95),
            )
        # add vertical lines as error bars
        ax.vlines(
            x=x,
            ymin=lower,
            ymax=upper,
            color=[.4]*3,
            lw=1.5,
        )
        ax.set_ylim(0,1.05)
        # ax.set_xlim(5,5.5)
        ax.set_xticks(range(1,7))
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('Block #')
        ax.set_ylabel(f'{modality.upper()}+ response probability')

    plt.tight_layout()

    pyfile_path = pathlib.Path(__file__)
    figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}_{modality}+")
    fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches="tight")



    # %%
if __name__ == "__main__":
    plot()