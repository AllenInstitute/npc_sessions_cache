# %%
# import matplotlib.pyplot as plt
import pathlib

import numpy as np
import polars as pl
import npc_sessions_cache.figures.paper2.utils as utils
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
                )
            ),
            on="session_id",
            how="semi",
        )
        .filter(pl.col("is_first_block_aud"))
    )


# %%


def plot(with_color: bool = False, with_average_line: bool = False):
    # %%
    perf = get_filtered_performance()
    for modality, color in zip(("aud", "vis"), "mg"):
        fig, ax = plt.subplots(figsize=(2, 2))
        rate_col = f"{modality}_target_response_rate"
        df = perf.pivot(
            values=rate_col,
            index="subject_id",
            columns="block_index",
            aggregate_function="mean",
            sort_columns=True,
        ).drop("subject_id")
        x = np.arange(1, 7)
        y = df.to_numpy()
        c=[0.8]*3 if not with_color else color
        ax.plot(
            x,
            y.T,
            c=c,
            alpha=0.8 if not with_color else 0.25,
            lw=0.3,
        )
        if with_average_line:
            ax.plot(
                x,
                df.mean().to_numpy().squeeze(),
                c=c,
                lw=0.8,
            )
        ax.scatter(
            x,
            df.mean().to_numpy().squeeze(),
            c=list("cr") * 3 if modality == "aud" else list("rc") * 3,
            s=6,
            edgecolor="none",
            zorder=99,
        )
        lower = np.full(len(x), np.nan)
        upper = np.full(len(x), np.nan)
        for i in range(len(x)):
            ys = y[~np.isnan(y[:, i]), i]
            lower[i], upper[i] = np.percentile(
                [
                    np.nanmean(np.random.choice(ys, size=ys.size, replace=True))
                    for _ in range(1000)
                ],
                (5, 95),
            )
        # add vertical lines as error bars
        ax.vlines(
            x=x,
            ymin=lower,
            ymax=upper,
            color=[0.5] * 3,
            lw=1,
        )
        ax.set_ylim(0, 1.05)
        # ax.set_xlim(5,5.5)
        ax.set_xticks(range(1, 7))
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlabel("Block #")
        ax.set_ylabel(f"{modality.upper()}+ response probability")

        plt.tight_layout()

        pyfile_path = pathlib.Path(__file__)
        figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}_{modality}+_{'color' if with_color else 'grey'}")
        fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches="tight")
        df.describe(.5)
    
    # %%


if __name__ == "__main__":
    for with_color in (True, False):
        plot(with_color=with_color)
