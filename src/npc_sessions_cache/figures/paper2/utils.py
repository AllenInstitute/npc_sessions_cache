from __future__ import annotations

import functools
import logging
import pathlib
import tempfile

import npc_lims
import nrrd
import numba
import numpy as np
import numpy.typing as npt
import polars as pl
import upath
import zarr
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["pdf.fonttype"] = 42

logger = logging.getLogger(__name__)

CCF_MIDLINE_ML = 5700

CACHE_VERSION = "v0.0.235"


@functools.cache
def get_component_lf(nwb_component: npc_lims.NWBComponentStr) -> pl.DataFrame:
    path = npc_lims.get_cache_path(
        nwb_component,
        version=CACHE_VERSION,
        consolidated=True,
    )
    logger.info(f"Reading dataframe from {path}")
    return pl.scan_parquet(path.as_posix())


@functools.cache
def get_component_df(nwb_component: npc_lims.NWBComponentStr) -> pl.DataFrame:
    return get_component_lf(nwb_component).collect()


@functools.cache
def get_component_zarr(nwb_component: npc_lims.NWBComponentStr) -> zarr.Group:
    path = npc_lims.get_cache_path(
        nwb_component,
        version=CACHE_VERSION,
        consolidated=True,
    )
    logger.info(f"Reading zarr file from {path}")
    return zarr.open(path)


@functools.cache
def get_ccf_structure_tree_df() -> pl.DataFrame:
    local_path = upath.UPath(
        "//allen/programs/mindscope/workgroups/np-behavior/ccf_structure_tree_2017.csv"
    )
    cloud_path = upath.UPath(
        "https://raw.githubusercontent.com/cortex-lab/allenCCF/master/structure_tree_safe_2017.csv"
    )
    path = local_path if local_path.exists() else cloud_path
    logging.info(f"Using CCF structure tree from {path.as_posix()}")
    return (
        pl.read_csv(path.as_posix())
        .lazy()
        .with_columns(
            color_hex_int=pl.col("color_hex_triplet").str.to_integer(base=16),
            color_hex_str=pl.lit("0x") + pl.col("color_hex_triplet"),
        )
        .with_columns(
            r=pl.col("color_hex_triplet")
            .str.slice(0, 2)
            .str.to_integer(base=16)
            .mul(1 / 255),
            g=pl.col("color_hex_triplet")
            .str.slice(2, 2)
            .str.to_integer(base=16)
            .mul(1 / 255),
            b=pl.col("color_hex_triplet")
            .str.slice(4, 2)
            .str.to_integer(base=16)
            .mul(1 / 255),
        )
        .with_columns(
            color_rgb=pl.concat_list("r", "g", "b"),
        )
        .drop("r", "g", "b")
    ).collect()


@functools.cache
def get_good_units_df() -> pl.DataFrame:
    good_units = (
        get_component_lf("session")
        .filter(pl.col("keywords").list.contains("templeton").not_())
        .join(
            other=(
                get_component_lf("performance")
                .filter(
                    pl.col("same_modal_dprime") > 1.0,
                    pl.col("cross_modal_dprime") > 1.0,
                )
                .group_by(pl.col("session_id"))
                .agg(
                    [
                        (pl.col("block_index").count() > 3).alias("pass"),
                    ],
                )
                .filter("pass")
                .drop("pass")
            ),
            on="session_id",
            how="semi",  # only keep rows in left table (sessions) that have match in right table (ie pass performance)
        )
        .join(
            other=(
                get_component_lf("units").filter(
                    pl.col("isi_violations_ratio") < 0.5,
                    pl.col("amplitude_cutoff") < 0.1,
                    pl.col("presence_ratio") > 0.95,
                )
            ),
            on="session_id",
        )
        .join(
            other=(
                get_component_lf("electrode_groups")
                .rename(
                    {
                        "name": "electrode_group_name",
                        "location": "implant_location",
                    }
                )
                .select("session_id", "electrode_group_name", "implant_location")
            ),
            on=("session_id", "electrode_group_name"),
        )
        .with_columns((pl.col("ccf_ml") > CCF_MIDLINE_ML).alias("is_right_hemisphere"))
        .join(
            other=get_ccf_structure_tree_df().lazy(),
            right_on="acronym",
            left_on="location",
        )
    ).collect()
    logger.info(f"Fetched {len(good_units)} good units")
    return good_units

from typing import TypeVar

T = TypeVar("T", pl.DataFrame, pl.LazyFrame)
def filter_prod_sessions(
    df: T,
    cross_modal_dprime_threshold: float = 1.0,
    late_autorewards: bool | None = None,
) -> T:
    """
    Filter the dataframe to only include sessions that are pass dprime threshold
    specified in at least 3 blocks. 
    
    usage:
    electrodes = get_component_df("electrodes").pipe(filter_prod_sessions, cross_modal_dprime_threshold=1.0)    
    """
    prod_trials = get_prod_trials(cross_modal_dprime_threshold, late_autorewards)
    if isinstance(df, pl.LazyFrame):
        prod_trials = prod_trials.lazy()
    return (
        df
        .join(
            other=prod_trials,
            on="session_id",
            how="semi", # only keep rows in left table that have match in right table (ie prod sessions)
        )
    )

@functools.cache
def get_prod_trials(
    cross_modal_dprime_threshold: float = 1.0, late_autorewards: bool | None = None
) -> pl.DataFrame:
    if late_autorewards is None:
        late_autorewards_expr = pl.lit(True)
    elif late_autorewards == True:
        late_autorewards_expr = (
            pl.col("keywords").list.contains("late_autorewards") == True
        )
    elif late_autorewards == False:
        late_autorewards_expr = (
            pl.col("keywords").list.contains("early_autorewards") == True
        )

    return (
        get_component_df("trials")
        .join(
            other=(
                get_component_df("session").filter(
                    pl.col("keywords").list.contains("production"),
                    ~pl.col("keywords").list.contains("issues"),
                    pl.col("keywords").list.contains("task"),
                    pl.col("keywords").list.contains("ephys"),
                    pl.col("keywords").list.contains("ccf"),
                    ~pl.col("keywords").list.contains("opto_perturbation"),
                    ~pl.col("keywords").list.contains("opto_control"),
                    ~pl.col("keywords").list.contains("injection_perturbation"),
                    ~pl.col("keywords").list.contains("injection_control"),
                    ~pl.col("keywords").list.contains("hab"),
                    ~pl.col("keywords").list.contains("training"),
                    ~pl.col("keywords").list.contains("context_naive"),
                    ~pl.col("keywords").list.contains("templeton"),
                    late_autorewards_expr,
                )
            ),
            on="session_id",
            how="semi",
        )
        # exclude sessions based on task performance:
        .join(
            other=(
                get_component_df("performance")
                .filter(
                    # pl.col('same_modal_dprime') > 1.0,
                    pl.col("cross_modal_dprime")
                    > cross_modal_dprime_threshold,
                )
                .with_columns(
                    pl.col("block_index")
                    .count()
                    .over("session_id")
                    .alias("n_passing_blocks"),
                )
                .filter(
                    pl.col("n_passing_blocks") > 3,
                )
            ),
            on="session_id",
            how="semi",
        )
        # filter blocks with too few trials:
        .with_columns(
            pl.col("trial_index_in_block")
            .max()
            .over("session_id", "block_index")
            .alias("n_trials_in_block"),
        )
        .filter(
            pl.col("n_trials_in_block") > 10,
        )
        # filter sessions with too few blocks:
        .filter(
            pl.col("block_index").n_unique().over("session_id") == 6,
            pl.col("block_index").max().over("session_id") == 5,
        )
        # add a column that indicates if the first block in a session is aud context:
        .with_columns(
            (pl.col("context_name").first() == "aud")
            .over("session_id")
            .alias("is_first_block_aud"),
        )
    )


def copy_parquet_files_to_home() -> None:
    for component in (
        "units",
        "session",
        "subject",
        "trials",
        "epochs",
        "performance",
        "devices",
        "electrode_groups",
        "electrodes",
    ):
        source = npc_lims.get_cache_path(
            component,
            version=CACHE_VERSION,
            consolidated=True,
        )
        dest = upath.UPath(
            f"//allen/ai/homedirs/ben.hardcastle/dr-dashboard/data/{CACHE_VERSION}/{component}.parquet"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())


def savefig(py__file__: str, fig: plt.Figure, suffix: str):
    pyfile_path = pathlib.Path(py__file__)
    suffix = suffix[1:] if suffix.startswith("_") else suffix
    figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}_{suffix}")
    fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches="tight")


def write_unit_context_columns(units_df: pl.DataFrame | None = None) -> None:
    """Takes ~5 hours for all units"""
    import tqdm

    import npc_sessions_cache.plots.ephys as ephys

    all_new_cols = []
    if units_df is None:
        units_df = get_good_units_df()
    for unit_id, session_id in tqdm.tqdm(
        units_df["unit_id", "session_id"].iter_rows(), total=len(get_good_units_df())
    ):
        new_cols = {}
        spike_times_session_id = "_".join(unit_id.split("_")[:2])
        unit_spike_times = get_component_zarr("spike_times")[spike_times_session_id][
            unit_id
        ][:]
        trials = get_component_df("trials").filter(pl.col("session_id") == session_id)
        new_cols["unit_id"] = unit_id
        for stim in ("vis", "aud"):
            for target in ("target", "nontarget"):
                for context in ("vis", "aud"):
                    psth = ephys.makePSTH_numba(
                        spikes=unit_spike_times,
                        startTimes=trials.filter(
                            pl.col(f"is_{stim}_{target}"),
                            pl.col(f"is_{context}_context"),
                        )["quiescent_start_time"].to_numpy(),
                        windowDur=1.5,
                        binSize=0.025,
                    )
                    new_cols[f"{stim}_{target}_{context}_context_baseline_rate"] = (
                        np.mean(psth)
                    )
                a = new_cols[f"{stim}_{target}_aud_context_baseline_rate"]
                v = new_cols[f"{stim}_{target}_vis_context_baseline_rate"]
                new_cols[f"{stim}_{target}_context_selectivity_index"] = (a - v) / (
                    a + v
                )
        all_new_cols.append(new_cols)
    pl.DataFrame(all_new_cols).write_parquet("unit_context_columns.parquet")


def get_units_with_context_columns(
    units_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if units_df is None:
        units_df = get_good_units_df()
    return units_df.join(pl.read_parquet("unit_context_columns.parquet"), on="unit_id")


@numba.njit
def makePSTH_numba(
    spikes: npt.NDArray[np.floating],
    startTimes: npt.NDArray[np.floating],
    windowDur: float,
    binSize: float = 0.001,
    convolution_kernel: float = 0.05,
):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel / 2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0, windowDur + binSize, binSize)
    convkernel = np.ones(int(convolution_kernel / binSize))
    counts = np.zeros(bins.size - 1)
    for i, start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start + windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd] - start, bins)[0]

    counts = counts / startTimes.size
    counts = np.convolve(counts, convkernel) / (binSize * convkernel.size)
    return (
        counts[convkernel.size - 1 : -convkernel.size],
        bins[: -convkernel.size - 1],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    copy_parquet_files_to_home()
