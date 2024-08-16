import functools
import logging
import polars as pl
import npc_lims
import upath
import zarr

logger = logging.getLogger(__name__)

CCF_MIDLINE_ML = 5700

CACHE_VERSION = 'v0.0.231'

@functools.cache
def get_component_lf(nwb_component: npc_lims.NWBComponentStr) -> pl.LazyFrame:
    path = npc_lims.get_cache_path(
        nwb_component, 
        version=CACHE_VERSION,
        consolidated=True,
    )
    logger.info(f"Reading dataframe from {path}")
    return pl.scan_parquet(path)

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
    local_path = upath.UPath('//allen/programs/mindscope/workgroups/np-behavior/ccf_structure_tree_2017.csv')
    cloud_path = upath.UPath('https://raw.githubusercontent.com/cortex-lab/allenCCF/master/structure_tree_safe_2017.csv')
    path = local_path if local_path.exists() else cloud_path
    logging.info(f"Using CCF structure tree from {path.as_posix()}")
    return (
        pl.scan_csv(path.as_posix())
        .with_columns(
            color_hex_int=pl.col('color_hex_triplet').str.to_integer(base=16),
            color_hex_str=pl.lit('0x') + pl.col('color_hex_triplet'),
        )
        .with_columns(
            r=pl.col('color_hex_triplet').str.slice(0, 2).str.to_integer(base=16).mul(1/255),
            g=pl.col('color_hex_triplet').str.slice(2, 2).str.to_integer(base=16).mul(1/255),
            b=pl.col('color_hex_triplet').str.slice(4, 2).str.to_integer(base=16).mul(1/255),
        )
        .with_columns(
            color_rgb=pl.concat_list('r', 'g', 'b'),
        )
        .drop('r', 'g', 'b')
    ).collect()

@functools.cache
def get_good_units_df() -> pl.DataFrame:
    good_units = (
        get_component_lf("session")
        .filter(
            pl.col('keywords').list.contains('templeton').not_()
        )
        .join(
            other=(
                get_component_lf("performance")
                .filter(
                    pl.col('same_modal_dprime') > 1.0,
                    pl.col('cross_modal_dprime') > 1.0,
                )
                .group_by(
                    pl.col('session_id')).agg(
                    [
                        (pl.col('block_index').count() > 3).alias('pass'),
                    ],  
                )
                .filter('pass')
                .drop('pass')
            ),
            on='session_id',
            how="semi", # only keep rows in left table (sessions) that have match in right table (ie pass performance)
        )
        .join(
            other=(
                get_component_lf("units")
                .filter(
                    pl.col('isi_violations_ratio') < 0.5,
                    pl.col('amplitude_cutoff') < 0.1,
                    pl.col('presence_ratio') > 0.95,
                )
            ),
            on='session_id',
        )
        .join(
            other=(
                get_component_lf('electrode_groups')
                    .rename({
                        'name': 'electrode_group_name',
                        'location': "implant_location",
                    })
                    .select('session_id', 'electrode_group_name', 'implant_location')
            ),
            on=('session_id', 'electrode_group_name'),
        )
        .with_columns((pl.col('ccf_ml') > CCF_MIDLINE_ML).alias('is_right_hemisphere'))
        .join(
            other=get_ccf_structure_tree_df().lazy(),
            right_on='acronym',
            left_on='location',
        )
    ).collect()
    logger.info(f"Fetched {len(good_units)} good units")
    return good_units

def copy_parquet_files_to_home() -> None:
    for component in ('units', 'session', 'subject', 'trials', 'epochs', 'performance', 'devices', 'electrode_groups', 'electrodes'):
        source = npc_lims.get_cache_path(
            component, 
            version=CACHE_VERSION,
            consolidated=True,
        )
        dest = upath.UPath(f'//allen/ai/homedirs/ben.hardcastle/dr-dashboard/data/{CACHE_VERSION}/{component}.parquet')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    copy_parquet_files_to_home()