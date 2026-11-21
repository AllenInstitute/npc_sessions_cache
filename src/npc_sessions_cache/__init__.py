import importlib.metadata

import dotenv
import numpy as np

from npc_sessions_cache.notebooks import *
from npc_sessions_cache.plots import *
from npc_sessions_cache.utils import *

__version__ = importlib.metadata.version("npc-sessions-cache")

_ = dotenv.load_dotenv(
    dotenv.find_dotenv(usecwd=True)
)  # take environment variables from .env

np.seterr(divide="ignore", invalid="ignore")
# suppress common warning from sam's DynamicRoutingAnalysisUtils

def _patch_property_dict_string_nan() -> None:
    """Fill missing values in `str`-dtype trials columns with "" (as npc_sessions does for
    object columns). Under pandas>=3 string columns get the `str` dtype, so npc_sessions<=0.0.298
    skips the fill and NWB writing fails on mixed str/NaN columns (e.g. trials.opto_label in
    670248_2023-08-03). Only changes sessions that would otherwise fail; fixed upstream in
    npc_sessions property_dict.py."""
    import pandas as pd
    import npc_sessions.trials.property_dict as property_dict

    cls = property_dict.PropertyDict
    if getattr(cls.to_dataframe, "_str_nan_patched", False):
        return
    original = cls.to_dataframe

    def to_dataframe(self, *args, **kwargs):
        df = original(self, *args, **kwargs)
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
                df[col] = df[col].fillna("")
        return df

    to_dataframe._str_nan_patched = True  # type: ignore[attr-defined]
    cls.to_dataframe = to_dataframe  # type: ignore[method-assign]


_patch_property_dict_string_nan()


def _patch_empty_quiescent_interval_violations() -> None:
    """npc_sessions<=0.0.298: DynamicTable.from_dataframe fails on an empty dataframe, so NWB
    writing fails for sessions with no quiescent-interval violations (e.g. 728917_2024-10-04).
    Build the empty table directly instead (fixed upstream in npc_sessions sessions.py). Only
    changes sessions that would otherwise fail."""
    import numpy as np
    import npc_sessions
    import pynwb

    for klass in npc_sessions.DynamicRoutingSession.__mro__:
        desc = klass.__dict__.get("_quiescent_interval_violations")
        if desc is not None:
            break
    else:
        return
    if getattr(desc.func, "_empty_patched", False):
        return
    original = desc.func

    def func(self):
        try:
            return original(self)
        except IndexError:
            if len(self.sam.quiescentViolationFrames) != 0:
                raise
        table = pynwb.core.DynamicTable(
            name="quiescent_interval_violations",
            description="times at which the subject made contact with the lick spout during a pre-trial quiescent interval, triggering a restart of the trial",
        )
        table.add_column(
            name="timestamps",
            description="time at which the subject made contact with the lick spout during a pre-trial quiescent interval, triggering a restart of the trial",
            data=np.array([], dtype=np.float64),
        )
        return table

    func._empty_patched = True  # type: ignore[attr-defined]
    desc.func = func


_patch_empty_quiescent_interval_violations()
