from __future__ import annotations

import importlib.metadata

import typing

import npc_lims

if typing.TYPE_CHECKING:
    import pandas as pd


def get_npc_sesions_version() -> str:
    """Get the version of the `npc_sessions` package."""
    return importlib.metadata.version("npc_sessions")


def assert_s3_write_credentials() -> None:
    test = npc_lims.DR_DATA_REPO / "test.txt"
    test.touch()
    test.unlink()

def get_valid_epochs(epochs: pd.DataFrame) -> pd.DataFrame:
    """Rows of an epochs table for usable stim blocks: without the rows that have
    `is_qc_pass == False` (blocks excluded by npc_sessions: they keep a row, with the tag
    'invalid', the reason in `notes` and no intervals table). Tables without the column
    (older npc_sessions) are returned as they are.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"script_name": ["RFMapping", "DynamicRouting1"], "is_qc_pass": [True, False]})
    >>> get_valid_epochs(df).script_name.to_list()
    ['RFMapping']
    >>> get_valid_epochs(df.drop(columns="is_qc_pass")).script_name.to_list()
    ['RFMapping', 'DynamicRouting1']
    """
    if "is_qc_pass" not in epochs.columns:
        return epochs
    return epochs[[v is not False and v != False for v in epochs["is_qc_pass"]]]  # noqa: E712


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
