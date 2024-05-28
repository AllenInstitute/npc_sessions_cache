from __future__ import annotations

import importlib.metadata

import npc_lims


def get_npc_sesions_version() -> str:
    """Get the version of the `npc_sessions` package."""
    return importlib.metadata.version("npc_sessions")


def assert_s3_write_credentials() -> None:
    test = npc_lims.DR_DATA_REPO / "test.txt"
    test.touch()
    test.unlink()

if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
