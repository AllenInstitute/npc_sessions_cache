from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import logging
import pathlib
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import aind_session
import npc_session
import npc_lims.metadata.codeocean_utils as codeocean_utils
import npc_lims.paths.s3 as npc_lims_s3

DEFAULT_SESSION_DIR = pathlib.Path(
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\qc"
    r"\3_ccf_annotations\ccf_aligned_ephys"
)
DEFAULT_OUTPUT_FILENAME = "session_asset_paths.csv"

SESSION_ID_IN_FILENAME = re.compile(r"(?P<session>\d{6}_\d{4}-\d{2}-\d{2})")

logger = logging.getLogger(__name__)


@dataclass
class SessionAssetPaths:
    session_id: str
    input_session_id: str = ""
    source_path: str = ""
    aind_session_id: str = ""
    raw_data_asset_id: str = ""
    raw_data_dir: str = ""
    raw_data_lookup_error: str = ""
    sorted_data_asset_id: str = ""
    sorted_data_asset_name: str = ""
    sorted_data_dir: str = ""
    sorted_data_lookup_error: str = ""


def discover_session_ids(session_dir: pathlib.Path) -> dict[str, str]:
    if not session_dir.exists():
        raise FileNotFoundError(f"Session dir does not exist: {session_dir}")
    if not session_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {session_dir}")

    session_id_to_source_path: dict[str, str] = {}
    for path in sorted(p for p in session_dir.iterdir() if p.is_file()):
        if not (match := SESSION_ID_IN_FILENAME.search(path.stem)):
            continue
        session_id = normalize_session_id(match.group("session"))
        session_id_to_source_path.setdefault(session_id, path.as_posix())
    return session_id_to_source_path


def normalize_session_id(session_id: str) -> str:
    try:
        return str(npc_session.SessionRecord(session_id))
    except ValueError:
        return aind_session.Session(session_id).id


def get_aind_session(session_id: str) -> aind_session.Session:
    with contextlib.suppress(ValueError):
        return aind_session.Session(session_id)

    record = npc_session.SessionRecord(session_id)
    sessions = aind_session.get_sessions(
        subject_id=record.subject,
        date=record.date,
        platform="ecephys",
    )
    if not sessions:
        sessions = aind_session.get_sessions(
            subject_id=record.subject,
            date=record.date,
        )
    if len(sessions) <= record.idx:
        raise IndexError(
            f"Found {len(sessions)} AIND session(s) for {session_id}, "
            f"but session index is {record.idx}: {[s.id for s in sessions]}"
        )
    return sessions[record.idx]


def get_npc_session_id(session_id: str, session: aind_session.Session) -> str:
    with contextlib.suppress(ValueError):
        return str(npc_session.SessionRecord(session_id))

    sessions = aind_session.get_sessions(
        subject_id=session.subject_id,
        date=session.date,
        platform=session.platform,
    )
    idx = sessions.index(session) if session in sessions else 0
    record = npc_session.SessionRecord(f"{session.subject_id}_{session.date}")
    if idx:
        record = record.with_idx(idx)
    return str(record)


def get_session_asset_paths(
    session_id: str,
    source_path: str = "",
) -> SessionAssetPaths:
    row = SessionAssetPaths(
        session_id=normalize_session_id(session_id),
        input_session_id=session_id,
        source_path=source_path,
    )

    session: aind_session.Session | None = None
    try:
        session = get_aind_session(row.session_id)
        row.session_id = get_npc_session_id(row.session_id, session)
        row.aind_session_id = session.id
        row.raw_data_dir = session.raw_data_dir.as_posix()
        try:
            row.raw_data_asset_id = session.raw_data_asset.id
        except Exception as exc:
            logger.debug(
                "Raw data asset ID lookup failed for %s: %s",
                row.session_id,
                exc,
            )
    except Exception as exc:
        row.raw_data_lookup_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Raw data lookup failed for %s: %s", row.session_id, exc)

    try:
        sorted_asset = codeocean_utils.get_session_sorted_data_asset(row.session_id)
        row.sorted_data_asset_id = sorted_asset.id
        row.sorted_data_asset_name = sorted_asset.name
        row.sorted_data_dir = npc_lims_s3.get_data_asset_s3_path(
            sorted_asset
        ).as_posix()
    except Exception as exc:
        row.sorted_data_lookup_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Sorted asset lookup failed for %s: %s", row.session_id, exc)

    return row


def read_existing_rows(output: pathlib.Path) -> dict[str, dict[str, str]]:
    if not output.exists():
        return {}

    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if "session_id" not in (reader.fieldnames or ()):
            raise ValueError(f"{output} has no 'session_id' column")
        return {
            row["session_id"]: {field: row.get(field, "") for field in fieldnames()}
            for row in reader
            if row.get("session_id")
        }


def fieldnames() -> tuple[str, ...]:
    return tuple(SessionAssetPaths.__dataclass_fields__)


def write_unique_rows(
    output: pathlib.Path,
    rows_by_session_id: dict[str, dict[str, str]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows_by_session_id.values())


def iter_completed_rows(
    session_id_to_source_path: dict[str, str],
    max_workers: int,
) -> Iterable[SessionAssetPaths]:
    if max_workers == 1:
        for session_id, source_path in session_id_to_source_path.items():
            yield get_session_asset_paths(session_id, source_path)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_session_id = {
            executor.submit(
                get_session_asset_paths,
                session_id,
                source_path,
            ): session_id
            for session_id, source_path in session_id_to_source_path.items()
        }
        for future in concurrent.futures.as_completed(future_to_session_id):
            session_id = future_to_session_id[future]
            try:
                yield future.result()
            except Exception as exc:
                logger.exception("Lookup failed unexpectedly for %s", session_id)
                yield SessionAssetPaths(
                    session_id=session_id,
                    source_path=session_id_to_source_path[session_id],
                    raw_data_lookup_error=f"{type(exc).__name__}: {exc}",
                    sorted_data_lookup_error=f"{type(exc).__name__}: {exc}",
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List raw-data and latest sorted-data S3 dirs for sessions."
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        help=(
            "Session IDs to process. If omitted, unique session IDs are discovered "
            "from --session-dir filenames."
        ),
    )
    parser.add_argument(
        "--session-dir",
        type=pathlib.Path,
        default=DEFAULT_SESSION_DIR,
        help=f"Directory to discover sessions from. Default: {DEFAULT_SESSION_DIR}",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help=(
            "CSV output path. "
            f"Default: ./{DEFAULT_OUTPUT_FILENAME} in the current directory."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of sessions to process, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent lookups. Use 1 for serial processing.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    output = args.output or pathlib.Path(DEFAULT_OUTPUT_FILENAME)
    rows_by_session_id = read_existing_rows(output)

    if args.sessions:
        session_id_to_source_path = {
            normalize_session_id(session_id): ""
            for session_id in args.sessions
        }
    else:
        session_id_to_source_path = discover_session_ids(args.session_dir)

    if args.limit is not None:
        session_id_to_source_path = dict(
            tuple(session_id_to_source_path.items())[: args.limit]
        )

    logger.info("Processing %d unique session(s)", len(session_id_to_source_path))
    for idx, row in enumerate(
        iter_completed_rows(session_id_to_source_path, args.max_workers),
        start=1,
    ):
        rows_by_session_id[row.session_id] = asdict(row)
        write_unique_rows(output, rows_by_session_id)
        logger.info(
            "Wrote %d/%d: %s",
            idx,
            len(session_id_to_source_path),
            row.session_id,
        )


if __name__ == "__main__":
    main()
