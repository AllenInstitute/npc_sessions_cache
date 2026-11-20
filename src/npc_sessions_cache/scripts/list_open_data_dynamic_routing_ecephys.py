from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import pathlib
import sys
from collections import deque
from collections.abc import Iterator

import boto3
import npc_lims
import npc_session
from botocore import UNSIGNED
from botocore.client import BaseClient
from botocore.config import Config

DEFAULT_S3_ROOT = "s3://aind-open-data"
DEFAULT_S3_PREFIX = "ecephys"
DEFAULT_CONTAINS = "ecephys"

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OpenDataDynamicRoutingEcephysFolder:
    dir_name: str
    s3_path: str
    subject_id: int
    training_db_sources: str


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Return bucket and key prefix from an s3:// URL."""
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URL, got {s3_url!r}")
    bucket_and_prefix = s3_url.removeprefix("s3://").strip("/")
    bucket, separator, prefix = bucket_and_prefix.partition("/")
    if not bucket or not separator and not bucket_and_prefix:
        raise ValueError(f"Expected a bucket in {s3_url!r}")
    return bucket, prefix.strip("/")


def join_s3_prefix(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def get_s3_client(signed: bool = False) -> BaseClient:
    if signed:
        return boto3.client("s3")
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def get_folder_name(prefix: str) -> str:
    return prefix.rstrip("/").rsplit("/", maxsplit=1)[-1]


def get_training_subject_sources(include_nsb: bool = True) -> dict[int, str]:
    sources_by_subject: dict[int, set[str]] = {}
    for nsb, source in ((False, "sam"), (True, "nsb")):
        if nsb and not include_nsb:
            continue
        for subject in npc_lims.get_subjects_from_training_db(nsb=nsb):
            sources_by_subject.setdefault(int(subject), set()).add(source)
    return {
        subject: ",".join(sorted(sources))
        for subject, sources in sources_by_subject.items()
    }


def iter_s3_folder_prefixes(
    client: BaseClient,
    bucket: str,
    prefix: str = "",
    recursive: bool = False,
    page_size: int = 1000,
) -> Iterator[str]:
    prefixes_to_list = deque([prefix.strip("/")])
    visited_prefixes: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")

    while prefixes_to_list:
        current_prefix = prefixes_to_list.popleft()
        if current_prefix in visited_prefixes:
            continue
        visited_prefixes.add(current_prefix)

        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=current_prefix,
            Delimiter="/",
            PaginationConfig={"PageSize": page_size},
        ):
            for common_prefix in page.get("CommonPrefixes", ()):
                folder_prefix = common_prefix["Prefix"]
                yield folder_prefix
                if recursive:
                    prefixes_to_list.append(folder_prefix)


def iter_matching_s3_folder_paths(
    client: BaseClient,
    bucket: str,
    prefix: str,
    contains: str,
    recursive: bool,
    page_size: int,
) -> Iterator[str]:
    contains = contains.casefold()
    for folder_prefix in iter_s3_folder_prefixes(
        client=client,
        bucket=bucket,
        prefix=prefix,
        recursive=recursive,
        page_size=page_size,
    ):
        dir_name = get_folder_name(folder_prefix)
        if contains and contains not in dir_name.casefold():
            continue
        yield f"s3://{bucket}/{folder_prefix.rstrip('/')}"


def extract_subject_id(s3_path: str) -> int | None:
    dir_name = get_folder_name(s3_path)
    return npc_session.extract_subject(dir_name) or npc_session.extract_subject(s3_path)


def get_dynamic_routing_ecephys_folders(
    s3_root: str,
    s3_prefix: str,
    contains: str,
    include_nsb: bool,
    signed: bool,
    recursive: bool,
    page_size: int,
    limit: int | None,
) -> tuple[OpenDataDynamicRoutingEcephysFolder, ...]:
    bucket, root_prefix = parse_s3_url(s3_root)
    prefix = join_s3_prefix(root_prefix, s3_prefix)
    client = get_s3_client(signed=signed)
    training_subjects = get_training_subject_sources(include_nsb=include_nsb)
    logger.info(
        "Loaded %d Dynamic Routing subject(s) from npc_lims training db",
        len(training_subjects),
    )

    rows: list[OpenDataDynamicRoutingEcephysFolder] = []
    scanned = 0
    with_subject = 0
    for s3_path in iter_matching_s3_folder_paths(
        client=client,
        bucket=bucket,
        prefix=prefix,
        contains=contains,
        recursive=recursive,
        page_size=page_size,
    ):
        scanned += 1
        if limit is not None and scanned > limit:
            break
        subject_id = extract_subject_id(s3_path)
        if subject_id is None:
            continue
        with_subject += 1
        if subject_id not in training_subjects:
            continue
        rows.append(
            OpenDataDynamicRoutingEcephysFolder(
                dir_name=get_folder_name(s3_path),
                s3_path=s3_path,
                subject_id=subject_id,
                training_db_sources=training_subjects[subject_id],
            )
        )

    logger.info(
        "Scanned %d folder(s); extracted subject IDs from %d; matched %d",
        scanned if limit is None else min(scanned, limit),
        with_subject,
        len(rows),
    )
    return tuple(rows)


def fieldnames() -> tuple[str, ...]:
    return tuple(OpenDataDynamicRoutingEcephysFolder.__dataclass_fields__)


def write_csv(
    path: pathlib.Path,
    rows: tuple[OpenDataDynamicRoutingEcephysFolder, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)


def print_tsv(rows: tuple[OpenDataDynamicRoutingEcephysFolder, ...]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames(), dialect="excel-tab")
    writer.writeheader()
    writer.writerows(dataclasses.asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List aind-open-data ecephys folders whose extracted subject ID is in "
            "the Dynamic Routing training database."
        )
    )
    parser.add_argument(
        "--s3-root",
        default=DEFAULT_S3_ROOT,
        help=f"S3 root to search. Default: {DEFAULT_S3_ROOT}",
    )
    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        help=(
            "S3 prefix under --s3-root to list. Default is 'ecephys' because AIND "
            "ecephys asset directories are top-level prefixes beginning with that."
        ),
    )
    parser.add_argument(
        "--contains",
        default=DEFAULT_CONTAINS,
        help=(
            "Case-insensitive substring required in each folder name. Use an empty "
            "string to disable this filter. Default: ecephys"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Descend into subfolders before applying --contains. This can be slow.",
    )
    parser.add_argument(
        "--no-nsb",
        dest="include_nsb",
        action="store_false",
        help="Exclude NSB subjects from npc_lims.get_subjects_from_training_db().",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Use configured AWS credentials instead of unsigned public S3 access.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional CSV output path. If omitted, rows are printed as TSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional limit on matching S3 folders to inspect, useful for "
            "smoke tests."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="S3 list_objects_v2 page size. Default: 1000",
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
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
    logging.getLogger("aiobotocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("s3fs").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    rows = get_dynamic_routing_ecephys_folders(
        s3_root=args.s3_root,
        s3_prefix=args.s3_prefix,
        contains=args.contains,
        include_nsb=args.include_nsb,
        signed=args.signed,
        recursive=args.recursive,
        page_size=args.page_size,
        limit=args.limit,
    )
    if args.output is not None:
        write_csv(args.output, rows)
        logger.info("Wrote %d row(s) to %s", len(rows), args.output)
    else:
        print_tsv(rows)


if __name__ == "__main__":
    main()
