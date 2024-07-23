"""
>>> import tempfile
>>> dummy_record = Record(project='DynamicRouting', session_id='668755_2023-08-29', date='2023-08-29', time='13:11:16', subject=668755, subject_age='P203D', subject_sex='M', subject_genotype='wt/wt', rig='NP3', experimenters=['Corbett Bennett'], notes=None, issues=[], epochs=['RFMapping', 'OptoTagging', 'Spontaneous', 'SpontaneousRewards', 'DynamicRouting1', 'SpontaneousRewards', 'OptoTagging'], allen_path='//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_668755_20230829', cloud_path='s3://aind-ephys-data/ecephys_668755_2023-08-29_13-11-16', task_version='stage 5 ori AMN moving', ephys_day=2, behavior_day=2, is_ephys=True, is_sync=True, is_video=True, is_templeton=False, is_annotated=True, is_hab=False, is_task=True, is_spontaneous=True, is_spontaneous_rewards=True, is_rf_mapping=True, is_optotagging=False, is_optotagging_control=True, is_opto_perturbation=False, is_opto_perturbation_control=False, is_injection_perturbation=False, is_timing_issues=False, is_invalid_times=False, is_production=True, is_naive=False, is_context_naive=False, probe_letters_available='ABCDEF', perturbation_areas=None, areas_hit=['ACAd', 'AMv', 'CP', 'EPd', 'HY', 'LHA', 'LSc', 'LSr', 'MOp', 'MOs', 'MS', 'NDB', 'OLF', 'ORBl', 'ORBvl', 'PAL', 'PR', 'PVH', 'PVHd', 'RT', 'SF', 'SI', 'STR', 'TH', 'VAL', 'VM', 'VPL'], n_passing_blocks=3, task_duration=3633.41261, intramodal_dprime_vis=2.4083675609776805, intramodal_dprime_aud=2.2121366948636805, intermodal_dprime_vis_blocks=[1.9145058250555569, 0.04842155051939967, 1.1183427509671398], intermodal_dprime_aud_blocks=[2.025975796915895, 1.1099094446952567, 1.9805176374313975])

>>> test_path = pathlib.Path(tempfile.TemporaryDirectory().name)
>>> test_store = RecordStore(test_path)
>>> key = dummy_record.session_id
>>> test_store[key] = dummy_record
>>> assert len(test_store) == 1
>>> assert len(tuple(test_path.iterdir())) == 1
>>> # test getting values from cache + disk by adding a record on disk
>>> _ = (path := test_store.get_record_path(key)).with_stem(nkey := key.replace('-29', '-30')).write_bytes(path.read_bytes())
>>> assert len(test_store) == 2
>>> assert test_store[nkey] == test_store[key] 
>>> df = test_store.to_pandas()
>>> del test_store[dummy_record.session_id]
>>> assert len(test_store) == 1
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import logging
import pathlib
from collections.abc import Iterator, Mapping
import traceback

import npc_lims
import npc_session
import npc_sessions
import numpy as np
import pandas as pd
import upath
from typing_extensions import Self

logger = logging.getLogger(__name__)

DEFAULT_SESSION_METADATA_PATH = upath.UPath('s3://aind-scratch-data/dynamic-routing/session_metadata')

@dataclasses.dataclass#(frozen=True, unsafe_hash=True)
class Record:
    """A row in the sessions table"""

    # required - should be available for all sessions ------------------ #
    project: str
    session_id: npc_session.SessionRecord
    date: npc_session.DateRecord
    time: npc_session.TimeRecord
    subject: npc_session.SubjectRecord
    subject_age: int
    subject_sex: str
    subject_genotype: str
    implant: str | None = None
    # dye: str | None = None
    rig: str
    experimenters: list[str] | None
    notes: str | None
    issues: list[str]
    epochs: list[str]

    allen_path: str
    cloud_path: str | None

    task_version: str | None
    ephys_day: int | None
    behavior_day: int | None

    is_ephys: bool
    is_sync: bool
    is_video: bool
    is_templeton: bool
    is_annotated: bool
    is_hab: bool
    is_task: bool
    is_spontaneous: bool
    is_spontaneous_rewards: bool
    is_rf_mapping: bool
    is_optotagging: bool
    is_optotagging_control: bool
    is_opto_perturbation: bool
    is_opto_perturbation_control: bool
    is_injection_perturbation: bool

    is_timing_issues: bool
    is_invalid_times: bool
    is_production: bool

    is_naive: bool
    is_context_naive: bool # better would be `days_of_context_training`

    # currently not possible ------------------------------------------- #
    # is_injection_perturbation_control: bool #! injection metadata not in cloud, Vayle needs to update
    # is_duragel: bool | None = None
    # virus_name: str | None = None
    # virus_area: str | None = None

    # issues dependant - may be none if the session has issues --------- #
    probe_letters_available: str | None = None
    perturbation_areas: list[str] | None = None
    areas_hit: list[str] | None = None

    # behavior stuff --------------------------------------------------- #
    n_passing_blocks: int | None = None
    task_duration: float | None = None
    intramodal_dprime_vis: float | None = None
    intramodal_dprime_aud: float | None = None
    intermodal_dprime_vis_blocks: list[float | None] | None = None
    intermodal_dprime_aud_blocks: list[float | None] | None = None

    def to_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self),
            indent=4,
        )

    @classmethod
    def from_json(cls, json_content: str | bytes | Mapping) -> Self:
        if not isinstance(json_content, Mapping):
            json_content = json.loads(json_content)
        assert isinstance(json_content, Mapping)
        return cls(**json_content)

class RecordStore(collections.abc.MutableMapping):

    def __init__(
        self, 
        path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_METADATA_PATH / 'records',
        create: bool = True,
    ) -> None:
        self.path = upath.UPath(str(path))
        logger.debug(f"{self.__class__.__name__} path set: {self.path}")
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise FileNotFoundError(f"Path does not exist or is not accessible: {self.path}")

        if not self.path.protocol and not self.path.is_dir():
            raise ValueError(f"Expected record store path to be a directory: {self.path}")

        # setup internal cache for Record objects
        self._cache: dict[npc_session.SessionRecord, Record] = {}
        # a list of missing Records avoids repeated slow checks on disk for records that are not present in the store
        self._missing: set[npc_session.SessionRecord] = set()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path.as_posix()})"

    def get_record_path(self, key: str | npc_session.SessionRecord) -> upath.UPath:
        return self.path / f"{npc_session.SessionRecord(key)}.json"

    def read_record(self, key: str | npc_session.SessionRecord) -> Record:
        path = self.get_record_path(key)
        logger.debug(f"Reading record from {path.as_posix()}")
        return Record.from_json(path.read_bytes())

    def write_record(self, record: Record) -> None:
        path = self.get_record_path(record.session_id)
        if path.exists():
            logger.debug(f"Overwriting record at {path.as_posix()}")
        else:
            logger.debug(f"writing new record to {path.as_posix()}")
        path.write_text(record.to_json())

    @staticmethod
    def _normalize_key(key: str) -> npc_session.SessionRecord:
        return npc_session.SessionRecord(key)

    def __getitem__(self, key: str | npc_session.SessionRecord) -> Record:
        key = self._normalize_key(key)
        if key in self._cache:
            logger.debug(f"Record for {key} fetched from cache")
            return self._cache[key]
        if key in self._missing:
            logger.debug(f"{key} in 'missing' list: previously established that record does not exist on disk")
            raise KeyError(f"{key} not in RecordStore")
        try:
            record = self.read_record(key)
        except FileNotFoundError:
            self._missing.add(key)
            logger.debug(f"Record for {key} does not exist on disk: added to 'missing' list")
            raise KeyError(f"{key} not in RecordStore")
        else:
            self._cache[key] = record
            logger.debug(f"Added {key} to cache")
            return record

    def __setitem__(self, key: str | npc_session.SessionRecord, value: Record) -> None:
        key = self._normalize_key(key)
        self.write_record(value)
        self._cache[key] = value
        logger.debug(f"Added {key} to cache")
        self._missing.discard(key)
        logger.debug(f"Discarded {key} from 'missing' list (if present)")

    def __delitem__(self, key: str | npc_session.SessionRecord) -> None:
        key = self._normalize_key(key)
        del self._cache[key]
        logger.debug(f"Deleted {key} from cache")
        (path := self.get_record_path(key)).unlink(missing_ok=True)
        logger.debug(f"Deleted {path.as_posix()}")
        self._missing.add(key)
        logger.debug(f"Added {key} to 'missing' list after deletion of record")

    def __iter__(self) -> Iterator[str]:
        yield from iter(self._cache)
        yield from (
            self._normalize_key(path.stem)
            for path in self.path.glob("*.json")
            if self._normalize_key(path.stem) not in self._cache
        )

    def __len__(self):
        return len(tuple(iter(self)))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([dataclasses.asdict(v) for v in self.values()])

    def from_pandas(self, df: pd.DataFrame) -> None:
        for record in df.itertuples(index=False):
            self[record.session_id] = Record(**record._asdict())
            
def get_session_record(session_id: str | npc_session.SessionRecord, session: npc_sessions.DynamicRoutingSession | None = None) -> Record:
    if session is None:
        session = npc_sessions.Session(session_id)
    assert session is not None

    if session.is_task:
        trials = session.trials[:]
        performance = session.performance[:]
    epochs = session.epochs[:]
    epochs = epochs.stim_name.to_list()
    def is_in_epochs(name):
        return any(name.strip('_').lower() == epoch.lower() for epoch in epochs)
    if session.is_annotated:
        units = session.units[:]

    def get_intermodal_dprime(modality: str) -> list[float | None]:
        return [
            v if not np.isnan(v) else None
            for v in performance.query(f"rewarded_modality == '{modality}'").cross_modal_dprime.to_numpy()
        ]

    return Record(
        project="DynamicRouting" if not session.is_templeton else "Templeton",
        session_id=session.id,
        date=session.id.date,
        time=npc_session.TimeRecord(session.session_start_time.time()),
        subject=session.id.subject,
        subject_age=session.subject.age,
        subject_sex=session.subject.sex,
        subject_genotype=session.subject.genotype,
        implant=session.probe_insertion_info['shield']['name'],
        rig=session.rig,
        experimenters=session.experimenter,
        notes=session.notes,
        issues=session.info.issues,
        epochs=epochs,
        allen_path=session.info.allen_path.as_posix(),
        cloud_path=session.info.cloud_path.as_posix(),
        task_version=session.task_version if session.is_task else None,
        ephys_day=session.info.experiment_day if session.is_ephys else None,
        behavior_day=session.info.behavior_day if session.is_task else None,
        is_ephys=session.is_ephys,
        is_sync=session.is_sync,
        is_video=session.is_video,
        is_templeton=session.is_templeton,
        is_annotated=session.is_annotated,
        is_hab=session.is_hab,
        is_task=session.is_task,
        is_spontaneous=is_in_epochs('Spontaneous'),
        is_spontaneous_rewards=is_in_epochs('SpontaneousRewards'),
        is_rf_mapping=is_in_epochs('RFMapping'),
        is_optotagging="optotagging" in session.keywords,
        is_optotagging_control="optotagging_control" in session.keywords,
        is_opto_perturbation=(is_opto_task := "opto_perturbation" in session.keywords),
        is_opto_perturbation_control="opto_perturbation_control" in session.keywords,
        is_injection_perturbation=session.info.session_kwargs.get('is_injection_perturbation', False),
        # is_injection_perturbation_control=session.info.session_kwargs.get('is_injection_perturbation_control', False),
        is_timing_issues="timing_issues" in epochs.tags.explode().unique(),
        is_invalid_times="invalid_times" in epochs.tags.explode().unique(),
        is_production=session.info.session_kwargs.get('is_production', True),
        is_naive=session.info.session_kwargs.get('is_naive', False),
        is_context_naive=session.info.session_kwargs.get('is_context_naive', False) or session.is_templeton,
        probe_letters_available="".join(session.probe_letters_to_use),
        perturbation_areas=sorted(trials.opto_label.unique()) if is_opto_task else None,
        areas_hit=sorted(units.structure.unique()) if session.is_annotated else None, # add injection areas
        n_passing_blocks=len(performance.query("cross_modal_dprime >= 1.5")) if session.is_task else None,
        task_duration=trials.stop_time.max() - trials.start_time.min() if session.is_task else None,
        intramodal_dprime_vis=performance.vis_intra_dprime.mean() if session.is_task else None,
        intramodal_dprime_aud=performance.aud_intra_dprime.mean() if session.is_task else None,
        intermodal_dprime_vis_blocks=get_intermodal_dprime('vis') if session.is_task else None,
        intermodal_dprime_aud_blocks=get_intermodal_dprime('aud') if session.is_task else None,
    )

def write_session_record(
    session_id: str | npc_session.SessionRecord,
    store_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_METADATA_PATH / 'records',
    skip_existing: bool = True,
    skip_previously_failed: bool = True,
) -> None:
    store = RecordStore(store_path)
    error_path = store.path / 'errors'
    error_path.mkdir(parents=True, exist_ok=True)
    store._missing.update({store._normalize_key(p.stem) for p in error_path.glob('*.txt')})
    key = store._normalize_key(session_id)
    print(key)
    if skip_existing and key in store:
        logger.info(f"Skipping {key} - record already exists in store")
        return
    error = error_path / f"{key}.txt"
    if skip_previously_failed and error.exists():
        logger.info(f"Skipping {key} - previously failed to get record")
        return
    if key in store:
        logger.info(f"Clearing exsiting record for {key} before fetching new record (in case it errors)")
        del store[key]
    try:
        store[key] = get_session_record(session_id)
    except Exception:
        error.write_text(traceback.format_exc())
    else:
        error.unlink(missing_ok=True)
        logger.info(f"Removed {error.as_posix()} after successful record write")

def write_session_table_from_records(
    store_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_METADATA_PATH / 'records',
    table_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_METADATA_PATH / 'sessions.parquet',
    update_existing: bool = False,
) -> None:
    store = RecordStore(store_path)
    store_path = upath.UPath(store_path)
    table_path = upath.UPath(table_path)
    existing_df = None
    if table_path.exists():
        if not update_existing:
            table_path.unlink()
        else:
            existing_df: pd.DataFrame = get_session_table(table_path)
    df = store.to_pandas()
    if existing_df:
        df = existing_df.update(df)
    # write to disk
    getattr(df, f"to_{table_path.suffix.strip('.')}")(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Failed to write session table to {table_path}")

def get_session_table(
    table_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_METADATA_PATH / 'sessions.parquet',
) -> pd.DataFrame:
    table_path = upath.UPath(table_path)
    return getattr(pd, f"read_{table_path.suffix.strip('.')}")(table_path)

def check_session_table()

if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
