
from __future__ import annotations

import collections.abc
import dataclasses
import importlib
import importlib.metadata
import json
import logging
import pathlib
from collections.abc import Iterator, Mapping
import tempfile
import traceback
from typing import Callable, Iterable, Union

import npc_lims
import npc_session
import npc_sessions
import numpy as np
import pandas as pd
import upath
from typing_extensions import Self, TypeAlias

from npc_sessions_cache.utils import misc
import matplotlib.figure

logger = logging.getLogger(__name__)

DEFAULT_SESSION_QC_PATH = upath.UPath('s3://aind-scratch-data/dynamic-routing/qc')

Data: TypeAlias = Union[Mapping, matplotlib.figure.Figure, str]

@dataclasses.dataclass(frozen=True)
class QCElement:
    
    data: Mapping | matplotlib.figure.Figure | str
    session_id: str | npc_session.SessionRecord
    """data generated by the QC function"""
    module_name: str
    """name of python module that generated the QC"""
    function_name: str
    """name of function that generated the QC"""
    is_error: bool = False
    """whether the QC data is an error message"""
    
    def get_filename(self, stem_suffix: str = "", type_suffix: str | None = None) -> str:
        if not type_suffix:
            if isinstance(self.data, str):
                type_suffix = ".txt" if not self.is_error else ".error"
            elif isinstance(self.data, matplotlib.figure.Figure):
                type_suffix = ".png"
            elif isinstance(self.data, Mapping):
                type_suffix = ".json"
            else:
                raise ValueError(f"Unsupported data type: {type(self.data)}")
        if not type_suffix.startswith('.'):
            type_suffix = '.' + type_suffix
        if stem_suffix and not stem_suffix.startswith('_'):
            stem_suffix = '_' + stem_suffix
        return f"{self.session_id}{stem_suffix}{type_suffix}"
    
    def write(
        self,
        root_path: str | pathlib.Path | upath.UPath,
        stem_suffix: str = "",
        type_suffix: str| None = None,
        filename: str | None = None,
        **kwargs,
    ) -> upath.UPath:
        root_path = upath.UPath(root_path)
        if self.module_name not in root_path.parts:
            root_path /= self.module_name
        if self.function_name not in root_path.parts:
            root_path /= self.function_name
        root_path.mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = self.get_filename(stem_suffix=stem_suffix, type_suffix=type_suffix)
        path = root_path / filename
        logger.debug(f"Writing QC data to {path}")
        if isinstance(self.data, str):
            path.write_text(self.data, **kwargs)
        elif isinstance(self.data, matplotlib.figure.Figure):
            if not path.protocol:
                self.data.savefig(path, **kwargs)
            else: # savefig not compatible with cloud storage
                with tempfile.TemporaryDirectory() as tmpdir:
                    self.data.savefig((f := pathlib.Path(tmpdir) / path.name), **kwargs)
                    path.write_bytes(f.read_bytes())
        elif isinstance(self.data, Mapping):
            path.write_text(json.dumps(self.data))
        else:
            raise ValueError(f"Unsupported data type: {type(self.data)}")
        return path
    
class QCStore(collections.abc.Mapping):

    def __init__(
        self,
        module_name: str,
        function_name: str,
        root_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_QC_PATH,
        duplicate_path: str | pathlib.Path | upath.UPath | None = None,
        create: bool = True,
    ) -> None:
        self.module_name = module_name
        self.function_name = function_name
        self.path = upath.UPath(str(root_path)) / self.module_name / self.function_name
        logger.debug(f"{self.__class__.__name__} path set: {self.path}")
        if duplicate_path is not None:
            self.duplicate_path = upath.UPath(str(duplicate_path)) / self.module_name / self.function_name
        else:
            self.duplicate_path = None
        for path in self.all_paths:
            if create:
                path.mkdir(parents=True, exist_ok=True)
            elif not path.exists():
                raise FileNotFoundError(f"Path does not exist or is not accessible: {path}")

            if not path.protocol and not path.is_dir():
                raise ValueError(f"Expected record store path to be a directory: {path}")
        # setup internal cache
        self._cache: dict[npc_session.SessionRecord, tuple[upath.UPath, ...]] = {}
        # a list of missing elements avoids repeated slow checks on disk for records that are not present in the store
        self._missing: set[npc_session.SessionRecord] = set(
            self._normalize_key(p) for p in self.path.glob("*.error")
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module_name={self.module_name!r}, function_name={self.function_name!r}, root={self.path!r})"

    @property
    def all_paths(self) -> tuple[upath.UPath, ...]:
        return (self.path, self.duplicate_path) if self.duplicate_path else (self.path, )
    
    def get_record_glob(self, key: str | npc_session.SessionRecord) -> str:
        return f"{self._normalize_key(key)}*"

    def delete_data(self, key: str | npc_session.SessionRecord) -> None:
        key = self._normalize_key(key)
        for root_path in self.all_paths:
            for path in root_path.glob(self.get_record_glob(key)):
                path.unlink(missing_ok=True)
                logger.debug(f"Deleted {path.as_posix()}")
        self._cache.pop(key, None)
        self._missing.add(key)
        
    def write_data(
        self, 
        key: str | npc_session.SessionRecord, 
        data: Data | QCElement | Iterable[Data | QCElement],
        is_error: bool = False,
        ) -> None:
        key = self._normalize_key(key)
        if isinstance(data, str) or not isinstance(data, Iterable):
            data = [data]
        data = tuple(data)
        self.delete_data(key)
        new_paths = []
        for idx, element in enumerate(data):
            if not isinstance(element, QCElement):
                element = QCElement(data=element, session_id=key, module_name=self.module_name, function_name=self.function_name, is_error=is_error)
            else:
                assert element.module_name == self.module_name
                assert element.function_name == self.function_name
            for root_path in self.all_paths:
                new_path = element.write(root_path, stem_suffix=f"_{idx}" if len(data) > 1 else "")
            new_paths.append(new_path)
        self._cache[key] = tuple(new_paths)
        self._missing.discard(key)
    
    def is_errored(self, key: str | npc_session.SessionRecord) -> bool:
        if key in self:
            return any(element.suffix == ".error" for element in self[key])
        return False
    
    @staticmethod
    def _normalize_key(key: str) -> npc_session.SessionRecord:
        return npc_session.SessionRecord(key)

    def __getitem__(self, key: str | npc_session.SessionRecord) -> tuple[upath.UPath, ...]:
        key = self._normalize_key(key)
        if key in self._cache:
            logger.debug(f"Paths for {key} fetched from cache")
            return self._cache[key]
        if key in self._missing:
            logger.debug(f"{key} in 'missing' list: previously established that data does not exist on disk")
            raise KeyError(f"{key} not in store")
        paths = tuple(self.path.glob(self.get_record_glob(key)))
        if not paths:
            self._missing.add(key)
            logger.debug(f"Data for {key} does not exist on disk: added to 'missing' list")
            raise KeyError(f"{key} not in store")
        else:
            self._cache[key] = paths
            logger.debug(f"Added {key} to cache")
            return paths

    def __iter__(self) -> Iterator[str]:
        yield from iter(self._cache)
        yield from (
            self._normalize_key(path.stem)
            for path in self.path.glob("*")
            if self._normalize_key(path.stem) not in self._cache
        )

    def __len__(self):
        return len(tuple(iter(self)))

def normalize_function_name(name: str) -> str:
    if name.startswith("plot_"):
        return name[5:]
    return name

def get_qc_module_names() -> tuple[str, ...]:
    """list of names of py files in the plots directory
    
    >>> assert get_qc_module_names()
    """
    # get path to plots directory
    plots_path = pathlib.Path(__file__).parent.parent / 'plots'
    return tuple(
        path.stem
        for path in plots_path.glob('*.py')
        if path.stem not in ('__init__', 'utils')
    )
    
def get_qc_functions(module_name: str | None = None) -> dict[tuple[str, str], Callable]:
    """returns {(module_name, function_name): function}
    >>> get_qc_functions()
    """
    if module_name:
        modules: tuple[str, ...] = (module_name, )
    else:
        modules = get_qc_module_names()
    functions = {}
    for m in modules:
        module = importlib.import_module(f"npc_sessions_cache.plots.{m}")
        for name in dir(module):
            if not name.startswith("plot_"):
                continue
            functions[(m, normalize_function_name(name))] = getattr(module, name)
    return functions
    
def write_session_qc(
    session_id: str | npc_session.SessionRecord,
    store_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_QC_PATH,
    duplicate_path: str | pathlib.Path | upath.UPath | None = None,
    skip_existing: bool = True,
    skip_previously_failed: bool = True,
    session: npc_sessions.DynamicRoutingSession | None = None,
) -> None:
    if session is None:
        session = npc_sessions.DynamicRoutingSession(session_id)
    for (module_name, function_name), function  in get_qc_functions().items():
        store = QCStore(module_name, function_name, root_path=store_path, duplicate_path=duplicate_path)
        key = store._normalize_key(session_id)
        if skip_existing and key in store:
            logger.info(f"Skipping {key} - qc data already exists")
            continue
        if skip_previously_failed and store.is_errored(key):
            logger.info(f"Skipping {key} - previously failed to write qc data")
            continue
        logger.info(f"Running {module_name}.plot_{function_name} for {session_id}")
        try:
            data = function(session)
            is_error = False
        except Exception:
            data = traceback.format_exc()
            is_error = True
        if data is None:
            logger.warning(f"{module_name}.plot_{function_name} returned None - update it to return one or more plt.Fig, dict or str")
            continue
        store.write_data(key=key, data=data, is_error=is_error)


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )