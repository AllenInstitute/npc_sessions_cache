
from __future__ import annotations

import collections.abc
import dataclasses
import importlib
import importlib.metadata
import inspect
import json
import logging
import pathlib
import tempfile
import traceback
from collections.abc import Iterable, Iterator, Mapping
from types import FunctionType
from typing import Callable, Union
import typing

import matplotlib.figure
import npc_session
import npc_sessions
import upath
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

DEFAULT_SESSION_QC_PATH = upath.UPath('s3://aind-scratch-data/dynamic-routing/qc')

Data: TypeAlias = Union[Mapping, matplotlib.figure.Figure, str]

@dataclasses.dataclass(frozen=True)
class QCElement:

    data: Mapping | matplotlib.figure.Figure | str | upath.UPath
    session_id: str | npc_session.SessionRecord
    """data generated by the QC function"""
    module_name: str
    """name of python module that generated the QC"""
    function_name: str
    """name of function that generated the QC"""
    
    def get_filename(self, stem_suffix: str = "", type_suffix: str | None = None) -> str:
        if not type_suffix:
            if isinstance(self.data, str):
                type_suffix = ".txt" if not self.data.startswith("Traceback") else ".error"
            elif isinstance(self.data, matplotlib.figure.Figure):
                type_suffix = ".png"
            elif isinstance(self.data, Mapping):
                type_suffix = ".json"
            elif isinstance(self.data, upath.UPath):
                type_suffix = ".link"
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
        elif isinstance(self.data, upath.UPath):
            path.write_text(self.data.as_posix(), **kwargs)
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
        create: bool = True,
    ) -> None:
        self.path = upath.UPath(str(root_path)) / module_name / function_name
        logger.debug(f"{self.__class__.__name__} path set: {self.path}")
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise FileNotFoundError(f"Path does not exist or is not accessible: {self.path}")

        if not self.path.protocol and not self.path.is_dir():
            raise ValueError(f"Expected record store path to be a directory: {self.path}")
        self.module_name = module_name
        self.function_name = function_name
        # setup internal cache
        self._cache: dict[str, tuple[upath.UPath, ...]] = {}
        # a list of missing elements avoids repeated slow checks on disk for records that are not present in the store
        self._missing: set[str] = set()
        self._errored: set[str] = {
            self.normalize_key(p.stem) for p in self.path.glob("*.error")
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module_name={self.module_name!r}, function_name={self.function_name!r}, root={self.path!r})"

    def get_record_glob(self, key: str | npc_session.SessionRecord) -> str:
        return f"{self.normalize_key(key)}*"

    def delete_data(self, key: str | npc_session.SessionRecord) -> None:
        key = self.normalize_key(key)
        for path in self.path.glob(self.get_record_glob(key)):
            path.unlink(missing_ok=True)
            logger.debug(f"Deleted {path.as_posix()}")
        self._cache.pop(key, None)
        self._missing.add(key)

    def write_data(
        self,
        key: str | npc_session.SessionRecord,
        data: Data | QCElement | Iterable[Data | QCElement],
        ) -> None:
        key = self.normalize_key(key)
        if isinstance(data, (str, Mapping)) or not isinstance(data, Iterable):
            data = [data]
        data = tuple(data)
        self.delete_data(key)
        paths = []
        for idx, element in enumerate(data):
            if not isinstance(element, QCElement):
                element = QCElement(data=element, session_id=key, module_name=self.module_name, function_name=self.function_name)
            else:
                assert element.module_name == self.module_name
                assert element.function_name == self.function_name
            path = element.write(self.path, stem_suffix=f"_{idx}" if len(data) > 1 else "")
            paths.append(path)
        self._cache[key] = tuple(paths)
        self._missing.discard(key)
        self._errored.discard(key)

    def is_errored(self, key: str | npc_session.SessionRecord) -> bool:
        if key in self:
            return any(element.suffix == ".error" for element in self[key])
        if key in self._errored:
            return True
        if (self.path / f"{self.normalize_key(key)}.error").exists():
            self._errored.add(key)
            return True
        return False

    @staticmethod
    def normalize_key(key: str) -> str:
        return str(npc_session.SessionRecord(key))

    def __getitem__(self, key: str | npc_session.SessionRecord) -> tuple[upath.UPath, ...]:
        key = self.normalize_key(key)
        if key in self._cache:
            logger.debug(f"Paths for {key} fetched from cache")
            return self._cache[key]
        if key in self._missing:
            logger.debug(f"{key} in 'missing' list: previously established that data does not exist on disk")
            raise KeyError(f"{key} qc data not found")
        if key in self._errored:
            logger.debug(f"{key} is in 'errored' list: previously established that data is an error")
            raise KeyError(f"{key} qc data not available - previously errored")
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
            self.normalize_key(path.stem)
            for path in self.path.glob("*")
            if self.normalize_key(path.stem) not in self._cache
            and path.suffix != ".error"
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
    plots_path = pathlib.Path(__file__).parent.parent / 'qc_evaluations'
    return tuple(
        path.stem
        for path in plots_path.glob('*.py')
        if path.stem not in ('__init__', 'utils', 'plot_utils')
    )

def get_qc_functions(module_name: str | None = None) -> dict[tuple[str, str], FunctionType]:
    """returns {(module_name, function_name): function}
    >>> get_qc_functions()
    """
    if module_name:
        modules: tuple[str, ...] = (module_name, )
    else:
        modules = get_qc_module_names()
    functions = {}
    for m in modules:
        module = importlib.import_module(f"npc_sessions_cache.qc_evaluations.{m}")
        for name in dir(module):
            if not name.startswith("plot_"):
                continue
            if "utils" in name.split("_"):
                continue
            callable_obj = getattr(module, name)
            if not inspect.isfunction(callable_obj):
                continue
            functions[(m, normalize_function_name(name))] = callable_obj
    return functions

def write_output_from_single_function(
    session_id: str | npc_session.SessionRecord,
    function: Callable[npc_sessions.DynamicRoutingSession, Data],
    function_name: str,
    module_name: str = "other",
    store_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_QC_PATH,
    skip_existing: bool = True,
    skip_previously_failed: bool = True,
    session: npc_sessions.DynamicRoutingSession | None = None,
) -> None:
    if session is None:
        session = npc_sessions.DynamicRoutingSession(session_id)
    function_name = normalize_function_name(function_name)
    store = QCStore(module_name, function_name, root_path=store_path)
    key = store.normalize_key(session_id)
    if skip_existing and key in store:
        logger.info(f"Skipping {module_name}.{function_name} for {key} - data already exists")
        return
    if skip_previously_failed and store.is_errored(key):
        logger.info(f"Skipping {module_name}.{function_name} for {key} - previously failed to write data")
        return
    logger.info(f"Running {module_name}.{function_name} for {session_id}")
    try:
        data = function(session)
    except Exception:
        data = traceback.format_exc()
    if data is None:
        logger.warning(f"{module_name}.{function_name} returned None - update it to return one or more of {typing.get_args(Data)}") # type: ignore
        return
    store.write_data(key, data)

def write_instructions_for_qc_item(
    function: Callable[npc_sessions.DynamicRoutingSession, Data],
    function_name: str,
    module_name: str,
    store_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_QC_PATH,
    ) -> None:
    function_name = normalize_function_name(function_name)
    store = QCStore(module_name, function_name, root_path=store_path)
    if (doc := inspect.getdoc(function)):
        # write general interpretation docs for the function if they exist
        if (path := store.path.parent / 'docs.json').exists():
            docs = json.loads(path.read_text())
        else:
            docs = {}
        docs[function_name] = doc
        logger.debug(f"Writing docs for {module_name}.{function_name} to {path}")
        path.write_text(json.dumps(docs, indent=4))
    
    module = inspect.getmodule(function)
    if (instructions := getattr(module, 'instructions', {})):
        # write specific instructions for the function if they exist
        if (path := store.path.parent / 'instructions.json').exists():
            instr = json.loads(path.read_text())
        else:
            instr = {}
        instr[function_name] = instructions.get(function, "")
        logger.debug(f"Writing instructions for {module_name}.{function_name} to {path}")
        path.write_text(json.dumps(instr, indent=4))
        
    
def write_session_qc(
    session_id: str | npc_session.SessionRecord,
    store_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_QC_PATH,
    skip_existing: bool = True,
    skip_previously_failed: bool = True,
    session: npc_sessions.DynamicRoutingSession | None = None,
) -> None:
    if session is None:
        session = npc_sessions.DynamicRoutingSession(session_id)
    for (module_name, function_name), function in get_qc_functions().items():
        write_output_from_single_function(
            session_id,
            function=function,
            function_name=normalize_function_name(function_name),
            module_name=module_name,
            store_path=store_path,
            skip_existing=skip_existing,
            skip_previously_failed=skip_previously_failed,
            session=session,
        )
        write_instructions_for_qc_item(
            function=function,
            function_name=normalize_function_name(function_name),
            module_name=module_name,
            store_path=store_path,
        )
        
def copy_current_qc_data(
    session_id: str | npc_session.SessionRecord,
    output_path: str | pathlib.Path | upath.UPath,
    store_path: str | pathlib.Path | upath.UPath = DEFAULT_SESSION_QC_PATH,
    function_name_filter: str | None = None,
) -> None:
    output_path = upath.UPath(output_path)
    store_path = upath.UPath(store_path)
    key = QCStore.normalize_key(session_id)
    for path in store_path.rglob(f"*{function_name_filter or ''}/*{key}*"):
        new_path = output_path / path.relative_to(store_path)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_bytes(path.read_bytes())

if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
