import functools
import pathlib

import npc_session
from npc_sessions_cache.qc_evaluations._probes import *

for name in tuple(globals().keys()):
    func = globals()[name]
    if name.startswith('plot_') and callable(func):
        globals()[name] = functools.partial(func, probe_letter=npc_session.ProbeRecord(pathlib.Path(__file__).stem))
