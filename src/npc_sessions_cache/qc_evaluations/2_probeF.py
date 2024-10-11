import functools

import npc_sessions_cache.qc_evaluations._probes

for var in npc_sessions_cache.qc_evaluations._probes:
    if callable(var):
        globals()[var.__name__] = functools.partial(var, probe_letter="F")
