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