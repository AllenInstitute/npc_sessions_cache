import dotenv
from npc_lims import tracked
from npc_session import *

from npc_sessions.sessions import *
from npc_sessions.trials import *
from npc_sessions.utils import *

_ = dotenv.load_dotenv(
    dotenv.find_dotenv(usecwd=True)
)  # take environment variables from .env

Session = DynamicRoutingSession  # temp alias for backwards compatibility
sessions = [DynamicRoutingSession(info.session) for info in tracked if info.is_uploaded]
"""Uploaded sessions, tracked in npc_lims via `tracked_sessions.yaml`"""