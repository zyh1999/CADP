import sys
import os
from functools import partial
from .multiagentenv import MultiAgentEnv
from  .starcraft2 import StarCraft2Env
#from smac.env import MultiAgentEnv, StarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except:
    gfootball = False

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
