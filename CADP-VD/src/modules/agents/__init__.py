REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
from .atten_rnn_agent import ATTRNNAgent
REGISTRY["att_rnn"] = ATTRNNAgent