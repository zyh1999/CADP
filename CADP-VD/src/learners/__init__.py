from .q_learner import QLearner
from .qplex_learner_teacher import DMAQ_qattenLearner as QPlexLearner_teacher
from .q_learner_teacher import  QLearner as QLearner_teacher
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_teacher"] = QLearner_teacher
REGISTRY["qplex_learner_teacher"] = QPlexLearner_teacher
