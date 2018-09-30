import gym
from abc import ABC,abstractmethod


class GAILEnv(gym.Env,ABC):

    """
    Feature mapper is a function from states and context to features
    A single environment can have different mappers like
    direct mapper ---- state -> feature
    partial mapper ----- part of state space  -> feature
    complex mapper ----- derive complex features using state dependent on context
    """
    def __init__(self,feature_mapper=None):
        super().__init__(self)
        if(feature_mapper):
            self.set_feature_mapper(feature_mapper)
        else:
            self.set_feature_mapper(self.direct_mapper)

    def extract_features(self,state,context):
        """converts state s to feature z, given context c"""
        return self.feature_mapper(state,context)

    def set_feature_mapper(self,feature_mapper):
        self.feature_mapper = feature_mapper

    def direct_mapper(self,state,context):
        return state

    @abstractmethod
    def play_expert(self):
        pass

