from .custom_linear import *

class State:
    _SEED = 15

    def get_seed():
        return State._SEED

    def set_seed(v):
        State._SEED = v

