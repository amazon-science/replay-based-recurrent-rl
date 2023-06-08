from __future__ import print_function, division
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DictToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_state(m):
    """
    This code returns model states
    """
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state


def load_model_states(path):
    """
    Load previously learned model
    """
    checkpoint = torch.load(path, map_location="cpu")
    m_states = checkpoint["model_states"]
    m_params = checkpoint["args"]

    return m_states, DictToObj(**m_params)
