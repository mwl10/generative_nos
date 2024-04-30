"""
Utils
"""

import time
import jax
import json
import jax.numpy as jnp
import numpy as onp
from collections import OrderedDict
from scipy.spatial import KDTree
from collections import defaultdict
import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Point
from torch.utils.data import DataLoader
from jax.tree_util import tree_map
from torch.utils.data import Dataset
from torch.utils.data import default_collate
import numpy as onp
import inspect
import re

import numpy as np


def split_linear_string(input_string):
    # Define a regular expression pattern to match the desired format
    pattern = re.compile(r'([a-zA-Z]+)_([0-9]+)')

    # Use the pattern to search for matches in the input string
    match = pattern.match(input_string)

    if match:
        # Extract the matched groups (word and number)
        word = match.group(1)
        number = int(match.group(2))

        # Return the extracted values
        return word, number
    else:
        # Return None if no match is found
        return None

def split_conv_string(input_string):
    # Define a regular expression pattern to match the desired format
    pattern = re.compile(r'([a-zA-Z]+)_([0-9]+)_([0-9]+)_?([0-9]*)_([a-zA-Z]+)')

    # Use the pattern to search for matches in the input string
    match = pattern.match(input_string)

    if match:
        # Extract the matched groups (words and numbers)
        words = match.group(1)
        number1 = int(match.group(2))
        number2 = int(match.group(3))
        number3 = int(match.group(4)) if match.group(4) else None
        operation = match.group(5).upper()

        # Return the extracted values
        return words, number1, number2, number3, operation
    else:
        # Return None if no match is found
        return None

class fstr:
    def __init__(self, payload):
        self.payload = payload
    def __str__(self):
        vars = inspect.currentframe().f_back.f_globals.copy()
        vars.update(inspect.currentframe().f_back.f_locals)
        return self.payload.format(**vars)
    def __add__(self, another):
        if isinstance(another, fstr):
            self.payload += another.payload
        else:
            self.payload += another

