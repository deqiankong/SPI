import os
from typing import Callable, Iterable, List, Dict, Union, Tuple
import json
import itertools
from pathlib import Path

import numpy as np

from src.data_utils.wow import (
    TOKEN_NOCHOSEN,
    TOKEN_KNOWLEDGE,
)

ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<speaker1>', '<speaker2>', TOKEN_KNOWLEDGE]}


def add_special_tokens(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)