from dataclasses import dataclass
from contextlib import suppress, nullcontext

import torch
import torch.nn as nn


@dataclass(eq=False)  # Allows for the hash function to propagate to nn.Module.__hash__
class BaseModel(nn.Module):
    """
    Base class for PyTorch networks, expanding nn.Module.

    Initialization parameters will be automatically added to the state_dict and checked when loading a checkpoint

    Required:
        :method forward: Model forward pass (PyTorch standard)

    Helpers:
        :classmethod from_ckpt: Restores a model from a checkpoint file. Can make use of overloaded state_dicts.
        :staticmethod add_parser_args: Add any additional arguments required for command line parsing
        :method state_dict: PyTorch state_dict + additional parameters needed for initialization
        :method load_state_dict: Check init parameters match the loaded state_dict and PyTorch load

    """

    def __post_init__(self):
        """Initialize network and nn.Module."""
        super().__init__()

    def __hash__(self):
        return super().__hash__()

    @classmethod
    def from_ckpt(cls, ckpt_file, key=None, strict=True):
        """
        Create network from a saved checkpoint.

        :param ckpt_file: File containing saved checkpoint.
        :param key: Function of one argument used to extract the network state_dict (same as built-in "sort" key)
        :param strict: Strictly enforce matching keys between the checkpoint and the model.
        :return: Restored class
        """
        ckpt_dict = torch.load(ckpt_file)
        ckpt_dict = key(ckpt_dict) if key else ckpt_dict

        manager = nullcontext() if strict else suppress(KeyError)
        with manager:
            kwargs = {k: ckpt_dict[k] for k in cls.__dataclass_fields__}

        model = cls(**kwargs)
        model.load_state_dict(ckpt_dict, strict=strict)
        return model

    @classmethod
    def from_dict(cls, in_dict):
        """Instantiate class from dict. Ignores any unrecognized arguments."""
        new_dict = {k: v for k, v in in_dict.items() if k in cls.__dataclass_fields__}
        return cls(**new_dict)

    @staticmethod
    def add_parser_args(parser):
        """Add required parameters for parsing."""
        raise NotImplementedError

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        for k in self.__dataclass_fields__:
            state_dict[k] = getattr(self, k)
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict.copy()
        manager = nullcontext() if strict else suppress(KeyError)

        for k in self.__dataclass_fields__:
            with manager:
                v = state_dict.pop(k)
                assert self.__dict__[k] == v, f'Parameter "{k}" mismatch. ({self.__dict__[k]} vs. {v})'

        super().load_state_dict(state_dict, strict=strict)

    def forward(self, *args, **kwargs):
        """Network forward pass."""
        raise NotImplementedError
