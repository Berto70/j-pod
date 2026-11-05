"""Asteroid discovery providers."""

from .mpchecker import query_mpc_identification
from .skybot import query_skybot

__all__ = ["query_mpc_identification", "query_skybot"]
