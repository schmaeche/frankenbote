"""Frankenbote — a personal weekly news digest for Franconia, Bavaria and Germany."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("frankenbote")
except PackageNotFoundError:
    __version__ = "unknown"