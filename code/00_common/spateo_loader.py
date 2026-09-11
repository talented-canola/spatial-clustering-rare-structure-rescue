"""Surgical loader: import spateo's clustering leaf modules WITHOUT triggering
spateo/__init__.py (which cascades into tdr/tools/plotting and their heavy deps:
vtk, pyvista, geopandas, tensorflow, etc.).

Usage: import spateo_loader; then use spateo_loader.utils / .find_neighbors / .leiden
"""
import sys
import types

# --- matplotlib compatibility shim: spateo 1.1.0 targets matplotlib<=3.5.3, ---
# --- which had plt.register_cmap. Newer matplotlib (>=3.8) removed it.        ---
import matplotlib
import matplotlib.pyplot as plt

if not hasattr(plt, "register_cmap"):
    def _register_cmap(name=None, cmap=None, **kwargs):
        matplotlib.colormaps.register(cmap, name=name, force=True)
    plt.register_cmap = _register_cmap

REPO = r"F:/BGI/task3/spateo-release-main"


def _pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


# Register lightweight parent packages (skip their real __init__.py)
_pkg("spateo", REPO + "/spateo")
_pkg("spateo.external", REPO + "/spateo/external")
_pkg("spateo.tools", REPO + "/spateo/tools")
_pkg("spateo.tools.cluster", REPO + "/spateo/tools/cluster")

# Import the leaf modules needed for clustering (light deps only)
from spateo.tools.cluster import utils
from spateo.tools.cluster import leiden
from spateo.tools import find_neighbors

__all__ = ["utils", "leiden", "find_neighbors"]
