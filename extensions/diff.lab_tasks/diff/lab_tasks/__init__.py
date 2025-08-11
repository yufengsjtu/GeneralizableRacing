# Copyright (c) 2022-2024, The Octi Lab Project Developers.
# All rights reserved.

"""Package containing task implementations for various robotic environments."""

import os
import toml

# Conveniences to other module directories via relative paths
DIFFLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

DIFFLAB_TASKS_METADATA = toml.load(os.path.join(DIFFLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = DIFFLAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

from omni.isaac.lab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)