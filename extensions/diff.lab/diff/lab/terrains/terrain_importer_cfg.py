from __future__ import annotations
from omni.isaac.lab.utils import configclass

from .terrain_importer import TerrainImporter
from omni.isaac.lab.terrains import TerrainImporterCfg as terrain_importer_cfg

@configclass
class TerrainImporterCfg(terrain_importer_cfg):
    """Configuration for the terrain manager."""

    class_type: type = TerrainImporter
