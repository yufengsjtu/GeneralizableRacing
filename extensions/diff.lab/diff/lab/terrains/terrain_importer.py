from omni.isaac.lab.terrains import TerrainImporter as terrain_importer
from .terrain_generator import TerrainGenerator
from omni.isaac.lab.terrains import TerrainImporterCfg
import omni.isaac.lab.sim as sim_utils
import torch
import numpy as np
class TerrainImporter(terrain_importer):
    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create a dict of meshes
        self.meshes = dict()
        self.warp_meshes = dict()
        self.env_origins = None
        self.terrain_origins = None
        # private variables
        self._terrain_flat_patches = dict()

        #
        self.extras = {}

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = TerrainGenerator(cfg=self.cfg.terrain_generator, device=self.device)
            # obtain extra information
            if hasattr(terrain_generator, "extras"):
                if terrain_generator.cfg.curriculum:
                    gate_pose = torch.tensor(np.stack(terrain_generator.extras["gate_pose"]), device=self.device).reshape(terrain_generator.cfg.num_cols, terrain_generator.cfg.num_rows, -1, 7)
                    self.extras["gate_pose"] = gate_pose
                    next_gate_id = torch.tensor(terrain_generator.extras["next_gate_id"], device=self.device).reshape(terrain_generator.cfg.num_cols, terrain_generator.cfg.num_rows)
                    self.extras["next_gate_id"] = next_gate_id
                else:
                    gate_pose = torch.tensor(np.stack(terrain_generator.extras["gate_pose"]), device=self.device).reshape(terrain_generator.cfg.num_rows, terrain_generator.cfg.num_cols, -1, 7).transpose(0, 1)
                    self.extras["gate_pose"] = gate_pose
                    next_gate_id = torch.tensor(terrain_generator.extras["next_gate_id"], device=self.device).reshape(terrain_generator.cfg.num_rows, terrain_generator.cfg.num_cols).transpose(0, 1)
                    self.extras["next_gate_id"] = next_gate_id

    
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)    