"""Heterogeneous-terrain Go2 env cfgs for the oracle-gap positive control.

Two single-type terrains built on the UNCHANGED Go2 rough velocity env (same
observations incl. height scan, same rewards and events, same pretrained
policy checkpoint):

  - HeteroFlatGo2EnvCfg  : every patch is a flat plane
  - HeteroRoughGo2EnvCfg : every patch is max-difficulty uniform-noise rough
    (the `random_rough` sub-terrain of ROUGH_TERRAINS_CFG at difficulty 1.0)

Pooling episodes from the two tasks emulates a deployment over genuinely
heterogeneous ground while keeping the terrain type of every episode known by
construction. Loaded lazily via gym registry entry points (see eval_hetero.py),
so this module must not be imported before the simulation app starts.
"""
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

FLAT_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)

ROUGH_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(1.0, 1.0),
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        )
    },
)


class HeteroFlatGo2EnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = FLAT_GEN
        self.scene.terrain.max_init_terrain_level = None


class HeteroRoughGo2EnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_GEN
        self.scene.terrain.max_init_terrain_level = None


MID_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(0.6, 0.6),
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        )
    },
)


class HeteroMidGo2EnvCfg(UnitreeGo2RoughEnvCfg):
    """Moderate-difficulty uniform rough: keeps success healthy so the paired
    basis is not starved the way max difficulty is."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = MID_GEN
        self.scene.terrain.max_init_terrain_level = None
