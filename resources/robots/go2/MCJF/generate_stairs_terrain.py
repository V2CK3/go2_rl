#!/usr/bin/env python3
"""Generate MuJoCo stairs scenes matched to Isaac Gym training difficulty.

Training source (``legged_gym/utils/terrain.py`` + ``Go2StairsCfg``)::

    difficulty = terrain_level / num_rows          # num_rows = 10
    step_height = 0.05 + 0.18 * difficulty        # meters
    step_width  = 0.31                            # meters (fixed)
    platform_size = 3.0                           # meters

Writes ``scene_stairs_L{0..9}/scene.xml`` + ``README.md`` for each level.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # .../MCJF
MCJF = ROOT

# Match Go2StairsCfg.terrain
NUM_ROWS = 10
STEP_WIDTH = 0.31
PLATFORM_SIZE = 3.0
N_STEPS = 8  # ~ half of 8 m env minus platform / step_width
HALF_WIDTH = 1.2  # lane half-width (m); full width = 2.4 m
SPAWN_PAD_HALF = (1.5, 2.0, 0.01)


def difficulty_for_level(level: int) -> float:
    """Curriculum difficulty for terrain row ``level`` in ``[0, num_rows)``."""
    if not 0 <= level < NUM_ROWS:
        raise ValueError(f"level must be in [0, {NUM_ROWS}), got {level}")
    return level / NUM_ROWS


def step_height_for_difficulty(difficulty: float) -> float:
    """Isaac Gym stairs riser height (m)."""
    return 0.05 + 0.18 * difficulty


def level_rgba(level: int) -> str:
    """Visual tint: cool/easy -> warm/hard."""
    t = level / max(NUM_ROWS - 1, 1)
    r = 0.35 + 0.35 * t
    g = 0.55 - 0.20 * t
    b = 0.50 - 0.25 * t
    return f"{r:.2f} {g:.2f} {b:.2f} 1"


def _quat_wxyz(w=1.0, x=0.0, y=0.0, z=0.0) -> str:
    return f"{w} {x} {y} {z}"


def add_box(
    geoms: list,
    pos,
    size,
    rgba="0.45 0.45 0.48 1",
    name=None,
    *,
    material=None,
    contype: int = 1,
    conaffinity: int = 1,
    group: int | None = None,
    quat: str | None = None,
) -> None:
    n = f' name="{name}"' if name else ""
    mat = f' material="{material}"' if material else ""
    grp = f' group="{group}"' if group is not None else ""
    q = quat if quat is not None else _quat_wxyz()
    rgba_attr = "" if material else f' rgba="{rgba}"'
    geoms.append(
        f'    <geom{n} type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
        f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" '
        f'quat="{q}"{mat}{rgba_attr} contype="{contype}" conaffinity="{conaffinity}"{grp}/>'
    )


def add_stairs_up(
    geoms: list,
    *,
    x0: float,
    y0: float,
    step_h: float,
    step_w: float,
    n_steps: int,
    half_width: float,
    name_prefix: str,
    rgba: str,
) -> float:
    """Ascending stairs along +x. Returns x of last tread front edge."""
    for i in range(n_steps):
        h = (i + 1) * step_h
        x = x0 + i * step_w + 0.5 * step_w
        add_box(
            geoms,
            pos=(x, y0, 0.5 * h),
            size=(0.5 * step_w, half_width, 0.5 * h),
            rgba=rgba,
            name=f"{name_prefix}_up_{i}",
        )
    return x0 + n_steps * step_w


def add_platform(
    geoms: list,
    *,
    x0: float,
    y0: float,
    length: float,
    half_width: float,
    height: float,
    name: str,
    rgba: str,
) -> float:
    add_box(
        geoms,
        pos=(x0 + 0.5 * length, y0, 0.5 * height),
        size=(0.5 * length, half_width, 0.5 * height),
        rgba=rgba,
        name=name,
    )
    return x0 + length


def add_stairs_down(
    geoms: list,
    *,
    x0: float,
    y0: float,
    step_h: float,
    step_w: float,
    n_steps: int,
    half_width: float,
    top_height: float,
    name_prefix: str,
    rgba: str,
) -> float:
    """Descending stairs along +x from top_height."""
    for i in range(n_steps):
        h = top_height - i * step_h
        if h <= 1e-4:
            break
        x = x0 + i * step_w + 0.5 * step_w
        add_box(
            geoms,
            pos=(x, y0, 0.5 * h),
            size=(0.5 * step_w, half_width, 0.5 * h),
            rgba=rgba,
            name=f"{name_prefix}_dn_{i}",
        )
    return x0 + n_steps * step_w


def scene_dir_for_level(level: int) -> Path:
    return MCJF / f"scene_stairs_L{level}"


def build_level_scene(level: int) -> str:
    """Build MJCF for one curriculum stairs level (up -> platform -> down)."""
    difficulty = difficulty_for_level(level)
    step_h = step_height_for_difficulty(difficulty)
    step_w = STEP_WIDTH
    rgba = level_rgba(level)
    geoms: list[str] = []

    add_box(
        geoms,
        pos=(0.0, 0.0, SPAWN_PAD_HALF[2]),
        size=SPAWN_PAD_HALF,
        rgba="0.25 0.28 0.30 1",
        name="spawn_pad",
    )

    x0 = SPAWN_PAD_HALF[0]  # start stairs at front edge of spawn pad
    x = add_stairs_up(
        geoms,
        x0=x0,
        y0=0.0,
        step_h=step_h,
        step_w=step_w,
        n_steps=N_STEPS,
        half_width=HALF_WIDTH,
        name_prefix="stairs",
        rgba=rgba,
    )
    top = N_STEPS * step_h
    x = add_platform(
        geoms,
        x0=x,
        y0=0.0,
        length=PLATFORM_SIZE,
        half_width=HALF_WIDTH,
        height=top,
        name="stairs_plat",
        rgba=rgba,
    )
    add_stairs_down(
        geoms,
        x0=x,
        y0=0.0,
        step_h=step_h,
        step_w=step_w,
        n_steps=N_STEPS,
        half_width=HALF_WIDTH,
        top_height=top,
        name_prefix="stairs",
        rgba=rgba,
    )

    xml = f"""<mujoco model="go2 scene_stairs_L{level}">
  <include file="../go2.xml"/>
  <!-- go2.xml sets meshdir="assets"; override so meshes resolve from MCJF/ -->
  <compiler meshdir="../assets"/>

  <statistic center="6 0 0.4" extent="10"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{os.linesep.join(geoms)}
  </worldbody>
</mujoco>
"""
    return xml


def write_level_readme(level: int, out_dir: Path) -> None:
    difficulty = difficulty_for_level(level)
    step_h = step_height_for_difficulty(difficulty)
    top = N_STEPS * step_h
    run_len = N_STEPS * STEP_WIDTH
    total_x = SPAWN_PAD_HALF[0] + run_len + PLATFORM_SIZE + run_len

    text = f"""# scene_stairs_L{level} — training curriculum level {level}

MuJoCo stairs scene matched to Isaac Gym `Go2StairsCfg` / `terrain.py` curriculum.

## Training formula

From `legged_gym/utils/terrain.py` (`make_terrain` + `curiculum`):

```
difficulty   = level / num_rows          # num_rows = {NUM_ROWS}
step_height  = 0.05 + 0.18 * difficulty  # meters
step_width   = 0.31                      # meters (fixed)
platform_size = 3.0                      # meters
```

Stair type in training: `pyramid_stairs_terrain` (stairs-up / stairs-down columns).

## This scene (level {level})

| Parameter | Value |
|-----------|-------|
| `level` | {level} |
| `difficulty` | {difficulty:.2f} (`{level}/{NUM_ROWS}`) |
| `step_height` (riser) | **{step_h:.4f} m** ({step_h * 100:.2f} cm) |
| `step_width` (tread) | **{STEP_WIDTH:.2f} m** ({STEP_WIDTH * 100:.0f} cm) |
| `n_steps` (each side) | {N_STEPS} |
| platform length | {PLATFORM_SIZE:.1f} m |
| platform / top height | {top:.4f} m |
| lane half-width | {HALF_WIDTH:.1f} m (full {2 * HALF_WIDTH:.1f} m) |
| layout | spawn pad → up stairs → platform → down stairs (+x) |
| approx. run length (x) | {total_x:.2f} m from origin |

Sim2sim uses box stacks (up / platform / down) with the same `step_height` /
`step_width` / platform length as training. Geometry is a linear corridor, not
a full heightfield pyramid.

## How to load

In `sim2sim_deploy/sim2sim_go2_stairs.py` set:

```python
difficulty_level = {level}  # 0 .. {NUM_ROWS - 1}
```

Or open this file directly:

```
resources/robots/go2/MCJF/scene_stairs_L{level}/scene.xml
```

## Regenerate

```bash
python resources/robots/go2/MCJF/generate_stairs_terrain.py
python resources/robots/go2/MCJF/generate_stairs_terrain.py --level {level}
```

See also: `../STAIRS_DIFFICULTY.md` for the full level table.
"""
    (out_dir / "README.md").write_text(text)


def write_index_readme(out_path: Path) -> None:
    rows = []
    for level in range(NUM_ROWS):
        d = difficulty_for_level(level)
        h = step_height_for_difficulty(d)
        rows.append(
            f"| {level} | {d:.2f} | {h:.4f} | {h * 100:.2f} | "
            f"`scene_stairs_L{level}/` |"
        )
    table = "\n".join(rows)
    text = f"""# Stairs difficulty levels (sim2sim ↔ training)

Curriculum stairs parameters from Isaac Gym training
(`Go2StairsCfg.terrain.num_rows = {NUM_ROWS}`, `legged_gym/utils/terrain.py`):

```
difficulty  = level / {NUM_ROWS}
step_height = 0.05 + 0.18 * difficulty   # m
step_width  = 0.31                       # m (all levels)
platform    = 3.0                        # m
```

`max_init_terrain_level = 5` only affects **initial** spawn rows; the map still
has rows `0 .. {NUM_ROWS - 1}`.

## Level table

| Level | difficulty | step_height (m) | step_height (cm) | Scene folder |
|------:|-----------:|----------------:|-----------------:|:-------------|
{table}

Fixed for all levels: `step_width = {STEP_WIDTH} m`, `platform_size = {PLATFORM_SIZE} m`,
`n_steps = {N_STEPS}` per side.

## Scene index

| Folder | Role |
|--------|------|
| `scene_0/` | Flat ground |
| `scene_stairs_L0/` … `scene_stairs_L9/` | Curriculum stairs by difficulty |
| `scene_1/` | Legacy mixed multi-lane stairs (not difficulty-indexed) |
| `scene_2/` | Legacy heightfield terrain |

## sim2sim

```python
# sim2sim_deploy/sim2sim_go2_stairs.py
difficulty_level = 4  # pick 0..9
```

Regenerate all levels:

```bash
python resources/robots/go2/MCJF/generate_stairs_terrain.py
```
"""
    out_path.write_text(text)


def generate_level(level: int) -> Path:
    out_dir = scene_dir_for_level(level)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.xml").write_text(build_level_scene(level))
    write_level_readme(level, out_dir)
    return out_dir


def main():
    # Usage:
    #   python resources/robots/go2/MCJF/generate_stairs_terrain.py
    #   python resources/robots/go2/MCJF/generate_stairs_terrain.py --level 4
    #
    # Training match:
    #   difficulty = level / 10
    #   step_height = 0.05 + 0.18 * difficulty
    #   step_width  = 0.31
    #   platform    = 3.0
    parser = argparse.ArgumentParser(
        description="Generate curriculum stairs MJCF scenes (scene_stairs_L0..L9)"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help=f"Generate a single level in [0, {NUM_ROWS}). Default: all levels.",
    )
    args = parser.parse_args()

    levels = [args.level] if args.level is not None else list(range(NUM_ROWS))
    for level in levels:
        out_dir = generate_level(level)
        d = difficulty_for_level(level)
        h = step_height_for_difficulty(d)
        print(
            f"Wrote {out_dir}/scene.xml  "
            f"(level={level} difficulty={d:.2f} H={h:.4f}m W={STEP_WIDTH}m)"
        )

    write_index_readme(MCJF / "STAIRS_DIFFICULTY.md")
    print(f"Wrote {MCJF / 'STAIRS_DIFFICULTY.md'}")


if __name__ == "__main__":
    main()
