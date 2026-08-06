#!/usr/bin/env python3
"""Generate a MuJoCo scene with mixed stairs + random rough terrain for Go2 sim2sim."""

from __future__ import annotations

import argparse
import os
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent  # .../MCJF
MCJF = ROOT
SCENE_DIR = MCJF / "scene_1"
OUT_XML = SCENE_DIR / "scene.xml"

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


def add_random_boxes(
    geoms: list,
    *,
    rng: np.random.Generator,
    x_range,
    y_range,
    n: int,
    name_prefix: str,
) -> None:
    for i in range(n):
        sx = float(rng.uniform(0.15, 0.45))
        sy = float(rng.uniform(0.15, 0.45))
        sz = float(rng.uniform(0.03, 0.12))
        x = float(rng.uniform(*x_range))
        y = float(rng.uniform(*y_range))
        add_box(
            geoms,
            pos=(x, y, 0.5 * sz),
            size=(0.5 * sx, 0.5 * sy, 0.5 * sz),
            rgba="0.55 0.42 0.30 1",
            name=f"{name_prefix}_{i}",
        )


def _save_png_gray(path: Path, img: np.ndarray) -> None:
    """Save HxW uint8 grayscale PNG."""
    assert img.dtype == np.uint8 and img.ndim == 2
    h, w = img.shape
    try:
        from PIL import Image

        Image.fromarray(img, mode="L").save(path)
    except ImportError:

        def _chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        raw = b"".join(b"\x00" + row.tobytes() for row in img)
        png = b"\x89PNG\r\n\x1a\n"
        png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        png += _chunk(b"IDAT", zlib.compress(raw, 9))
        png += _chunk(b"IEND", b"")
        path.write_bytes(png)


def make_rough_hfield(path: Path, res: int = 256, seed: int = 0) -> None:
    """Multi-octave random rough heightfield with bumps / pits / ridges."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, res),
        np.linspace(0.0, 1.0, res),
        indexing="ij",
    )

    field = np.zeros((res, res), dtype=np.float64)

    # Multi-octave band-limited noise (approx Perlin via summed sines + noise).
    for octave in range(1, 7):
        amp = 1.0 / (1.4 ** (octave - 1))
        fx = float(rng.uniform(2.0, 9.0) * octave)
        fy = float(rng.uniform(2.0, 9.0) * octave)
        phx = float(rng.uniform(0.0, 2.0 * np.pi))
        phy = float(rng.uniform(0.0, 2.0 * np.pi))
        field += amp * np.sin(2.0 * np.pi * fx * xx + phx) * np.cos(
            2.0 * np.pi * fy * yy + phy
        )
        # Cross-term ridges
        if octave <= 3:
            field += 0.35 * amp * np.sin(
                2.0 * np.pi * (fx * xx + 0.7 * fy * yy) + phx
            )

    # Spatially correlated noise via multi-scale bilinear upsampling.
    noise = rng.standard_normal((res, res))
    for k in (8, 16, 32, 64):
        grid = rng.standard_normal((k, k)).astype(np.float64)
        try:
            from PIL import Image

            up = np.asarray(
                Image.fromarray(grid.astype(np.float32), mode="F").resize(
                    (res, res), resample=Image.BILINEAR
                ),
                dtype=np.float64,
            )
        except Exception:
            idx = np.linspace(0, k - 1, res).astype(int)
            up = grid[np.ix_(idx, idx)]
        field += (0.55 / np.sqrt(k)) * up
    field += 0.08 * noise

    # Random gaussian bumps / pits
    n_features = int(rng.integers(18, 36))
    for _ in range(n_features):
        cx = float(rng.uniform(0.05, 0.95))
        cy = float(rng.uniform(0.05, 0.95))
        sigma = float(rng.uniform(0.02, 0.12))
        amp = float(rng.uniform(-0.9, 1.2))
        field += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))

    # Occasional sharp rock-like spikes (clipped later by normalize)
    n_spikes = int(rng.integers(6, 14))
    for _ in range(n_spikes):
        cx = float(rng.uniform(0.1, 0.9))
        cy = float(rng.uniform(0.1, 0.9))
        sigma = float(rng.uniform(0.008, 0.025))
        amp = float(rng.uniform(0.4, 1.5))
        field += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))

    # Soften edges so the patch blends into the floor.
    edge = np.minimum.reduce([xx, 1.0 - xx, yy, 1.0 - yy])
    taper = np.clip(edge / 0.08, 0.0, 1.0)
    field = (field - field.mean()) * taper

    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    # Stretch contrast a bit so relief is more visible.
    field = np.clip(field**0.85, 0.0, 1.0)
    img = (field * 255.0).astype(np.uint8)
    _save_png_gray(path, img)


def build_scene(seed: int = 42) -> str:
    """Build scene_1 MJCF string.

    Shape conventions (MuJoCo box ``size`` = half-extents):
      full_x = 2*size_x, full_y = 2*size_y, full_z = 2*size_z
    Stairs: each tread is a box whose height grows with step index (solid under the step).
    """
    rng = np.random.default_rng(seed)
    geoms: list[str] = []
    assets: list[str] = []

    # --- Floor spawn pad (box) ---
    # Full footprint 3.0 m (x) x 9.0 m (y), thickness 2 cm; center at origin.
    # MuJoCo size=(1.5, 4.5, 0.01) => half-extents.
    add_box(
        geoms,
        pos=(0.0, 0.0, 0.01),
        size=(1.5, 4.5, 0.01),
        rgba="0.25 0.28 0.30 1",
        name="spawn_pad",
    )

    # --- Stair lanes (box stacks): up -> platform -> down, along +x from x0=1.5 ---
    # Fields: name, lane_y, step_h, step_w, n_steps, half_width, platform_len, rgba, title
    #   step_h  = riser height (m) per step
    #   step_w  = tread depth along +x (m) per step
    #   half_w  = half lane width along y (full width = 2*half_w)
    #   plat_len = flat top length along +x (m)
    lanes = [
        ("laneA", 0.0, 0.08, 0.31, 8, 0.90, 2.0, "0.50 0.52 0.55 1", "Lane A"),   # mid: H8cm W31cm
        ("laneB", 2.8, 0.12, 0.28, 8, 0.85, 1.8, "0.42 0.55 0.48 1", "Lane B"),   # tall/narrow
        ("laneC", -2.8, 0.18, 0.25, 6, 0.85, 1.5, "0.55 0.45 0.40 1", "Lane C"),  # steep
        ("laneD", 5.5, 0.05, 0.35, 10, 0.90, 2.2, "0.40 0.48 0.58 1", "Lane D"), # gentle/wide
    ]

    for name, y0, step_h, step_w, n_steps, half_w, plat_len, rgba, title in lanes:
        # Up-stairs: step i has height (i+1)*step_h, box size=(step_w/2, half_w, h/2)
        x = add_stairs_up(
            geoms,
            x0=1.5,
            y0=y0,
            step_h=step_h,
            step_w=step_w,
            n_steps=n_steps,
            half_width=half_w,
            name_prefix=name,
            rgba=rgba,
        )
        top = n_steps * step_h  # platform / top riser height
        # Platform: box length=plat_len, width=2*half_w, height=top
        x = add_platform(
            geoms,
            x0=x,
            y0=y0,
            length=plat_len,
            half_width=half_w,
            height=top,
            name=f"{name}_plat",
            rgba=rgba,
        )
        # Down-stairs: descending risers from top_height along +x
        add_stairs_down(
            geoms,
            x0=x,
            y0=y0,
            step_h=step_h,
            step_w=step_w,
            n_steps=n_steps,
            half_width=half_w,
            top_height=top,
            name_prefix=name,
            rgba=rgba,
        )

    # --- Random obstacle boxes (type=box) ---
    # Each box: full edges uniform in [0.15,0.45] m (xy), height [0.03,0.12] m.
    add_random_boxes(
        geoms, rng=rng, x_range=(0.2, 1.3), y_range=(-5.5, -3.8), n=18, name_prefix="randL",
    )
    add_random_boxes(
        geoms, rng=rng, x_range=(0.2, 1.3), y_range=(3.8, 5.0), n=14, name_prefix="randR",
    )
    add_random_boxes(
        geoms, rng=rng, x_range=(12.0, 16.0), y_range=(-2.0, 2.0), n=25, name_prefix="randFar",
    )

    # --- Rough heightfield patches (type=hfield) ---
    # size=(radius_x, radius_y, elevation_z, base_z): footprint 2*rx x 2*ry, peak ~elevation_z.
    # PNG is 256x256 grayscale multi-octave noise (see make_rough_hfield).
    hfield_geoms: list[str] = []
    patches = [
        # name, file, size(rx,ry,elev,base), pos, rgba, seed_off
        ("rough_main", "rough_hfield.png", (3.2, 2.2, 0.16, 0.02), (14.5, 0.0, 0.0), "0.35 0.50 0.35 1", 0),
        ("rough_n", "rough_hfield_n.png", (2.0, 1.6, 0.10, 0.02), (13.0, 3.8, 0.0), "0.40 0.48 0.32 1", 17),
        ("rough_s", "rough_hfield_s.png", (2.2, 1.5, 0.14, 0.02), (15.5, -3.5, 0.0), "0.32 0.45 0.38 1", 91),
        ("rough_far", "rough_hfield_far.png", (2.5, 2.0, 0.20, 0.02), (18.5, 1.0, 0.0), "0.30 0.42 0.30 1", 203),
    ]
    for hname, fname, size, pos, rgba, seed_off in patches:
        out_path = SCENE_DIR / fname
        make_rough_hfield(out_path, res=256, seed=seed + seed_off)
        # go2.xml meshdir="assets" -> resolve from MCJF/assets into scene_1/
        assets.append(
            f'    <hfield name="{hname}" '
            f'size="{size[0]} {size[1]} {size[2]} {size[3]}" '
            f'file="../scene_1/{fname}"/>'
        )
        hfield_geoms.append(
            f'    <geom name="{hname}_geom" type="hfield" hfield="{hname}" '
            f'pos="{pos[0]} {pos[1]} {pos[2]}" rgba="{rgba}"/>'
        )

    xml = f"""<mujoco model="go2 scene_1 stairs mixed">
  <include file="../go2.xml"/>
  <!-- go2.xml sets meshdir="assets"; override so meshes resolve from MCJF/ when this
       scene lives in scene_1/ -->
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
{os.linesep.join(assets)}
  </asset>

  <worldbody>
    <light pos="0 0 3.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{os.linesep.join(geoms)}
{os.linesep.join(hfield_geoms)}
  </worldbody>
</mujoco>
"""
    return xml


def main():
    # Usage:
    #   python resources/robots/go2/MCJF/generate_stairs_terrain.py
    #   python resources/robots/go2/MCJF/generate_stairs_terrain.py --seed 42
    #   python resources/robots/go2/MCJF/generate_stairs_terrain.py --out /path/to/scene.xml
    #
    # Writes scene_1/scene.xml + rough_hfield*.png. Shape / terrain sizes:
    #
    # [box] spawn_pad
    #   half-size (1.5, 4.5, 0.01) m  =>  footprint 3.0 x 9.0 m, thickness 2 cm, at origin
    #
    # [box] stair lanes (from x=1.5 m along +x: up -> platform -> down)
    #   MuJoCo box size = half-extents. Each up-step i: height=(i+1)*H, depth=W, width=2*half_w.
    #   Lane A y=0.0 : H=0.08 m, W=0.31 m, 8 steps, half_w=0.90, platform_len=2.0  (top=0.64 m)
    #   Lane B y=2.8 : H=0.12 m, W=0.28 m, 8 steps, half_w=0.85, platform_len=1.8  (top=0.96 m)
    #   Lane C y=-2.8: H=0.18 m, W=0.25 m, 6 steps, half_w=0.85, platform_len=1.5  (top=1.08 m)
    #   Lane D y=5.5 : H=0.05 m, W=0.35 m, 10 steps, half_w=0.90, platform_len=2.2 (top=0.50 m)
    #
    # [box] random obstacles (uniform samples)
    #   full xy edges [0.15, 0.45] m, height [0.03, 0.12] m
    #   zones: L n=18 x[0.2,1.3] y[-5.5,-3.8] | R n=14 x[0.2,1.3] y[3.8,5.0]
    #          Far n=25 x[12,16] y[-2,2]
    #
    # [hfield] rough patches (size = rx, ry, elev, base; footprint 2*rx x 2*ry)
    #   main  rx=3.2 ry=2.2 elev=0.16 base=0.02 @ (14.5, 0.0)
    #   north rx=2.0 ry=1.6 elev=0.10 base=0.02 @ (13.0, 3.8)
    #   south rx=2.2 ry=1.5 elev=0.14 base=0.02 @ (15.5,-3.5)
    #   far   rx=2.5 ry=2.0 elev=0.20 base=0.02 @ (18.5, 1.0)
    #   PNG 256x256 multi-octave noise + bumps/pits (seed + offset)
    #
    parser = argparse.ArgumentParser(
        description="Generate scene_1 mixed stairs + rough terrain MJCF"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(OUT_XML))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_scene(seed=args.seed))
    # Clean obsolete root-level copies if present.
    for name in (
        "scene_stairs_mixed.xml",
        "rough_hfield.png",
        "rough_hfield_n.png",
        "rough_hfield_s.png",
        "rough_hfield_far.png",
    ):
        old = MCJF / name
        if old.is_file():
            old.unlink()
    print(f"Wrote {out}")
    print(f"Wrote rough hfields under {SCENE_DIR}")


if __name__ == "__main__":
    main()
