# scene_1 — Mixed stairs + rough terrain

Go2 MuJoCo sim2sim scene used by `sim2sim_deploy/sim2sim_go2_stairs.py`.

## Layout

| Asset | Description |
|-------|-------------|
| `scene.xml` | Main MuJoCo scene (includes `../go2.xml`) |
| `rough_hfield*.png` | Heightfield textures for rough patches |
| `README.md` | This file |

Robot meshes stay in `../assets/` (via `go2.xml`).

## Spawn

- Flat **spawn pad** at origin: ±1.5 m (x) × ±4.5 m (y), thickness 2 cm
- Stairs begin at **x = 1.5 m**, ascending along **+x**

## Stair lanes

Each lane: **up stairs → top platform → down stairs**.

| Lane | y (m) | Step height H | Step width W | # steps | Platform length | Color |
|------|-------|---------------|--------------|---------|-----------------|-------|
| A | 0.0 | **8 cm** | **31 cm** | 8 | 2.0 m | gray |
| B | 2.8 | **12 cm** | **28 cm** | 8 | 1.8 m | green-gray |
| C | -2.8 | **18 cm** | **25 cm** | 6 | 1.5 m | brown |
| D | 5.5 | **5 cm** | **35 cm** | 10 | 2.2 m | blue-gray |

Lane half-widths ≈ 0.85–0.90 m.

## Random box obstacles

Scattered low boxes (height ~3–12 cm):

- Left of spawn: x∈[0.2, 1.3], y∈[-5.5, -3.8] (18 boxes)
- Right of spawn: x∈[0.2, 1.3], y∈[3.8, 5.0] (14 boxes)
- Far field: x∈[12, 16], y∈[-2, 2] (25 boxes)

## Rough heightfields

Multi-octave random relief (bumps / pits / ridges), default seed `42`:

| Patch | File | Half-size (x,y) | Peak elev. | Center pos (x,y,z) |
|-------|------|-----------------|------------|--------------------|
| main | `rough_hfield.png` | 3.2 × 2.2 m | 0.16 m | (14.5, 0.0, 0) |
| north | `rough_hfield_n.png` | 2.0 × 1.6 m | 0.10 m | (13.0, 3.8, 0) |
| south | `rough_hfield_s.png` | 2.2 × 1.5 m | 0.14 m | (15.5, -3.5, 0) |
| far | `rough_hfield_far.png` | 2.5 × 2.0 m | 0.20 m | (18.5, 1.0, 0) |

## Regenerate

```bash
python sim2sim_deploy/generate_stairs_terrain.py [--seed 42]
```

Outputs overwrite `scene.xml` and the four PNG heightfields in this folder.

## Run sim2sim

```bash
python sim2sim_deploy/sim2sim_go2_stairs.py
```
