# scene_0 — Flat ground

Default flat MuJoCo scene for Go2 (checkerboard floor only).

## Layout

| Asset | Description |
|-------|-------------|
| `scene.xml` | Flat plane + Go2 (`../go2.xml`) |
| `README.md` | This file |

## Contents

- Infinite ground plane (`floor`) with checker material
- Skybox / haze / headlight
- No stairs, boxes, or heightfields
- Robot spawn near origin (freejoint default from `go2.xml`)

## Used by

- `sim2sim_deploy/sim2sim_go2_base.py` (default)
- `sim2sim_deploy/sim2sim_go2_stairs.py --flat`

```bash
python sim2sim_deploy/sim2sim_go2_base.py
```
