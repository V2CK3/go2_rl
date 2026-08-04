"""Config snapshot / flatten / diff helpers (no isaacgym dependency)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# experiment_name -> (env config module path, env class, train class)
EXPERIMENT_CONFIG_SOURCES = {
    "go2_base": (
        "legged_gym/envs/go2/go2_base_config.py",
        "Go2BaseCfg",
        "Go2BaseCfgPPO",
    ),
    "go2_stairs": (
        "legged_gym/envs/go2/go2_stairs_config.py",
        "Go2StairsCfg",
        "Go2StairsCfgPPO",
    ),
}

# Keys most useful in a side-by-side comparison table.
HIGHLIGHT_KEYS = [
    "seed",
    "runner.experiment_name",
    "runner.run_name",
    "runner.max_iterations",
    "runner.num_steps_per_env",
    "runner.save_interval",
    "env.num_envs",
    "env.frame_stack",
    "env.num_single_obs",
    "env.num_observations",
    "env.episode_length_s",
    "terrain.mesh_type",
    "terrain.curriculum",
    "terrain.measure_heights",
    "control.action_scale",
    "control.stiffness.joint",
    "control.damping.joint",
    "control.decimation",
    "algorithm.learning_rate",
    "algorithm.gamma",
    "algorithm.lam",
    "algorithm.entropy_coef",
    "algorithm.clip_param",
    "algorithm.sym_loss",
    "algorithm.sym_coef",
    "algorithm.num_learning_epochs",
    "algorithm.num_mini_batches",
    "policy.actor_hidden_dims",
    "policy.critic_hidden_dims",
    "policy.init_noise_std",
    "commands.curriculum",
    "commands.ranges.lin_vel_x",
    "commands.ranges.lin_vel_y",
    "commands.ranges.ang_vel_yaw",
    "domain_rand.randomize_friction",
    "domain_rand.push_robots",
    "noise.add_noise",
    "noise.noise_level",
    "rewards.cycle_time",
    "rewards.base_height_target",
]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert config objects / nested classes to JSON-safe values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, type):
        try:
            return to_jsonable(obj())
        except Exception:
            return obj.__name__
    if hasattr(obj, "__dict__"):
        out = {}
        for key, val in vars(obj).items():
            if key.startswith("_"):
                continue
            if callable(val) and not isinstance(val, type):
                continue
            out[key] = to_jsonable(val)
        for key in dir(obj):
            if key.startswith("_") or key in out:
                continue
            val = getattr(obj, key)
            if isinstance(val, type):
                out[key] = to_jsonable(val)
            elif not callable(val):
                out[key] = to_jsonable(val)
        return out
    return str(obj)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict to dotted keys."""
    items: Dict[str, Any] = {}
    for key, val in d.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, dict):
            items.update(flatten_dict(val, path))
        else:
            items[path] = val
    return items


def format_value(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if abs(val) >= 1e-3 and abs(val) < 1e4:
            return f"{val:g}"
        return f"{val:.3e}"
    if isinstance(val, (list, tuple, dict)):
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    return str(val)


def values_equal(a: Any, b: Any) -> bool:
    return format_value(a) == format_value(b)


def merge_train_env_cfg(train_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge train + env configs the same way task_registry does."""
    return {**train_cfg, **env_cfg}


def _ensure_config_import_stubs() -> None:
    """Register lightweight package stubs so config files import without isaacgym."""
    import types

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    # legged_gym/__init__.py is safe (no isaacgym). Prefer the real module.
    if "legged_gym" not in sys.modules:
        import legged_gym  # noqa: F401

    # Avoid executing legged_gym/envs/__init__.py (pulls isaacgym via task_registry).
    if "legged_gym.envs" not in sys.modules or not hasattr(sys.modules["legged_gym.envs"], "__path__"):
        envs_pkg = types.ModuleType("legged_gym.envs")
        envs_pkg.__path__ = [os.path.join(REPO_ROOT, "legged_gym", "envs")]
        sys.modules["legged_gym.envs"] = envs_pkg

    if "legged_gym.envs.base" not in sys.modules:
        base_pkg = types.ModuleType("legged_gym.envs.base")
        base_pkg.__path__ = [os.path.join(REPO_ROOT, "legged_gym", "envs", "base")]
        sys.modules["legged_gym.envs.base"] = base_pkg

    if "legged_gym.envs.base.base_config" not in sys.modules:
        base_cfg_path = os.path.join(REPO_ROOT, "legged_gym", "envs", "base", "base_config.py")
        spec = importlib.util.spec_from_file_location(
            "legged_gym.envs.base.base_config", base_cfg_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["legged_gym.envs.base.base_config"] = module
        spec.loader.exec_module(module)


def load_config_module(rel_path: str):
    _ensure_config_import_stubs()
    abs_path = os.path.join(REPO_ROOT, rel_path)
    mod_name = "_exp_cfg_" + os.path.basename(rel_path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def snapshot_from_source(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Build a config snapshot from current source for an experiment name."""
    meta = EXPERIMENT_CONFIG_SOURCES.get(experiment_name)
    if meta is None:
        return None
    rel_path, env_cls_name, train_cls_name = meta
    module = load_config_module(rel_path)
    env_cls = getattr(module, env_cls_name)
    train_cls = getattr(module, train_cls_name)
    env_cfg = to_jsonable(env_cls())
    train_cfg = to_jsonable(train_cls())
    merged = merge_train_env_cfg(train_cfg, env_cfg)
    merged["_meta"] = {
        "source": "source_config",
        "config_file": rel_path,
        "env_class": env_cls_name,
        "train_class": train_cls_name,
        "note": "Snapshot from current source (may differ from the actual training-time config).",
    }
    return merged


def save_config_json(path: str, cfg: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)


def load_config_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diff_flat_configs(
    configs: Dict[str, Dict[str, Any]],
    only_highlight: bool = False,
    only_diff: bool = True,
) -> List[Dict[str, Any]]:
    """Build rows for a comparison table. configs: {run_label: nested_config}."""
    flat_map: Dict[str, Dict[str, Any]] = {}
    all_keys = set()
    for label, cfg in configs.items():
        flat = flatten_dict(cfg) if cfg else {}
        flat = {k: v for k, v in flat.items() if not k.startswith("_meta")}
        flat_map[label] = flat
        all_keys.update(flat.keys())

    if only_highlight:
        keys = [k for k in HIGHLIGHT_KEYS if k in all_keys]
        reward_keys = sorted(k for k in all_keys if k.startswith("rewards.scales."))
        keys.extend(reward_keys)
        seen = set()
        ordered = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        keys = ordered
    else:
        keys = sorted(all_keys)

    labels = list(configs.keys())
    rows = []
    for key in keys:
        row: Dict[str, Any] = {"key": key}
        vals = []
        for label in labels:
            val = flat_map[label].get(key)
            row[label] = val
            vals.append(val)
        differs = False
        for i in range(1, len(vals)):
            if not values_equal(vals[0], vals[i]):
                differs = True
                break
        present = [label for label in labels if key in flat_map[label]]
        if 0 < len(present) < len(labels):
            differs = True
        row["differs"] = differs
        if only_diff and not differs:
            continue
        rows.append(row)
    return rows
