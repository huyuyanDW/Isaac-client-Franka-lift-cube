import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_STATE8_DEFINITION = "eef_pos_xyz(3) + eef_axis_angle(3) + gripper_qpos(2)"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL line {line_idx} in {path}: {e}")
    return rows



def as_np(x, dtype=None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr



def has_nan_or_inf(arr: np.ndarray) -> tuple[bool, bool]:
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float32)
    return bool(np.isnan(arr).any()), bool(np.isinf(arr).any())



def remap_state8_from_state35(state35: np.ndarray, state_mode: str) -> np.ndarray | None:
    if state_mode == "zero":
        return np.zeros((8,), dtype=np.float32)

    if state_mode == "raw8":
        return state35[:8].astype(np.float32)

    if state_mode == "official":
        # official 8D state is now derived directly from robot EEF pose + gripper state
        # at collection time, not from state_35_raw. Therefore it cannot be recomputed here
        # unless those robot-state components were explicitly saved.
        return None

    raise ValueError(f"Unknown state_mode: {state_mode}")



def check_image_path(path_str: str, dataset_root: Path) -> list[str]:
    errors = []
    p = Path(path_str)
    if not p.is_absolute():
        p = dataset_root / p
    if not p.exists():
        errors.append(f"Missing image file: {p}")
    return errors



def validate_record(
    rec: dict,
    dataset_root: Path,
    expected_episode_idx: int,
    expected_step_idx: int,
    state_mode: str,
    meta_state8_definition: str | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    required_keys = [
        "episode_idx",
        "step_idx",
        "prompt",
        "task",
        "state_mode",
        "main_image_path",
        "wrist_image_path",
        "state_35_raw",
        "state_8_mapped",
        "action_chunk",
        "env_action_executed",
        "reward",
        "terminated",
        "truncated",
    ]
    for k in required_keys:
        if k not in rec:
            errors.append(f"Missing key: {k}")

    if errors:
        return errors, warnings

    if rec["episode_idx"] != expected_episode_idx:
        errors.append(
            f"episode_idx mismatch: got {rec['episode_idx']}, expected {expected_episode_idx}"
        )
    if rec["step_idx"] != expected_step_idx:
        errors.append(
            f"step_idx mismatch: got {rec['step_idx']}, expected {expected_step_idx}"
        )

    if rec["state_mode"] != state_mode:
        errors.append(f"state_mode mismatch: record={rec['state_mode']} meta={state_mode}")

    errors.extend(check_image_path(rec["main_image_path"], dataset_root))
    errors.extend(check_image_path(rec["wrist_image_path"], dataset_root))

    state35 = as_np(rec["state_35_raw"], dtype=np.float32).reshape(-1)
    state8 = as_np(rec["state_8_mapped"], dtype=np.float32).reshape(-1)

    if state35.shape[0] != 35:
        errors.append(f"state_35_raw length != 35, got {state35.shape}")
    if state8.shape[0] != 8:
        errors.append(f"state_8_mapped length != 8, got {state8.shape}")

    nan35, inf35 = has_nan_or_inf(state35)
    nan8, inf8 = has_nan_or_inf(state8)
    if nan35 or inf35:
        errors.append(f"state_35_raw contains nan/inf: nan={nan35} inf={inf35}")
    if nan8 or inf8:
        errors.append(f"state_8_mapped contains nan/inf: nan={nan8} inf={inf8}")

    record_state8_definition = rec.get("state8_definition")
    if meta_state8_definition is not None and record_state8_definition is not None:
        if record_state8_definition != meta_state8_definition:
            errors.append(
                "state8_definition mismatch between record and meta. "
                f"record={record_state8_definition!r} meta={meta_state8_definition!r}"
            )

    expected_state8 = None
    if state35.shape[0] == 35 and state8.shape[0] == 8:
        expected_state8 = remap_state8_from_state35(state35, state_mode)

    if expected_state8 is not None:
        if not np.allclose(state8, expected_state8, atol=1e-5, rtol=1e-5):
            errors.append(
                f"state_8_mapped does not match recomputed {state_mode} mapping.\n"
                f"saved={state8}\nexpected={expected_state8}"
            )
    elif state_mode == "official":
        effective_definition = record_state8_definition or meta_state8_definition
        if effective_definition != OFFICIAL_STATE8_DEFINITION:
            errors.append(
                "official state8_definition mismatch. "
                f"expected={OFFICIAL_STATE8_DEFINITION!r} got={effective_definition!r}"
            )
        else:
            warnings.append(
                "official 8D state is validated by stored definition/shape/nan checks only; "
                "it is no longer recomputable from state_35_raw alone."
            )

    action_chunk = as_np(rec["action_chunk"], dtype=np.float32)
    if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
        errors.append(f"action_chunk expected shape (T, 7), got {action_chunk.shape}")

    env_action = as_np(rec["env_action_executed"], dtype=np.float32)
    env_action_flat = env_action.reshape(-1)
    if env_action_flat.shape[0] != 7:
        errors.append(f"env_action_executed expected 7 values after flatten, got {env_action.shape}")

    nan_a, inf_a = has_nan_or_inf(env_action_flat)
    if nan_a or inf_a:
        errors.append(f"env_action_executed contains nan/inf: nan={nan_a} inf={inf_a}")

    if action_chunk.ndim == 2 and action_chunk.shape[1] == 7 and env_action_flat.shape[0] == 7:
        if not np.allclose(action_chunk[0], env_action_flat, atol=1e-5, rtol=1e-5):
            errors.append(
                f"env_action_executed does not match action_chunk[0].\n"
                f"chunk0={action_chunk[0]}\nexec={env_action_flat}"
            )

    reward = as_np(rec["reward"])
    terminated = as_np(rec["terminated"])
    truncated = as_np(rec["truncated"])

    if reward.size < 1:
        errors.append("reward is empty")
    if terminated.size < 1:
        errors.append("terminated is empty")
    if truncated.size < 1:
        errors.append("truncated is empty")

    return errors, warnings



def validate_temporal_consistency(records: list[dict]) -> list[str]:
    """
    In this env, state_35_raw[28:35] stores 'actions'.
    It should approximately match the previous step's executed 7D action.
    """
    errors: list[str] = []

    for i in range(1, len(records)):
        prev_action = as_np(records[i - 1]["env_action_executed"], dtype=np.float32).reshape(-1)
        curr_state35 = as_np(records[i]["state_35_raw"], dtype=np.float32).reshape(-1)

        if prev_action.shape[0] != 7 or curr_state35.shape[0] != 35:
            continue

        curr_last_action = curr_state35[28:35]
        if not np.allclose(curr_last_action, prev_action, atol=1e-3, rtol=1e-3):
            errors.append(
                f"Temporal mismatch at transition {i-1}->{i}: "
                f"next state35[28:35] != prev env_action_executed.\n"
                f"prev_action={prev_action}\nnext_last_action={curr_last_action}"
            )

    return errors



def main():
    parser = argparse.ArgumentParser(description="Validate rollout dataset collected by 09_collect_rollout_dataset.py")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/gpufree-data/isaac_client/datasets/rollout_dataset_v1",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    meta_path = dataset_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    meta = load_json(meta_path)
    state_mode = meta.get("state_mode", "official")
    meta_state8_definition = meta.get("state8_definition")

    episode_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found in {dataset_root}")

    total_records = 0
    total_errors = 0
    total_warnings = 0

    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] state_mode   = {state_mode}")
    print(f"[INFO] state8_def   = {meta_state8_definition}")
    print(f"[INFO] episodes     = {len(episode_dirs)}")

    for ep_dir in episode_dirs:
        episode_idx = int(ep_dir.name.split("_")[-1])
        steps_path = ep_dir / "steps.jsonl"
        if not steps_path.exists():
            print(f"[ERROR] Missing steps.jsonl in {ep_dir}")
            total_errors += 1
            continue

        records = load_jsonl(steps_path)
        print(f"\n[INFO] Checking {ep_dir.name}: {len(records)} records")
        total_records += len(records)

        ep_errors = []
        ep_warnings = []

        episode_meta_path = ep_dir / "episode_meta.json"
        if episode_meta_path.exists():
            episode_meta = load_json(episode_meta_path)
            episode_state8_definition = episode_meta.get("state8_definition")
            if (
                meta_state8_definition is not None
                and episode_state8_definition is not None
                and episode_state8_definition != meta_state8_definition
            ):
                ep_errors.append(
                    "episode_meta.json state8_definition mismatch. "
                    f"episode={episode_state8_definition!r} meta={meta_state8_definition!r}"
                )

        for step_idx, rec in enumerate(records):
            errs, warns = validate_record(
                rec=rec,
                dataset_root=dataset_root,
                expected_episode_idx=episode_idx,
                expected_step_idx=step_idx,
                state_mode=state_mode,
                meta_state8_definition=meta_state8_definition,
            )
            ep_errors.extend([f"[record {step_idx}] {e}" for e in errs])
            ep_warnings.extend([f"[record {step_idx}] {w}" for w in warns])

        ep_errors.extend(validate_temporal_consistency(records))

        if ep_errors:
            print(f"[ERROR] {ep_dir.name} failed with {len(ep_errors)} issues.")
            for e in ep_errors[:20]:
                print("  -", e)
            if len(ep_errors) > 20:
                print(f"  ... and {len(ep_errors) - 20} more")
            total_errors += len(ep_errors)
        else:
            print(f"[OK] {ep_dir.name} passed.")

        if ep_warnings:
            unique_warns = list(dict.fromkeys(ep_warnings))
            print(f"[WARN] {ep_dir.name} produced {len(unique_warns)} unique warnings.")
            for w in unique_warns[:3]:
                print("  -", w)
            total_warnings += len(unique_warns)

    print("\n[SUMMARY]")
    print(f"  total_records  = {total_records}")
    print(f"  total_errors   = {total_errors}")
    print(f"  total_warnings = {total_warnings}")

    if total_errors == 0:
        print("[SUCCESS] Dataset validation passed.")
    else:
        print("[FAIL] Dataset validation found issues.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
