import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import shutil

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect rollout dataset from Isaac Lab + openpi server")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--steps_per_episode", type=int, default=50)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--prompt", type=str, default="lift the cube")
parser.add_argument(
    "--state-mode",
    type=str,
    default="official",
    choices=["zero", "raw8", "official"],
)
parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument("--dataset_root", type=str, default="/root/gpufree-data/isaac_client/datasets/rollout_dataset_v1")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
        return
    except Exception:
        pass

    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
        return
    except Exception:
        np.save(str(path) + ".npy", img)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Frame is None")

    image = frame
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Unexpected image ndim: {image.ndim}, shape={image.shape}")

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image.astype(np.uint8)

    try:
        from PIL import Image
        image = np.array(Image.fromarray(image).resize((224, 224)))
    except Exception as e:
        raise RuntimeError(f"Failed to resize frame to 224x224: {e}")

    return image


def extract_state8_zero(raw_state: np.ndarray) -> np.ndarray:
    return np.zeros((8,), dtype=np.float32)


def extract_state8_raw8(raw_state: np.ndarray) -> np.ndarray:
    return raw_state[:8].astype(np.float32)


def quat_wxyz_to_axis_angle(quat_wxyz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < eps:
        return np.zeros((3,), dtype=np.float32)

    quat = quat / norm
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:]
    sin_half = float(np.linalg.norm(xyz))

    if sin_half < eps:
        return np.zeros((3,), dtype=np.float32)

    axis = xyz / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (axis * angle).astype(np.float32)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _find_first_available(scene: Any, names: list[str]):
    for name in names:
        try:
            return scene[name], name
        except Exception:
            continue
    raise KeyError(f"Could not find any of {names} in env.unwrapped.scene")


def _first_index_from_find_result(result: Any, what: str) -> int:
    if isinstance(result, tuple):
        indices = result[0]
    else:
        indices = result

    if hasattr(indices, "tolist"):
        indices = indices.tolist()

    if isinstance(indices, (list, tuple)):
        if len(indices) == 0:
            raise RuntimeError(f"find_{what} returned empty result")
        first = indices[0]
        if isinstance(first, (list, tuple)):
            if len(first) == 0:
                raise RuntimeError(f"find_{what} returned nested empty result")
            first = first[0]
        return int(first)

    return int(indices)


def resolve_robot_state_handles(env) -> dict[str, Any]:
    scene = env.unwrapped.scene
    robot, robot_name = _find_first_available(scene, ["robot", "Robot"])

    eef_body_idx = None
    eef_body_name = None
    eef_candidates = [
        "panda_hand",
        "hand",
        "ee_link",
        "panda_link8",
    ]
    for candidate in eef_candidates:
        try:
            eef_result = robot.find_bodies(candidate)
            eef_body_idx = _first_index_from_find_result(eef_result, "bodies")
            eef_body_name = candidate
            break
        except Exception:
            continue
    if eef_body_idx is None:
        raise RuntimeError(f"Could not resolve EEF body from candidates: {eef_candidates}")

    finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    try:
        finger_result = robot.find_joints(finger_joint_names)
        if isinstance(finger_result, tuple):
            finger_indices = finger_result[0]
        else:
            finger_indices = finger_result
        if hasattr(finger_indices, "tolist"):
            finger_indices = finger_indices.tolist()
        finger_joint_indices = [int(x) for x in finger_indices]
    except Exception as e:
        raise RuntimeError(
            "Could not resolve gripper joints panda_finger_joint1/2. "
            "Please confirm the Franka articulation joint names."
        ) from e

    print(
        "[INFO] official8 uses robot="
        f"{robot_name}, eef_body={eef_body_name}[{eef_body_idx}], "
        f"gripper_joints={finger_joint_names}[{finger_joint_indices}]"
    )

    return {
        "robot": robot,
        "robot_name": robot_name,
        "eef_body_idx": eef_body_idx,
        "eef_body_name": eef_body_name,
        "finger_joint_indices": finger_joint_indices,
        "finger_joint_names": finger_joint_names,
    }


def _extract_eef_pos_quat_w(robot) -> tuple[np.ndarray, np.ndarray]:
    data = robot.data

    pos = None
    quat = None

    if hasattr(data, "body_link_pos_w") and hasattr(data, "body_link_quat_w"):
        pos = _to_numpy(data.body_link_pos_w)
        quat = _to_numpy(data.body_link_quat_w)
    elif hasattr(data, "body_pos_w") and hasattr(data, "body_quat_w"):
        pos = _to_numpy(data.body_pos_w)
        quat = _to_numpy(data.body_quat_w)
    elif hasattr(data, "body_link_state_w"):
        state = _to_numpy(data.body_link_state_w)
        pos = state[..., 0:3]
        quat = state[..., 3:7]
    elif hasattr(data, "body_state_w"):
        state = _to_numpy(data.body_state_w)
        pos = state[..., 0:3]
        quat = state[..., 3:7]
    else:
        raise RuntimeError(
            "Could not find EEF pose tensors on robot.data. "
            "Expected one of body_link_pos_w/body_link_quat_w, body_pos_w/body_quat_w, "
            "body_link_state_w, or body_state_w."
        )

    return pos, quat


def extract_state8_official_from_env(env, handles: dict[str, Any], env_idx: int = 0) -> np.ndarray:
    robot = handles["robot"]
    eef_body_idx = handles["eef_body_idx"]
    finger_joint_indices = handles["finger_joint_indices"]

    pos_all, quat_all = _extract_eef_pos_quat_w(robot)
    joint_pos_all = _to_numpy(robot.data.joint_pos)

    eef_pos = np.asarray(pos_all[env_idx, eef_body_idx], dtype=np.float32).reshape(3)
    eef_quat_wxyz = np.asarray(quat_all[env_idx, eef_body_idx], dtype=np.float32).reshape(4)
    eef_axis_angle = quat_wxyz_to_axis_angle(eef_quat_wxyz)
    gripper_qpos = np.asarray(joint_pos_all[env_idx, finger_joint_indices], dtype=np.float32).reshape(2)

    state8 = np.concatenate([eef_pos, eef_axis_angle, gripper_qpos], axis=0).astype(np.float32)
    if state8.shape != (8,):
        raise RuntimeError(f"official state8 shape mismatch: expected (8,), got {state8.shape}")
    return state8


def map_state8(raw_state: np.ndarray, mode: str, env=None, robot_state_handles=None) -> np.ndarray:
    if mode == "zero":
        return extract_state8_zero(raw_state)
    if mode == "raw8":
        return extract_state8_raw8(raw_state)
    if mode == "official":
        if env is None or robot_state_handles is None:
            raise ValueError("official state mode requires env and robot_state_handles")
        return extract_state8_official_from_env(env, robot_state_handles)
    raise ValueError(f"Unknown state mode: {mode}")


def action_chunk_to_env_action(action_chunk: Any, env):
    import torch

    payload = action_chunk
    if isinstance(action_chunk, dict):
        if "actions" in action_chunk:
            payload = action_chunk["actions"]
        elif "action" in action_chunk:
            payload = action_chunk["action"]
        elif len(action_chunk) == 1:
            payload = next(iter(action_chunk.values()))
        else:
            raise ValueError(
                f"action_chunk is dict but no obvious action field found. keys={list(action_chunk.keys())}"
            )

    try:
        action_arr = np.asarray(payload, dtype=np.float32)
    except Exception:
        if hasattr(payload, "tolist"):
            action_arr = np.array(payload.tolist(), dtype=np.float32)
        else:
            raise ValueError(f"Cannot convert payload to numpy. payload_type={type(payload)}")

    if action_arr.ndim == 2:
        action_vec = action_arr[0]
    elif action_arr.ndim == 1:
        action_vec = action_arr
    else:
        raise ValueError(f"Unexpected action array shape: {action_arr.shape}")

    if action_vec.shape[0] != 7:
        raise ValueError(f"Expected 7-dim action, got shape {action_vec.shape}")

    env_action = torch.as_tensor(
        action_vec[None, :],
        device=env.unwrapped.device,
        dtype=torch.float32,
    )
    return env_action, action_arr


def get_zero_env_action(env):
    import torch

    try:
        action_shape = env.action_space.shape
        return torch.zeros(action_shape, device=env.unwrapped.device)
    except Exception:
        sample = env.action_space.sample()
        return torch.as_tensor(sample, device=env.unwrapped.device) * 0.0


def is_black(frame: np.ndarray) -> bool:
    arr = np.asarray(frame)
    if arr.size == 0:
        return True
    return int(arr.max()) == 0 and int(arr.min()) == 0


def inject_wrist_camera_into_scene_cfg(env_cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    WRIST_POS = (0.02, -0.02, 0.06)
    WRIST_ROT = (0.700, 0.170, -0.170, -0.690)

    wrist_camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=WRIST_POS,
            rot=WRIST_ROT,
            convention="ros",
        ),
    )

    setattr(env_cfg.scene, "wrist_camera", wrist_camera_cfg)


def get_scene_wrist_rgb(env) -> np.ndarray:
    wrist_camera = env.unwrapped.scene["wrist_camera"]

    if "rgb" not in wrist_camera.data.output:
        raise RuntimeError(
            f"Wrist camera has no rgb output. keys={list(wrist_camera.data.output.keys())}"
        )

    rgb = wrist_camera.data.output["rgb"]

    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    else:
        rgb = np.asarray(rgb)

    if rgb.ndim == 4:
        rgb = rgb[0]

    if rgb.ndim != 3:
        raise ValueError(f"Unexpected wrist rgb ndim: {rgb.ndim}, shape={rgb.shape}")

    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    rgb = rgb.astype(np.uint8)
    return rgb


def build_openpi_obs(
    obs: dict[str, Any],
    prompt: str,
    main_frame: np.ndarray,
    wrist_frame: np.ndarray,
    state_mode: str,
    env=None,
    robot_state_handles=None,
):
    raw_state = obs["policy"][0].detach().float().cpu().numpy().astype(np.float32)
    state8 = map_state8(
        raw_state,
        state_mode,
        env=env,
        robot_state_handles=robot_state_handles,
    )

    main_image = preprocess_frame(main_frame)
    wrist_image = preprocess_frame(wrist_frame)

    openpi_obs = {
        "observation/image": main_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state8,
        "prompt": prompt,
    }
    return openpi_obs, raw_state, state8, main_image, wrist_image


def warmup_until_visible(env, warmup_steps: int):
    zero_action = get_zero_env_action(env)
    obs, info = env.reset()
    main_frame = None
    wrist_frame = None

    for i in range(warmup_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        main_frame = env.render()
        wrist_frame = get_scene_wrist_rgb(env)

        if (not is_black(main_frame)) and (not is_black(wrist_frame)):
            return obs, main_frame, wrist_frame

    return obs, main_frame, wrist_frame


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")


def main() -> int:
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from openpi_client import websocket_client_policy

    if not getattr(args_cli, "enable_cameras", False):
        raise RuntimeError("This script requires --enable_cameras")

    dataset_root = Path(args_cli.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    try:
        env_cfg.viewer.eye = (1.6, 1.2, 1.1)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.35)
        env_cfg.viewer.resolution = (640, 480)
        print("[INFO] Updated viewer eye/lookat/resolution.")
    except Exception as e:
        print("[WARN] Could not modify viewer config:", repr(e))

    inject_wrist_camera_into_scene_cfg(env_cfg)
    try:
        env_cfg.commands.object_pose.debug_vis = False
        print("[INFO] Disabled object_pose debug visualization.")
    except Exception as e:
        print("[WARN] Could not disable object_pose debug_vis:", repr(e))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args_cli.host,
        port=args_cli.port,
    )

    robot_state_handles = resolve_robot_state_handles(env) if args_cli.state_mode == "official" else None
    state8_definition = {
        "zero": "all-zero 8D state",
        "raw8": "raw policy observation first 8 dims",
        "official": "eef_pos_xyz(3) + eef_axis_angle(3) + gripper_qpos(2)",
    }[args_cli.state_mode]

    meta = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "episodes": args_cli.episodes,
        "steps_per_episode": args_cli.steps_per_episode,
        "prompt": args_cli.prompt,
        "state_mode": args_cli.state_mode,
        "state8_definition": state8_definition,
        "host": args_cli.host,
        "port": args_cli.port,
        "created_at_unix": time.time(),
        "notes": "Local intermediate rollout dataset before LeRobot v3 conversion.",
    }
    write_json(dataset_root / "meta.json", meta)

    print("[INFO] Environment created.")
    print("[INFO] Collecting dataset to:", dataset_root)
    print("[INFO] state8 definition:", state8_definition)

    total_steps = 0

    for episode_idx in range(args_cli.episodes):
        print(f"\n[INFO] ===== Episode {episode_idx} / {args_cli.episodes - 1} =====")

        episode_dir = dataset_root / f"episode_{episode_idx:05d}"
        main_dir = episode_dir / "frames_main"
        wrist_dir = episode_dir / "frames_wrist"
        steps_jsonl = episode_dir / "steps.jsonl"

        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)

        obs, current_main_frame, current_wrist_frame = warmup_until_visible(
            env=env,
            warmup_steps=args_cli.warmup_steps,
        )

        episode_summary = {
            "episode_idx": episode_idx,
            "prompt": args_cli.prompt,
            "task": args_cli.task,
            "state_mode": args_cli.state_mode,
            "state8_definition": state8_definition,
            "warmup_steps": args_cli.warmup_steps,
        }
        write_json(episode_dir / "episode_meta.json", episode_summary)

        for step_idx in range(args_cli.steps_per_episode):
            openpi_obs, raw_state, state8, processed_main, processed_wrist = build_openpi_obs(
                obs=obs,
                prompt=args_cli.prompt,
                main_frame=current_main_frame,
                wrist_frame=current_wrist_frame,
                state_mode=args_cli.state_mode,
                env=env,
                robot_state_handles=robot_state_handles,
            )

            main_path = main_dir / f"{step_idx:06d}.png"
            wrist_path = wrist_dir / f"{step_idx:06d}.png"
            save_png(processed_main, main_path)
            save_png(processed_wrist, wrist_path)

            print(f"[INFO] episode={episode_idx} step={step_idx}: before infer")
            action_chunk = client.infer(openpi_obs)
            print(f"[INFO] episode={episode_idx} step={step_idx}: after infer")

            env_action, action_chunk_array = action_chunk_to_env_action(action_chunk, env)
            obs, reward, terminated, truncated, info = env.step(env_action)

            reward_np = reward.detach().cpu().numpy() if hasattr(reward, "detach") else reward
            terminated_np = terminated.detach().cpu().numpy() if hasattr(terminated, "detach") else terminated
            truncated_np = truncated.detach().cpu().numpy() if hasattr(truncated, "detach") else truncated

            current_main_frame = env.render()
            current_wrist_frame = get_scene_wrist_rgb(env)

            record = {
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "prompt": args_cli.prompt,
                "task": args_cli.task,
                "state_mode": args_cli.state_mode,
                "state8_definition": state8_definition,
                "main_image_path": str(main_path),
                "wrist_image_path": str(wrist_path),
                "state_35_raw": raw_state.astype(np.float32),
                "state_8_mapped": state8.astype(np.float32),
                "action_chunk": action_chunk_array.astype(np.float32),
                "env_action_executed": env_action.detach().cpu().numpy().astype(np.float32),
                "reward": reward_np,
                "terminated": terminated_np,
                "truncated": truncated_np,
                "policy_timing": action_chunk.get("policy_timing", None) if isinstance(action_chunk, dict) else None,
                "server_timing": action_chunk.get("server_timing", None) if isinstance(action_chunk, dict) else None,
            }
            append_jsonl(steps_jsonl, record)

            total_steps += 1

            done_flag = bool(np.asarray(terminated_np).any()) or bool(np.asarray(truncated_np).any())
            print(
                f"[INFO] episode={episode_idx} step={step_idx} "
                f"reward={reward_np} terminated={terminated_np} truncated={truncated_np}"
            )

            if done_flag:
                print(f"[INFO] episode={episode_idx}: early stop at step {step_idx}")
                break

    env.close()
    print(f"[SUCCESS] Dataset collection finished. Total saved steps: {total_steps}")
    print(f"[SUCCESS] Output root: {dataset_root}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)