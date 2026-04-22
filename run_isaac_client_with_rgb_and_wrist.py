import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Isaac Lab -> openpi client with RGB + wrist camera (demo client)")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps", type=int, default=150, help="Maximum steps per episode.")
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
parser.add_argument("--debug_save_dir", type=str, default="/root/gpufree-data/isaac_client/debug_frames")
parser.add_argument("--run_out_root", type=str, default="/root/gpufree-data/isaac_client/infer_runs")
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--success_lift_threshold", type=float, default=0.03)
parser.add_argument("--stop_on_success", action="store_true", default=False)
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


def frame_to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Frame is None")

    image = frame
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Unexpected image ndim: {image.ndim}, shape={image.shape}")

    if image.shape[-1] == 4:
        image = image[..., :3]

    return image.astype(np.uint8)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    image = frame_to_uint8_rgb(frame)
    try:
        from PIL import Image

        image = np.array(Image.fromarray(image).resize((224, 224)))
    except Exception as e:
        raise RuntimeError(f"Failed to resize frame to 224x224: {e}")

    return image


def extract_raw_state(obs: dict[str, Any]) -> np.ndarray:
    return obs["policy"][0].detach().float().cpu().numpy().astype(np.float32)


def extract_state8_zero(raw_state: np.ndarray) -> np.ndarray:
    return np.zeros((8,), dtype=np.float32)


def extract_state8_raw8(raw_state: np.ndarray) -> np.ndarray:
    return raw_state[:8].astype(np.float32)


def _get_scene_robot(env):
    scene = env.unwrapped.scene

    candidate_keys = ["robot", "Robot"]
    for key in candidate_keys:
        try:
            return scene[key]
        except Exception:
            pass

    try:
        scene_keys = list(scene.keys())
    except Exception:
        scene_keys = []

    for key in scene_keys:
        if isinstance(key, str) and "robot" in key.lower():
            try:
                return scene[key]
            except Exception:
                pass

    raise RuntimeError(
        "Could not find robot articulation in env.unwrapped.scene. "
        f"Available scene keys: {scene_keys}"
    )


def _resolve_robot_cache(env) -> dict[str, Any]:
    cache_name = "_openpi_libero_state_cache"
    cache = getattr(env.unwrapped, cache_name, None)
    if cache is not None:
        return cache

    robot = _get_scene_robot(env)

    eef_body_ids, eef_body_names = robot.find_bodies("panda_hand", preserve_order=True)
    if len(eef_body_ids) != 1:
        raise RuntimeError(
            "Failed to resolve a unique end-effector body for panda_hand. "
            f"Matched names={eef_body_names}"
        )

    try:
        finger_joint_ids, finger_joint_names = robot.find_joints(
            ["panda_finger_joint1", "panda_finger_joint2"], preserve_order=True
        )
    except TypeError:
        finger_joint_ids, finger_joint_names = robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])

    if len(finger_joint_ids) != 2:
        raise RuntimeError(
            "Failed to resolve both Franka finger joints. "
            f"Matched names={finger_joint_names}"
        )

    cache = {
        "robot": robot,
        "eef_body_id": int(eef_body_ids[0]),
        "eef_body_name": eef_body_names[0],
        "finger_joint_ids": [int(idx) for idx in finger_joint_ids],
        "finger_joint_names": list(finger_joint_names),
    }
    setattr(env.unwrapped, cache_name, cache)
    return cache


def extract_state8_official(raw_state: np.ndarray, env) -> np.ndarray:
    import torch
    import isaaclab.utils.math as math_utils

    cache = _resolve_robot_cache(env)
    robot = cache["robot"]
    eef_body_id = cache["eef_body_id"]
    finger_joint_ids = cache["finger_joint_ids"]

    try:
        eef_pos = robot.data.body_link_pos_w[0, eef_body_id]
        eef_quat = robot.data.body_link_quat_w[0, eef_body_id]
    except Exception:
        eef_state = robot.data.body_link_state_w[0, eef_body_id]
        eef_pos = eef_state[:3]
        eef_quat = eef_state[3:7]

    eef_quat = math_utils.quat_unique(eef_quat.reshape(1, 4))
    eef_axis_angle = math_utils.axis_angle_from_quat(eef_quat).reshape(-1)

    finger_joint_tensor = torch.as_tensor(
        finger_joint_ids,
        device=robot.data.joint_pos.device,
        dtype=torch.long,
    )
    gripper_qpos = robot.data.joint_pos[0, finger_joint_tensor]

    state8 = torch.cat(
        [
            eef_pos.reshape(-1)[:3],
            eef_axis_angle[:3],
            gripper_qpos.reshape(-1)[:2],
        ],
        dim=0,
    )
    return state8.detach().float().cpu().numpy().astype(np.float32)


def map_state8(raw_state: np.ndarray, mode: str, env) -> np.ndarray:
    if mode == "zero":
        return extract_state8_zero(raw_state)
    if mode == "raw8":
        return extract_state8_raw8(raw_state)
    if mode == "official":
        return extract_state8_official(raw_state, env)
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
            pos=(0.02, -0.02, 0.06),
            rot=(0.700, 0.170, -0.170, -0.690),
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

    return rgb.astype(np.uint8)


def build_openpi_obs(
    obs: dict[str, Any],
    prompt: str,
    main_frame: np.ndarray,
    wrist_frame: np.ndarray,
    state_mode: str,
    env,
):
    raw_state = extract_raw_state(obs)
    state8 = map_state8(raw_state, state_mode, env)

    main_image = preprocess_frame(main_frame)
    wrist_image = preprocess_frame(wrist_frame)

    openpi_obs = {
        "observation/image": main_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state8,
        "prompt": prompt,
    }
    return openpi_obs, raw_state, state8, main_image, wrist_image


def warmup_until_visible(env, warmup_steps: int, debug_dir: Path):
    zero_action = get_zero_env_action(env)
    obs, info = env.reset()
    main_frame = None
    wrist_frame = None

    for i in range(warmup_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        main_frame = env.render()
        wrist_frame = get_scene_wrist_rgb(env)

        main_np = np.asarray(main_frame)
        wrist_np = np.asarray(wrist_frame)

        if main_np.ndim == 3:
            save_png(frame_to_uint8_rgb(main_np), debug_dir / f"warmup_main_{i+1}.png")
        if wrist_np.ndim == 3:
            save_png(frame_to_uint8_rgb(wrist_np), debug_dir / f"warmup_wrist_{i+1}.png")

        if (not is_black(main_frame)) and (not is_black(wrist_frame)):
            print(f"[INFO] warmup: got non-black main and wrist frames at step {i+1}")
            return obs, main_frame, wrist_frame

    print("[WARN] warmup: one of the cameras is still black after all warmup steps; proceeding anyway")
    return obs, main_frame, wrist_frame


def make_video_writer(path: Path, fps: int):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, macro_block_size=None)


def summarize_episode(
    episode_idx: int,
    prompt: str,
    steps_executed: int,
    rewards: list[float],
    max_lift_delta_z: float,
    terminated_any: bool,
    truncated_any: bool,
    success_lift_threshold: float,
    episode_dir: Path,
):
    success_by_lift = max_lift_delta_z >= success_lift_threshold
    return {
        "episode_idx": episode_idx,
        "prompt": prompt,
        "steps_executed": steps_executed,
        "reward_last": float(rewards[-1]) if rewards else None,
        "reward_max": float(max(rewards)) if rewards else None,
        "reward_mean": float(sum(rewards) / len(rewards)) if rewards else None,
        "max_lift_delta_z": float(max_lift_delta_z),
        "terminated_any": bool(terminated_any),
        "truncated_any": bool(truncated_any),
        "success_by_lift_threshold": bool(success_by_lift),
        "success_lift_threshold": float(success_lift_threshold),
        "main_video_path": str((episode_dir / "main.mp4").resolve()),
        "wrist_video_path": str((episode_dir / "wrist.mp4").resolve()),
    }


def main() -> int:
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from openpi_client import websocket_client_policy

    debug_dir = Path(args_cli.debug_save_dir)
    run_out_root = Path(args_cli.run_out_root)
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_out_root / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(args_cli, "enable_cameras", False):
        raise RuntimeError("This script requires --enable_cameras")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    try:
        env_cfg.viewer.eye = (2.8, 2.2, 1.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
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

    print("[INFO] Environment created.")
    print("[INFO] Connecting to openpi server at", f"{args_cli.host}:{args_cli.port}")
    print("[INFO] State mode:", args_cli.state_mode)
    print("[INFO] Run dir:", run_dir)

    if args_cli.state_mode == "official":
        cache = _resolve_robot_cache(env)
        print("[INFO] official state = eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2)")
        print("[INFO] EEF body:", cache["eef_body_name"], "id=", cache["eef_body_id"])
        print("[INFO] Finger joints:", cache["finger_joint_names"], "ids=", cache["finger_joint_ids"])

    summary = {
        "task": args_cli.task,
        "host": args_cli.host,
        "port": args_cli.port,
        "prompt": args_cli.prompt,
        "state_mode": args_cli.state_mode,
        "episodes": args_cli.episodes,
        "steps_per_episode": args_cli.steps,
        "video_fps": args_cli.video_fps,
        "success_lift_threshold": args_cli.success_lift_threshold,
        "run_dir": str(run_dir.resolve()),
        "episode_summaries": [],
        "episodes_total": 0,
        "episodes_success": 0,
        "episodes_failed": 0,
        "success_rate": 0.0,
        "max_lift_delta_z_best": 0.0,
        "max_lift_delta_z_mean": 0.0,
        "steps_executed_total": 0,
        "steps_executed_mean": 0.0,
    }

    for episode_idx in range(args_cli.episodes):
        print(f"\n[INFO] ===== Demo episode {episode_idx} / {args_cli.episodes - 1} =====")
        episode_dir = run_dir / f"episode_{episode_idx:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        obs, current_main_frame, current_wrist_frame = warmup_until_visible(
            env=env,
            warmup_steps=args_cli.warmup_steps,
            debug_dir=debug_dir,
        )

        if current_main_frame is None or current_wrist_frame is None:
            raise RuntimeError("Warmup failed to produce camera frames")

        initial_raw_state = extract_raw_state(obs)
        initial_object_z = float(initial_raw_state[20]) if initial_raw_state.shape[0] >= 21 else 0.0
        max_lift_delta_z = 0.0
        rewards = []
        terminated_any = False
        truncated_any = False
        steps_executed = 0

        main_writer = make_video_writer(episode_dir / "main.mp4", args_cli.video_fps)
        wrist_writer = make_video_writer(episode_dir / "wrist.mp4", args_cli.video_fps)

        try:
            for step_idx in range(args_cli.steps):
                openpi_obs, raw_state, state8, processed_main, processed_wrist = build_openpi_obs(
                    obs=obs,
                    prompt=args_cli.prompt,
                    main_frame=current_main_frame,
                    wrist_frame=current_wrist_frame,
                    state_mode=args_cli.state_mode,
                    env=env,
                )

                if episode_idx == 0 and step_idx == 0:
                    save_png(processed_main, debug_dir / "first_main_processed.png")
                    save_png(processed_wrist, debug_dir / "first_wrist_processed.png")

                main_writer.append_data(frame_to_uint8_rgb(current_main_frame))
                wrist_writer.append_data(frame_to_uint8_rgb(current_wrist_frame))

                action_chunk = client.infer(openpi_obs)
                env_action, action_chunk_array = action_chunk_to_env_action(action_chunk, env)

                obs, reward, terminated, truncated, info = env.step(env_action)
                current_main_frame = env.render()
                current_wrist_frame = get_scene_wrist_rgb(env)

                reward_np = reward.detach().cpu().numpy() if hasattr(reward, "detach") else reward
                term_np = terminated.detach().cpu().numpy() if hasattr(terminated, "detach") else terminated
                trunc_np = truncated.detach().cpu().numpy() if hasattr(truncated, "detach") else truncated

                reward_scalar = float(np.asarray(reward_np).reshape(-1)[0]) if np.asarray(reward_np).size else 0.0
                rewards.append(reward_scalar)
                terminated_any = terminated_any or bool(np.asarray(term_np).any())
                truncated_any = truncated_any or bool(np.asarray(trunc_np).any())
                steps_executed = step_idx + 1

                next_raw_state = extract_raw_state(obs)
                if next_raw_state.shape[0] >= 21:
                    current_lift_delta_z = float(next_raw_state[20] - initial_object_z)
                    max_lift_delta_z = max(max_lift_delta_z, current_lift_delta_z)
                else:
                    current_lift_delta_z = 0.0

                print(
                    f"[INFO] episode={episode_idx} step={step_idx + 1}/{args_cli.steps} "
                    f"reward={reward_scalar:.6f} terminated={term_np} truncated={trunc_np} "
                    f"lift_delta_z={current_lift_delta_z:.4f}"
                )

                if args_cli.stop_on_success and max_lift_delta_z >= args_cli.success_lift_threshold:
                    print(
                        f"[INFO] early stop: success_lift_threshold reached "
                        f"(max_lift_delta_z={max_lift_delta_z:.4f})"
                    )
                    break

                if terminated_any or truncated_any:
                    print(f"[INFO] episode ended at step {step_idx}")
                    break
        finally:
            main_writer.close()
            wrist_writer.close()

        episode_summary = summarize_episode(
            episode_idx=episode_idx,
            prompt=args_cli.prompt,
            steps_executed=steps_executed,
            rewards=rewards,
            max_lift_delta_z=max_lift_delta_z,
            terminated_any=terminated_any,
            truncated_any=truncated_any,
            success_lift_threshold=args_cli.success_lift_threshold,
            episode_dir=episode_dir,
        )
        write_json(episode_dir / "summary.json", episode_summary)
        summary["episode_summaries"].append(episode_summary)

    env.close()

    episode_summaries = summary["episode_summaries"]
    summary["episodes_total"] = len(episode_summaries)
    summary["episodes_success"] = int(sum(1 for ep in episode_summaries if ep.get("success_by_lift_threshold", False)))
    summary["episodes_failed"] = int(summary["episodes_total"] - summary["episodes_success"])
    summary["success_rate"] = (
        float(summary["episodes_success"] / summary["episodes_total"])
        if summary["episodes_total"] > 0
        else 0.0
    )
    summary["max_lift_delta_z_best"] = (
        float(max(ep.get("max_lift_delta_z", 0.0) for ep in episode_summaries))
        if episode_summaries
        else 0.0
    )
    summary["max_lift_delta_z_mean"] = (
        float(sum(ep.get("max_lift_delta_z", 0.0) for ep in episode_summaries) / len(episode_summaries))
        if episode_summaries
        else 0.0
    )
    summary["steps_executed_total"] = int(sum(ep.get("steps_executed", 0) for ep in episode_summaries))
    summary["steps_executed_mean"] = (
        float(summary["steps_executed_total"] / summary["episodes_total"])
        if summary["episodes_total"] > 0
        else 0.0
    )

    write_json(run_dir / "summary.json", summary)
    print("[EVAL]")
    print(f"episodes_total   = {summary['episodes_total']}")
    print(f"episodes_success = {summary['episodes_success']}")
    print(f"episodes_failed  = {summary['episodes_failed']}")
    print(f"success_rate     = {summary['success_rate'] * 100:.2f}%")
    print(f"lift_best        = {summary['max_lift_delta_z_best']:.4f}")
    print(f"lift_mean        = {summary['max_lift_delta_z_mean']:.4f}")
    print(f"steps_mean       = {summary['steps_executed_mean']:.2f}")
    print(f"run_dir          = {run_dir}")
    print("[SUCCESS] Demo client finished.")
    print("[SUCCESS] Run dir:", run_dir)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)