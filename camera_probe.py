import argparse
import sys
from pathlib import Path
import os
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe RGB rendering from Isaac Lab viewport")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=5)
from datetime import datetime

default_path = f"/root/gpufree-data/isaac_client/camera_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
parser.add_argument("--save_path", type=str, default=default_path)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def save_image(img: np.ndarray, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        Image.fromarray(img).save(path)
        print(f"[INFO] Saved image with PIL to: {path}")
        return
    except Exception as e:
        print("[WARN] PIL save failed:", repr(e))

    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
        print(f"[INFO] Saved image with imageio to: {path}")
        return
    except Exception as e:
        print("[WARN] imageio save failed:", repr(e))

    npy_path = str(path) + ".npy"
    np.save(npy_path, img)
    print(f"[WARN] Could not save PNG. Saved raw numpy array to: {npy_path}")


def main() -> int:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    
    try:
        env_cfg.viewer.eye = (2.5, 2.5, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        env_cfg.viewer.resolution = (640, 480)
        print("[INFO] Updated viewer eye/lookat/resolution.")
    except Exception as e:
        print("[WARN] Could not modify viewer config:", repr(e))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    print("[INFO] Environment created with render_mode='rgb_array'.")

    obs, info = env.reset()
    print("[INFO] Reset done.")
    print("[INFO] obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))

    # Step a few times so the renderer has a stable frame
    try:
        action_shape = env.action_space.shape
        actions = torch.zeros(action_shape, device=env.unwrapped.device)
    except Exception:
        sample = env.action_space.sample()
        actions = torch.as_tensor(sample, device=env.unwrapped.device)

    frame = None
    for i in range(args_cli.steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        frame = env.render()
        print(f"[INFO] step={i+1}/{args_cli.steps}, render type={type(frame)}")

    if frame is None:
        print("[ERROR] env.render() returned None.")
        env.close()
        return 1

    if not isinstance(frame, np.ndarray):
        print("[ERROR] env.render() did not return a numpy array.")
        print("[ERROR] got:", type(frame))
        env.close()
        return 1

    print("[INFO] frame shape:", frame.shape, "dtype:", frame.dtype)
    sys.stdout.flush()

    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
        print("[INFO] Dropped alpha channel. New shape:", frame.shape)
        sys.stdout.flush()

    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"/root/gpufree-data/isaac_client/camera_probe_{ts}.png"
    print("[INFO] About to save image to:", save_path)
    sys.stdout.flush()

    save_image(frame, save_path)

    print("[INFO] save_image() returned")
    sys.stdout.flush()

    print("[INFO] About to close env")
    sys.stdout.flush()
    env.close()

    print("[SUCCESS] Camera probe finished.")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)