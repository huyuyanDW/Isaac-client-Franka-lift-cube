import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect scripted expert rollout dataset from Isaac Lab")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_attempts", type=int, default=100)
parser.add_argument("--steps_per_episode", type=int, default=120)
parser.add_argument("--prompt", type=str, default="lift the cube")
parser.add_argument("--state-mode", type=str, default="official", choices=["zero", "raw8", "official"])
parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="/root/gpufree-data/isaac_client/datasets/scripted_rollout_dataset_v1",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--save_failed", action="store_true", default=False)
parser.add_argument("--approach_height", type=float, default=0.08)
parser.add_argument("--grasp_height_offset", type=float, default=-0.012)
parser.add_argument("--close_trigger_z_tol", type=float, default=0.012)
parser.add_argument("--close_trigger_xy_tol", type=float, default=0.008)
parser.add_argument("--grasp_forward_bias_xy", type=float, default=0.008)
parser.add_argument("--pregrasp_backoff", type=float, default=0.07)
parser.add_argument("--close_forward_offset", type=float, default=0.0)
parser.add_argument("--lift_height", type=float, default=0.16)
parser.add_argument("--xy_tolerance", type=float, default=0.015)
parser.add_argument("--z_tolerance", type=float, default=0.015)
parser.add_argument("--pos_gain", type=float, default=8.0)
parser.add_argument("--max_pos_action", type=float, default=0.18)
parser.add_argument("--close_steps", type=int, default=16)
parser.add_argument("--descend_force_close_steps", type=int, default=6)
parser.add_argument("--settle_steps", type=int, default=4)
parser.add_argument("--settle_min_steps", type=int, default=1)
parser.add_argument("--settle_down_hold_z", type=float, default=-0.0002)
parser.add_argument("--gripper_closed_sum_threshold", type=float, default=0.03)
parser.add_argument("--gripper_motion_eps", type=float, default=0.0015)
parser.add_argument("--gripper_stable_steps", type=int, default=4)
parser.add_argument("--gripper_open_action", type=float, default=1.0)
parser.add_argument("--gripper_close_action", type=float, default=-1.0)
parser.add_argument("--success_lift_delta", type=float, default=0.06)
parser.add_argument("--success_reward_threshold", type=float, default=5.0)
parser.add_argument("--abort_object_xy_shift", type=float, default=0.08)
parser.add_argument("--abort_object_drop_delta", type=float, default=0.03)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

STATE8_DEFINITION_OFFICIAL = "eef_pos_xyz(3) + eef_axis_angle(3) + gripper_qpos(2)"


@dataclass
class RobotHandles:
    robot: Any
    eef_body_name: str
    eef_body_index: int
    left_finger_body_name: str
    left_finger_body_index: int
    right_finger_body_name: str
    right_finger_body_index: int
    finger_joint_names: list[str]
    finger_joint_indices: list[int]


@dataclass
class EpisodeContext:
    initial_object_pos_root: np.ndarray
    last_object_pos_root: np.ndarray
    initial_object_z: float
    stage: str = "pregrasp_high"
    close_countdown: int = 0
    descend_counter: int = 0
    settle_countdown: int = 0
    grasp_hand_to_center_root: np.ndarray | None = None
    grasp_approach_dir_root: np.ndarray | None = None
    close_hold_target_root: np.ndarray | None = None
    close_last_gripper_sum: float | None = None
    close_stable_counter: int = 0
    success: bool = False
    success_reason: str = ""


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


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
    image = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
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


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float32).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float32).reshape(4)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    out = quat_multiply_wxyz(quat_multiply_wxyz(q, vq), quat_conjugate_wxyz(q))
    return out[1:4]


def world_to_root_frame(pos_w: np.ndarray, root_pos_w: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    rel = np.asarray(pos_w, dtype=np.float32) - np.asarray(root_pos_w, dtype=np.float32)
    q_inv = quat_conjugate_wxyz(root_quat_wxyz)
    return quat_rotate_wxyz(q_inv, rel).astype(np.float32)


def quat_wxyz_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    quat = quat / norm
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    axis = xyz / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    return (axis * angle).astype(np.float32)


def extract_state8_zero(raw_state: np.ndarray, env=None, handles=None) -> np.ndarray:
    return np.zeros((8,), dtype=np.float32)


def extract_state8_raw8(raw_state: np.ndarray, env=None, handles=None) -> np.ndarray:
    return raw_state[:8].astype(np.float32)


def find_robot_handles(env) -> RobotHandles:
    scene = env.unwrapped.scene
    robot = None
    robot_key = None
    for candidate in ["robot", "Robot"]:
        try:
            robot = scene[candidate]
            robot_key = candidate
            break
        except Exception:
            continue
    if robot is None:
        raise RuntimeError(f"Could not find robot in scene. Available keys: {list(scene.keys())}")

    eef_candidates = ["panda_hand", "hand", "ee_link", "tool0"]
    eef_body_name = None
    eef_body_index = None
    for body_name in eef_candidates:
        try:
            indices, names = robot.find_bodies(body_name)
            if len(indices) > 0:
                eef_body_name = str(names[0])
                eef_body_index = int(indices[0])
                break
        except Exception:
            continue
    if eef_body_index is None:
        raise RuntimeError("Could not resolve end-effector body.")

    finger_body_candidates = [
        ["panda_leftfinger", "panda_rightfinger"],
        ["leftfinger", "rightfinger"],
        ["left_finger", "right_finger"],
    ]
    left_finger_body_name = None
    left_finger_body_index = None
    right_finger_body_name = None
    right_finger_body_index = None
    for body_names in finger_body_candidates:
        try:
            indices, names = robot.find_bodies(body_names)
            if len(indices) == 2:
                left_finger_body_name = str(names[0])
                left_finger_body_index = int(indices[0])
                right_finger_body_name = str(names[1])
                right_finger_body_index = int(indices[1])
                break
        except Exception:
            continue
    if left_finger_body_index is None or right_finger_body_index is None:
        raise RuntimeError("Could not resolve finger bodies.")

    finger_joint_candidates = [
        ["panda_finger_joint1", "panda_finger_joint2"],
        ["finger_joint1", "finger_joint2"],
        ["left_finger_joint", "right_finger_joint"],
    ]
    finger_joint_names = None
    finger_joint_indices = None
    for joint_names in finger_joint_candidates:
        try:
            indices, names = robot.find_joints(joint_names)
            if len(indices) == 2:
                finger_joint_names = [str(names[0]), str(names[1])]
                finger_joint_indices = [int(indices[0]), int(indices[1])]
                break
        except Exception:
            continue
    if finger_joint_indices is None:
        raise RuntimeError("Could not resolve gripper finger joints.")

    print("[INFO] Resolved robot handles:")
    print(f"[INFO]   robot key          = {robot_key}")
    print(f"[INFO]   eef body          = {eef_body_name} (index={eef_body_index})")
    print(
        f"[INFO]   finger bodies      = {left_finger_body_name}/{right_finger_body_name} "
        f"(indices={left_finger_body_index}/{right_finger_body_index})"
    )
    print(f"[INFO]   finger joints      = {finger_joint_names} (indices={finger_joint_indices})")

    return RobotHandles(
        robot=robot,
        eef_body_name=eef_body_name,
        eef_body_index=eef_body_index,
        left_finger_body_name=left_finger_body_name,
        left_finger_body_index=left_finger_body_index,
        right_finger_body_name=right_finger_body_name,
        right_finger_body_index=right_finger_body_index,
        finger_joint_names=finger_joint_names,
        finger_joint_indices=finger_joint_indices,
    )


def read_root_pose_w(robot) -> tuple[np.ndarray, np.ndarray]:
    data = robot.data
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        return _to_numpy(data.root_pos_w)[0].astype(np.float32), _to_numpy(data.root_quat_w)[0].astype(np.float32)
    if hasattr(data, "root_state_w"):
        rs = _to_numpy(data.root_state_w)
        return rs[0, 0:3].astype(np.float32), rs[0, 3:7].astype(np.float32)
    raise RuntimeError("Could not read robot root pose from robot.data")


def read_eef_pos_quat_and_gripper(env, handles: RobotHandles) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = handles.robot
    data = robot.data
    pos = None
    quat = None
    if hasattr(data, "body_link_pos_w") and hasattr(data, "body_link_quat_w"):
        pos = _to_numpy(data.body_link_pos_w)[0, handles.eef_body_index].astype(np.float32)
        quat = _to_numpy(data.body_link_quat_w)[0, handles.eef_body_index].astype(np.float32)
    elif hasattr(data, "body_state_w"):
        bs = _to_numpy(data.body_state_w)
        pos = bs[0, handles.eef_body_index, 0:3].astype(np.float32)
        quat = bs[0, handles.eef_body_index, 3:7].astype(np.float32)
    else:
        raise RuntimeError("Could not read EEF pose from robot.data")
    joint_pos = _to_numpy(robot.data.joint_pos)
    gripper_qpos = joint_pos[0, handles.finger_joint_indices].astype(np.float32)
    return pos, quat, gripper_qpos


def read_eef_pos_root(env, handles: RobotHandles) -> np.ndarray:
    eef_pos_w, _, _ = read_eef_pos_quat_and_gripper(env, handles)
    root_pos_w, root_quat_wxyz = read_root_pose_w(handles.robot)
    return world_to_root_frame(eef_pos_w, root_pos_w, root_quat_wxyz)


def read_finger_body_positions_root(env, handles: RobotHandles) -> tuple[np.ndarray, np.ndarray]:
    robot = handles.robot
    data = robot.data
    if hasattr(data, "body_link_pos_w"):
        raw = _to_numpy(data.body_link_pos_w)
        left_pos_w = raw[0, handles.left_finger_body_index].astype(np.float32)
        right_pos_w = raw[0, handles.right_finger_body_index].astype(np.float32)
    elif hasattr(data, "body_state_w"):
        raw = _to_numpy(data.body_state_w)
        left_pos_w = raw[0, handles.left_finger_body_index, 0:3].astype(np.float32)
        right_pos_w = raw[0, handles.right_finger_body_index, 0:3].astype(np.float32)
    else:
        raise RuntimeError("Could not read finger body positions from robot.data")
    root_pos_w, root_quat_wxyz = read_root_pose_w(robot)
    left_root = world_to_root_frame(left_pos_w, root_pos_w, root_quat_wxyz)
    right_root = world_to_root_frame(right_pos_w, root_pos_w, root_quat_wxyz)
    return left_root, right_root


def read_grasp_geometry_root(env, handles: RobotHandles) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eef_pos_root = read_eef_pos_root(env, handles)
    left_root, right_root = read_finger_body_positions_root(env, handles)
    grasp_center_root = 0.5 * (left_root + right_root)
    hand_to_center = grasp_center_root - eef_pos_root
    hand_to_center_norm = float(np.linalg.norm(hand_to_center))
    if hand_to_center_norm <= 1e-6:
        approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        approach_dir = (hand_to_center / hand_to_center_norm).astype(np.float32)
    return eef_pos_root.astype(np.float32), grasp_center_root.astype(np.float32), hand_to_center.astype(np.float32), approach_dir.astype(np.float32)


def extract_state8_official(raw_state: np.ndarray, env, handles: RobotHandles) -> np.ndarray:
    eef_pos, eef_quat_wxyz, gripper_qpos = read_eef_pos_quat_and_gripper(env, handles)
    eef_axis_angle = quat_wxyz_to_axis_angle(eef_quat_wxyz)
    return np.concatenate([eef_pos, eef_axis_angle, gripper_qpos], axis=0).astype(np.float32)


def map_state8(raw_state: np.ndarray, mode: str, env, handles: RobotHandles) -> np.ndarray:
    if mode == "zero":
        return extract_state8_zero(raw_state, env=env, handles=handles)
    if mode == "raw8":
        return extract_state8_raw8(raw_state, env=env, handles=handles)
    if mode == "official":
        return extract_state8_official(raw_state, env=env, handles=handles)
    raise ValueError(f"Unknown state mode: {mode}")


def is_black(frame: np.ndarray) -> bool:
    arr = np.asarray(frame)
    return arr.size == 0 or (int(arr.max()) == 0 and int(arr.min()) == 0)


def get_zero_env_action(env):
    import torch
    try:
        return torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    except Exception:
        sample = env.action_space.sample()
        return torch.as_tensor(sample, device=env.unwrapped.device) * 0.0


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
        raise RuntimeError(f"Wrist camera has no rgb output. keys={list(wrist_camera.data.output.keys())}")
    rgb = _to_numpy(wrist_camera.data.output["rgb"])
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.ndim != 3:
        raise ValueError(f"Unexpected wrist rgb ndim: {rgb.ndim}, shape={rgb.shape}")
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8)


def build_record_payload(obs: dict[str, Any], prompt: str, main_frame: np.ndarray, wrist_frame: np.ndarray, state_mode: str, env, handles: RobotHandles):
    raw_state = obs["policy"][0].detach().float().cpu().numpy().astype(np.float32)
    state8 = map_state8(raw_state, state_mode, env=env, handles=handles)
    main_image = preprocess_frame(main_frame)
    wrist_image = preprocess_frame(wrist_frame)
    record_extra = {"state8_definition": STATE8_DEFINITION_OFFICIAL if state_mode == "official" else state_mode}
    if state_mode == "official":
        eef_pos, eef_quat_wxyz, gripper_qpos = read_eef_pos_quat_and_gripper(env, handles)
        record_extra.update({
            "eef_pos": eef_pos.astype(np.float32),
            "eef_quat_wxyz": eef_quat_wxyz.astype(np.float32),
            "eef_axis_angle": quat_wxyz_to_axis_angle(eef_quat_wxyz).astype(np.float32),
            "gripper_qpos": gripper_qpos.astype(np.float32),
        })
    return raw_state, state8, main_image, wrist_image, record_extra


def warmup_until_visible(env, warmup_steps: int):
    zero_action = get_zero_env_action(env)
    obs, info = env.reset()
    main_frame = None
    wrist_frame = None
    for _ in range(warmup_steps):
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


def clip_vec(v: np.ndarray, max_abs: float) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.float32), -max_abs, max_abs).astype(np.float32)


def object_pos_from_raw_state(raw_state: np.ndarray) -> np.ndarray:
    return raw_state[18:21].astype(np.float32)


def scripted_policy_action(obs: dict[str, Any], env, handles: RobotHandles, ctx: EpisodeContext) -> tuple[np.ndarray, dict]:
    raw_state = obs["policy"][0].detach().float().cpu().numpy().astype(np.float32)
    obj_pos_root = object_pos_from_raw_state(raw_state)
    eef_pos_root, grasp_center_root, hand_to_center_root, approach_dir_root = read_grasp_geometry_root(env, handles)
    _, _, gripper_qpos = read_eef_pos_quat_and_gripper(env, handles)
    gripper_sum = float(np.sum(np.abs(gripper_qpos)))
    ctx.last_object_pos_root = obj_pos_root.copy()

    desired_grasp_center = obj_pos_root.copy()
    desired_grasp_center[2] = obj_pos_root[2] + args_cli.grasp_height_offset
    approach_xy = approach_dir_root.copy()
    approach_xy[2] = 0.0
    approach_xy_norm = float(np.linalg.norm(approach_xy[:2]))
    if approach_xy_norm > 1e-6:
        approach_xy = approach_xy / approach_xy_norm
        desired_grasp_center[:2] = desired_grasp_center[:2] + approach_xy[:2] * float(args_cli.grasp_forward_bias_xy)

    pregrasp_high_center = desired_grasp_center.copy()
    pregrasp_high_center[2] += args_cli.approach_height

    pregrasp_eef = pregrasp_high_center - hand_to_center_root - approach_dir_root * args_cli.pregrasp_backoff
    descend_eef = desired_grasp_center - hand_to_center_root - approach_dir_root * args_cli.pregrasp_backoff

    xy_shift = float(np.linalg.norm(obj_pos_root[:2] - ctx.initial_object_pos_root[:2]))
    drop_delta = float(ctx.initial_object_z - obj_pos_root[2])
    abort_due_to_xy = xy_shift >= float(args_cli.abort_object_xy_shift)
    abort_due_to_drop = (
        ctx.stage in {"close", "settle", "lift"}
        and drop_delta >= float(args_cli.abort_object_drop_delta)
    )
    if abort_due_to_xy or abort_due_to_drop:
        ctx.stage = "abort"

    gripper = args_cli.gripper_open_action
    if ctx.stage == "pregrasp_high":
        target = pregrasp_eef
        pos_err = target - eef_pos_root
        xy_err = float(np.linalg.norm(pos_err[:2]))
        z_err = float(abs(pos_err[2]))
        if xy_err <= args_cli.xy_tolerance * 2.0 and z_err <= args_cli.approach_height * 0.4:
            ctx.stage = "descend"
            ctx.descend_counter = 0

    elif ctx.stage == "descend":
        target = descend_eef
        pos_err = target - eef_pos_root
        xy_err = float(np.linalg.norm(pos_err[:2]))
        eef_z_err = float(abs(eef_pos_root[2] - descend_eef[2]))
        grasp_center_z_err = float(abs(grasp_center_root[2] - desired_grasp_center[2]))
        ctx.descend_counter += 1
        close_ready_z = (
            min(eef_z_err, grasp_center_z_err) <= float(args_cli.close_trigger_z_tol)
            or eef_pos_root[2] <= descend_eef[2] + float(args_cli.close_trigger_z_tol) * 1.25
        )
        close_ready_xy = xy_err <= float(args_cli.close_trigger_xy_tol)
        force_close = (
            ctx.descend_counter >= int(args_cli.descend_force_close_steps)
            and xy_err <= float(args_cli.close_trigger_xy_tol) * 0.9
        )
        if (close_ready_xy and close_ready_z) or force_close:
            ctx.stage = "close"
            ctx.close_countdown = int(args_cli.close_steps)
            ctx.descend_counter = 0
            ctx.grasp_hand_to_center_root = hand_to_center_root.copy()
            ctx.grasp_approach_dir_root = approach_dir_root.copy()
            ctx.close_hold_target_root = eef_pos_root.copy()
            ctx.close_last_gripper_sum = gripper_sum
            ctx.close_stable_counter = 0

    elif ctx.stage == "close":
        target = ctx.close_hold_target_root.copy() if ctx.close_hold_target_root is not None else eef_pos_root.copy()
        target[2] = target[2] - 0.0015
        gripper = args_cli.gripper_close_action
        pos_err = target - eef_pos_root
        if ctx.close_last_gripper_sum is not None and abs(gripper_sum - ctx.close_last_gripper_sum) <= float(args_cli.gripper_motion_eps):
            ctx.close_stable_counter += 1
        else:
            ctx.close_stable_counter = 0
        ctx.close_last_gripper_sum = gripper_sum
        ctx.close_countdown -= 1
        if ctx.close_countdown <= 0:
            ctx.stage = "settle"
            ctx.settle_countdown = int(args_cli.settle_steps)

    elif ctx.stage == "settle":
        target = ctx.close_hold_target_root.copy() if ctx.close_hold_target_root is not None else eef_pos_root.copy()
        target[2] = target[2] + float(args_cli.settle_down_hold_z)
        gripper = args_cli.gripper_close_action
        pos_err = target - eef_pos_root
        if ctx.close_last_gripper_sum is not None and abs(gripper_sum - ctx.close_last_gripper_sum) <= float(args_cli.gripper_motion_eps):
            ctx.close_stable_counter += 1
        else:
            ctx.close_stable_counter = 0
        ctx.close_last_gripper_sum = gripper_sum
        ctx.settle_countdown -= 1
        gripper_closed = gripper_sum <= float(args_cli.gripper_closed_sum_threshold)
        gripper_stable = ctx.close_stable_counter >= int(args_cli.gripper_stable_steps)
        settle_elapsed = int(args_cli.settle_steps) - int(ctx.settle_countdown)
        if settle_elapsed >= int(args_cli.settle_min_steps) and (gripper_closed or gripper_stable):
            ctx.stage = "lift"
        elif ctx.settle_countdown <= 0:
            ctx.stage = "lift"

    elif ctx.stage == "lift":
        frozen_hand_to_center = ctx.grasp_hand_to_center_root.copy() if ctx.grasp_hand_to_center_root is not None else hand_to_center_root.copy()
        desired_lift_center = obj_pos_root.copy()
        desired_lift_center[2] = ctx.initial_object_z + args_cli.lift_height
        target = desired_lift_center - frozen_hand_to_center
        gripper = args_cli.gripper_close_action
        pos_err = target - eef_pos_root
        lift_delta = float(obj_pos_root[2] - ctx.initial_object_z)
        if lift_delta >= float(args_cli.success_lift_delta):
            ctx.success = True
            ctx.success_reason = f"object_lift_delta={lift_delta:.4f}"

    elif ctx.stage == "abort":
        target = eef_pos_root.copy()
        gripper = args_cli.gripper_open_action
        pos_err = np.zeros((3,), dtype=np.float32)

    else:
        raise ValueError(f"Unknown stage: {ctx.stage}")

    pos_cmd = clip_vec(args_cli.pos_gain * pos_err, args_cli.max_pos_action)
    rot_cmd = np.zeros((3,), dtype=np.float32)
    action_vec = np.concatenate([pos_cmd, rot_cmd, np.array([gripper], dtype=np.float32)], axis=0).astype(np.float32)
    debug = {
        "stage": ctx.stage,
        "eef_pos_root": eef_pos_root.astype(np.float32),
        "grasp_center_root": grasp_center_root.astype(np.float32),
        "target_pos_root": target.astype(np.float32),
        "desired_grasp_center_root": desired_grasp_center.astype(np.float32),
        "object_pos_root": obj_pos_root.astype(np.float32),
        "approach_dir_root": approach_dir_root.astype(np.float32),
        "grasp_height_offset": float(args_cli.grasp_height_offset),
        "close_trigger_z_tol": float(args_cli.close_trigger_z_tol),
        "close_trigger_xy_tol": float(args_cli.close_trigger_xy_tol),
        "grasp_forward_bias_xy": float(args_cli.grasp_forward_bias_xy),
        "xy_err": float(xy_err) if 'xy_err' in locals() else 0.0,
        "close_trigger_xy_tol": float(args_cli.close_trigger_xy_tol),
        "hand_to_center_root": hand_to_center_root.astype(np.float32),
        "gripper_qpos": gripper_qpos.astype(np.float32),
        "gripper_sum": gripper_sum,
        "close_countdown": int(ctx.close_countdown),
        "descend_counter": int(ctx.descend_counter),
        "settle_countdown": int(ctx.settle_countdown),
        "close_stable_counter": int(ctx.close_stable_counter),
        "pos_err": pos_err.astype(np.float32),
        "xy_err": float(np.linalg.norm(pos_err[:2])),
        "lift_delta_z": float(obj_pos_root[2] - ctx.initial_object_z),
        "object_xy_shift": xy_shift,
        "object_drop_delta": drop_delta,
    }
    return action_vec, debug


def action_vec_to_env_action(action_vec: np.ndarray, env):
    import torch
    action_vec = np.asarray(action_vec, dtype=np.float32).reshape(7)
    return torch.as_tensor(action_vec[None, :], device=env.unwrapped.device, dtype=torch.float32)


def info_contains_success(info: Any) -> bool | None:
    if isinstance(info, dict):
        for key in ["success", "is_success", "task_success"]:
            if key in info:
                arr = np.asarray(info[key])
                if arr.size:
                    return bool(arr.any())
    return None


def evaluate_episode_success(records: list[dict], ctx: EpisodeContext) -> tuple[bool, str]:
    if ctx.success:
        return True, ctx.success_reason
    if not records:
        return False, "empty_episode"
    last = records[-1]
    info_success = last.get("info_success", None)
    if info_success is True:
        return True, "info_success"
    reward = np.asarray(last.get("reward", 0.0), dtype=np.float32)
    max_reward = float(reward.max()) if reward.size else 0.0
    if max_reward >= float(args_cli.success_reward_threshold):
        return True, f"reward_threshold={max_reward:.4f}"
    lift_delta = float(ctx.last_object_pos_root[2] - ctx.initial_object_z)
    if lift_delta >= float(args_cli.success_lift_delta):
        return True, f"object_lift_delta={lift_delta:.4f}"
    return False, f"no_success_signal lift_delta={lift_delta:.4f} reward={max_reward:.4f}"


def main() -> int:
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg

    if not getattr(args_cli, "enable_cameras", False):
        raise RuntimeError("This script requires --enable_cameras")
    if args_cli.num_envs != 1:
        raise RuntimeError("This script currently only supports --num_envs 1.")

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
    except Exception:
        pass
    inject_wrist_camera_into_scene_cfg(env_cfg)
    try:
        env_cfg.commands.object_pose.debug_vis = False
    except Exception:
        pass

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    handles = find_robot_handles(env)

    meta = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "episodes_target": args_cli.episodes,
        "max_attempts": args_cli.max_attempts,
        "steps_per_episode": args_cli.steps_per_episode,
        "prompt": args_cli.prompt,
        "state_mode": args_cli.state_mode,
        "state8_definition": STATE8_DEFINITION_OFFICIAL if args_cli.state_mode == "official" else args_cli.state_mode,
        "approach_height": args_cli.approach_height,
        "grasp_height_offset": args_cli.grasp_height_offset,
        "close_trigger_xy_tol": args_cli.close_trigger_xy_tol,
        "pregrasp_backoff": args_cli.pregrasp_backoff,
        "close_forward_offset": args_cli.close_forward_offset,
        "lift_height": args_cli.lift_height,
        "xy_tolerance": args_cli.xy_tolerance,
        "z_tolerance": args_cli.z_tolerance,
        "pos_gain": args_cli.pos_gain,
        "max_pos_action": args_cli.max_pos_action,
        "close_steps": args_cli.close_steps,
        "descend_force_close_steps": args_cli.descend_force_close_steps,
        "settle_steps": args_cli.settle_steps,
        "settle_min_steps": args_cli.settle_min_steps,
        "settle_down_hold_z": args_cli.settle_down_hold_z,
        "settle_min_steps": args_cli.settle_min_steps,
        "settle_down_hold_z": args_cli.settle_down_hold_z,
        "gripper_closed_sum_threshold": args_cli.gripper_closed_sum_threshold,
        "gripper_motion_eps": args_cli.gripper_motion_eps,
        "gripper_stable_steps": args_cli.gripper_stable_steps,
        "abort_object_xy_shift": args_cli.abort_object_xy_shift,
        "abort_object_drop_delta": args_cli.abort_object_drop_delta,
        "gripper_open_action": args_cli.gripper_open_action,
        "gripper_close_action": args_cli.gripper_close_action,
        "success_lift_delta": args_cli.success_lift_delta,
        "success_reward_threshold": args_cli.success_reward_threshold,
        "created_at_unix": time.time(),
        "notes": "Scripted expert rollout dataset before LeRobot v3 conversion.",
    }
    write_json(dataset_root / "meta.json", meta)

    kept_episodes = 0
    attempted_episodes = 0
    total_saved_steps = 0

    while kept_episodes < args_cli.episodes and attempted_episodes < args_cli.max_attempts:
        episode_idx = kept_episodes
        attempt_idx = attempted_episodes
        attempted_episodes += 1
        print(f"\n[INFO] ===== Scripted attempt {attempt_idx} / keep target {args_cli.episodes} =====")

        episode_dir = dataset_root / f"episode_{episode_idx:05d}"
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        main_dir = episode_dir / "frames_main"
        wrist_dir = episode_dir / "frames_wrist"
        steps_jsonl = episode_dir / "steps.jsonl"

        obs, current_main_frame, current_wrist_frame = warmup_until_visible(env, args_cli.warmup_steps)
        initial_raw_state = obs["policy"][0].detach().float().cpu().numpy().astype(np.float32)
        initial_object_pos_root = object_pos_from_raw_state(initial_raw_state)
        ctx = EpisodeContext(
            initial_object_pos_root=initial_object_pos_root.copy(),
            last_object_pos_root=initial_object_pos_root.copy(),
            initial_object_z=float(initial_object_pos_root[2]),
        )

        write_json(episode_dir / "episode_meta.json", {
            "episode_idx": episode_idx,
            "attempt_idx": attempt_idx,
            "prompt": args_cli.prompt,
            "task": args_cli.task,
            "state_mode": args_cli.state_mode,
            "state8_definition": STATE8_DEFINITION_OFFICIAL if args_cli.state_mode == "official" else args_cli.state_mode,
            "warmup_steps": args_cli.warmup_steps,
            "controller": "scripted_state_machine_close_lower_and_drop_abort_fix",
        })

        records = []
        interrupted = False
        try:
            for step_idx in range(args_cli.steps_per_episode):
                raw_state, state8, processed_main, processed_wrist, record_extra = build_record_payload(
                    obs, args_cli.prompt, current_main_frame, current_wrist_frame, args_cli.state_mode, env, handles
                )
                main_path = main_dir / f"{step_idx:06d}.png"
                wrist_path = wrist_dir / f"{step_idx:06d}.png"
                save_png(processed_main, main_path)
                save_png(processed_wrist, wrist_path)

                action_vec, debug = scripted_policy_action(obs, env, handles, ctx)
                env_action = action_vec_to_env_action(action_vec, env)
                obs, reward, terminated, truncated, info = env.step(env_action)
                current_main_frame = env.render()
                current_wrist_frame = get_scene_wrist_rgb(env)

                reward_np = reward.detach().cpu().numpy() if hasattr(reward, "detach") else reward
                terminated_np = terminated.detach().cpu().numpy() if hasattr(terminated, "detach") else terminated
                truncated_np = truncated.detach().cpu().numpy() if hasattr(truncated, "detach") else truncated
                info_success = info_contains_success(info)

                record = {
                    "episode_idx": episode_idx,
                    "attempt_idx": attempt_idx,
                    "step_idx": step_idx,
                    "prompt": args_cli.prompt,
                    "task": args_cli.task,
                    "state_mode": args_cli.state_mode,
                    "state8_definition": record_extra["state8_definition"],
                    "main_image_path": str(main_path),
                    "wrist_image_path": str(wrist_path),
                    "state_35_raw": raw_state.astype(np.float32),
                    "state_8_mapped": state8.astype(np.float32),
                    "scripted_action": action_vec.astype(np.float32),
                    "action_chunk": action_vec.reshape(1, 7).astype(np.float32),
                    "env_action_executed": env_action.detach().cpu().numpy().astype(np.float32),
                    "reward": reward_np,
                    "terminated": terminated_np,
                    "truncated": truncated_np,
                    "info_success": info_success,
                    "script_stage": debug["stage"],
                    "eef_pos_root": debug["eef_pos_root"],
                    "target_pos_root": debug["target_pos_root"],
                    "grasp_center_root": debug["grasp_center_root"],
                    "desired_grasp_center_root": debug["desired_grasp_center_root"],
                    "object_pos_root": debug["object_pos_root"],
                    "approach_dir_root": debug["approach_dir_root"],
                    "hand_to_center_root": debug["hand_to_center_root"],
                    "pos_err": debug["pos_err"],
                    "lift_delta_z": debug["lift_delta_z"],
                    "object_xy_shift": debug["object_xy_shift"],
                    "object_drop_delta": debug["object_drop_delta"],
                    "xy_err": debug["xy_err"],
                    "gripper_qpos_debug": debug["gripper_qpos"],
                    "gripper_sum": debug["gripper_sum"],
                    "close_countdown": debug["close_countdown"],
                    "settle_countdown": debug["settle_countdown"],
                    "close_stable_counter": debug["close_stable_counter"],
                }
                if args_cli.state_mode == "official":
                    record["eef_pos"] = record_extra["eef_pos"]
                    record["eef_quat_wxyz"] = record_extra["eef_quat_wxyz"]
                    record["eef_axis_angle"] = record_extra["eef_axis_angle"]
                    record["gripper_qpos"] = record_extra["gripper_qpos"]
                append_jsonl(steps_jsonl, record)
                records.append(record)

                print(
                    f"[INFO] attempt={attempt_idx} keep_idx={episode_idx} step={step_idx} "
                    f"stage={debug['stage']} reward={reward_np} terminated={terminated_np} truncated={truncated_np} "
                    f"lift_delta_z={debug['lift_delta_z']:.4f} xy_err={debug['xy_err']:.4f} xy_shift={debug['object_xy_shift']:.4f} "
                    f"gripper_qpos={np.asarray(debug['gripper_qpos']).round(4)} gripper_sum={debug['gripper_sum']:.4f} "
                    f"close_cd={debug['close_countdown']} settle_cd={debug['settle_countdown']} stable={debug['close_stable_counter']}"
                )

                if ctx.success:
                    print(f"[INFO] scripted success reached: {ctx.success_reason}")
                    break
                if ctx.stage == "abort":
                    print("[INFO] scripted abort: object was pushed away or dropped too much.")
                    break
                if bool(np.asarray(terminated_np).any()) or bool(np.asarray(truncated_np).any()):
                    print(f"[INFO] attempt={attempt_idx}: episode ended at step {step_idx}")
                    break
        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            print("[WARN] KeyboardInterrupt detected. Stopping scripted collection.")
            break

        success, reason = evaluate_episode_success(records, ctx)
        if success:
            print(f"[OK] keep episode_{episode_idx:05d} because {reason}")
            kept_episodes += 1
            total_saved_steps += len(records)
        else:
            print(f"[WARN] discard attempt {attempt_idx}: {reason}")
            if args_cli.save_failed:
                print(f"[INFO] save_failed=True, keeping failed attempt in {episode_dir}")
                kept_episodes += 1
                total_saved_steps += len(records)
            else:
                shutil.rmtree(episode_dir, ignore_errors=True)

    env.close()
    print(f"[SUCCESS] Collection finished. kept_episodes={kept_episodes} attempts={attempted_episodes} total_saved_steps={total_saved_steps}")
    print(f"[SUCCESS] Output root: {dataset_root}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
