import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd


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


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stats_for_list_rows(rows: list[list[float]]) -> dict:
    arr = np.asarray(rows, dtype=np.float32)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def stats_for_scalar_rows(rows: list[float | int]) -> dict:
    arr = np.asarray(rows, dtype=np.float32).reshape(-1)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.shape[0])],
    }


def read_frame(path: Path) -> np.ndarray:
    try:
        img = imageio.imread(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path}: {e}")
    if img.ndim != 3:
        raise ValueError(f"Unexpected image ndim for {path}: {img.ndim}, shape={img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.asarray(img, dtype=np.uint8)


def write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)


def resolve_task_text(meta: dict, records: list[dict], default_task: str) -> str:
    candidates = [
        meta.get("prompt"),
        records[0].get("prompt") if records else None,
        default_task,
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return default_task


def chunk_index_for_episode(ep_idx: int, chunks_size: int) -> int:
    if chunks_size <= 0:
        raise ValueError(f"chunks_size must be > 0, got {chunks_size}")
    return ep_idx // chunks_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert local rollout dataset to a LeRobot v2.1-compatible local dataset layout "
            "that openpi can train on directly. Episodes are automatically split into chunks "
            "according to --chunks-size."
        )
    )
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--robot-type", type=str, default="franka")
    parser.add_argument("--default-task", type=str, default="lift the cube")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {out_root}. Pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta_path = src_root / "meta.json"
    dataset_meta = load_json(dataset_meta_path) if dataset_meta_path.exists() else {}

    episode_dirs = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found in {src_root}")

    episodes_jsonl: list[dict] = []
    episodes_stats_jsonl: list[dict] = []

    all_state: list[list[float]] = []
    all_actions: list[list[float]] = []
    all_timestamp: list[float] = []
    all_frame_index: list[int] = []
    all_episode_index: list[int] = []
    all_index: list[int] = []
    all_task_index: list[int] = []

    task_to_index: dict[str, int] = {}
    task_rows: list[dict] = []
    global_index = 0

    for ep_idx, ep_dir in enumerate(episode_dirs):
        steps_path = ep_dir / "steps.jsonl"
        if not steps_path.exists():
            raise FileNotFoundError(f"Missing steps.jsonl in {ep_dir}")

        records = load_jsonl(steps_path)
        if not records:
            raise RuntimeError(f"No records found in {steps_path}")

        ep_meta_path = ep_dir / "episode_meta.json"
        ep_meta = load_json(ep_meta_path) if ep_meta_path.exists() else {}
        task_text = resolve_task_text(ep_meta, records, args.default_task)
        if task_text not in task_to_index:
            task_idx = len(task_to_index)
            task_to_index[task_text] = task_idx
            task_rows.append({"task_index": task_idx, "task": task_text})
        task_idx = task_to_index[task_text]

        state_rows: list[list[float]] = []
        action_rows: list[list[float]] = []
        prompt_rows: list[str] = []
        timestamp_rows: list[float] = []
        frame_index_rows: list[int] = []
        episode_index_rows: list[int] = []
        index_rows: list[int] = []
        task_index_rows: list[int] = []

        main_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []

        for step_idx, rec in enumerate(records):
            state = np.asarray(rec["state_8_mapped"], dtype=np.float32).reshape(-1).tolist()
            action = np.asarray(rec["env_action_executed"], dtype=np.float32).reshape(-1).tolist()

            state_rows.append(state)
            action_rows.append(action)
            prompt_rows.append(task_text)
            timestamp_rows.append(float(step_idx / args.fps))
            frame_index_rows.append(int(step_idx))
            episode_index_rows.append(int(ep_idx))
            index_rows.append(int(global_index))
            task_index_rows.append(int(task_idx))

            all_state.append(state)
            all_actions.append(action)
            all_timestamp.append(float(step_idx / args.fps))
            all_frame_index.append(int(step_idx))
            all_episode_index.append(int(ep_idx))
            all_index.append(int(global_index))
            all_task_index.append(int(task_idx))
            global_index += 1

            main_path = ep_dir / "frames_main" / f"{step_idx:06d}.png"
            wrist_path = ep_dir / "frames_wrist" / f"{step_idx:06d}.png"
            main_frames.append(read_frame(main_path))
            wrist_frames.append(read_frame(wrist_path))

        df = pd.DataFrame(
            {
                "state": state_rows,
                "actions": action_rows,
                "prompt": prompt_rows,
                "timestamp": timestamp_rows,
                "frame_index": frame_index_rows,
                "episode_index": episode_index_rows,
                "index": index_rows,
                "task_index": task_index_rows,
            }
        )

        chunk_idx = chunk_index_for_episode(ep_idx, args.chunks_size)
        chunk_name = f"chunk-{chunk_idx:03d}"
        data_dir = out_root / "data" / chunk_name
        video_main_dir = out_root / "videos" / chunk_name / "image"
        video_wrist_dir = out_root / "videos" / chunk_name / "wrist_image"
        data_dir.mkdir(parents=True, exist_ok=True)
        video_main_dir.mkdir(parents=True, exist_ok=True)
        video_wrist_dir.mkdir(parents=True, exist_ok=True)

        out_parquet = data_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(out_parquet, index=False)

        out_main_mp4 = video_main_dir / f"episode_{ep_idx:06d}.mp4"
        out_wrist_mp4 = video_wrist_dir / f"episode_{ep_idx:06d}.mp4"
        write_video(main_frames, out_main_mp4, args.fps)
        write_video(wrist_frames, out_wrist_mp4, args.fps)

        length = len(records)
        episodes_jsonl.append({
            "episode_index": ep_idx,
            "tasks": [task_text],
            "length": length,
        })
        episodes_stats_jsonl.append({
            "episode_index": ep_idx,
            "stats": {
                "state": stats_for_list_rows(state_rows),
                "actions": stats_for_list_rows(action_rows),
                "timestamp": stats_for_scalar_rows(timestamp_rows),
                "frame_index": stats_for_scalar_rows(frame_index_rows),
                "episode_index": stats_for_scalar_rows(episode_index_rows),
                "index": stats_for_scalar_rows(index_rows),
                "task_index": stats_for_scalar_rows(task_index_rows),
            },
        })

    total_episodes = len(episodes_jsonl)
    total_frames = len(all_index)
    total_chunks = max(1, math.ceil(total_episodes / args.chunks_size))

    stats_json = {
        "state": stats_for_list_rows(all_state),
        "actions": stats_for_list_rows(all_actions),
        "timestamp": stats_for_scalar_rows(all_timestamp),
        "frame_index": stats_for_scalar_rows(all_frame_index),
        "episode_index": stats_for_scalar_rows(all_episode_index),
        "index": stats_for_scalar_rows(all_index),
        "task_index": stats_for_scalar_rows(all_task_index),
    }

    info_json = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_rows),
        "total_videos": total_episodes * 2,
        "total_chunks": total_chunks,
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.channels": 3,
                    "video.fps": args.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "wrist_image": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.channels": 3,
                    "video.fps": args.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "state": {"dtype": "float32", "shape": [8], "names": None},
            "actions": {"dtype": "float32", "shape": [7], "names": None},
            "prompt": {"dtype": "string", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "source_rollout_meta": dataset_meta,
    }

    write_json(meta_dir / "info.json", info_json)
    write_json(meta_dir / "stats.json", stats_json)
    write_jsonl(meta_dir / "tasks.jsonl", task_rows)
    write_jsonl(meta_dir / "episodes.jsonl", episodes_jsonl)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats_jsonl)

    print("[SUCCESS] Export finished.")
    print(f"[SUCCESS] Output root: {out_root}")
    print(f"[SUCCESS] total_episodes: {total_episodes}")
    print(f"[SUCCESS] total_frames: {total_frames}")
    print(f"[SUCCESS] total_chunks: {total_chunks}")
    print(f"[SUCCESS] total_tasks: {len(task_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())