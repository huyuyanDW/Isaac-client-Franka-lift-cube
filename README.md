# Isaac Client README
为实现"数据采集"-"训练格式转换"-"使用训练策略展示推理"链路，完成了6个脚本（最终只需4个，另外2个为检查相机与确认数据格式正确）
infer_runs为使用run_isaac_client_with_rag_and_wrist推理展示客户端的一次推理结果，准确率为90%（共20ep，失败2次）
推理策略使用collect_scripted_rollout_dataset采集的200ep微调训练的pi0.5
- paligemma_variant="gemma_2b_lora",
- action_expert_variant="gemma_300m_lora",
- batch_size=2,num_workers=0,num_train_steps=12000,log_interval=50,
- lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=200,
        peak_lr=1e-5,
        decay_steps=4000,
        decay_lr=1e-5,）
- 以下按顺序包含脚本介绍、执行逻辑以及常用指令

## 当前脚本
### 1. `camera_probe.py`
用途：相机诊断 / 视角探针

作用：
- 单独检查主视角相机是否正常出图
- 用于排查相机全黑、全白、视角异常等问题
- 不参与正式训练或评测主流程
- 新环境第一次检查相机
- 修改相机参数后快速看图
- 排查 wrist / main 图像异常时

---

### 2. `run_isaac_client_with_rgb_and_wrist.py`
用途：**在线推理展示 / 固定轮数评测客户端**

作用：
- 连接 openpi policy server
- 将 `main image + wrist image + state + prompt` 送入策略
- 支持固定 episode 数运行
- 自动保存每轮的 `main.mp4 / wrist.mp4 / summary.json`
- 运行结束后自动汇总成功率
- **展示推理效果**
- **固定轮数评测**
- 不负责正式数据集采集
- 快速比较不同 checkpoint 的在线表现

---

### 3. `collect_rollout_dataset.py`
用途：**模型 rollout 采集**

作用：
- 使用当前在线策略执行任务
- 将 rollout 保存为中间格式数据集
- 生成 `episode_xxxxx/frames_main/frames_wrist/steps.jsonl/meta.json`
- **用来采集模型真实 rollout**
- **用来做正式评估留档**
- 微调后跑 10 条 / 20 条 probe，统计实际成功情况
- 为后续人工复查保留完整 rollout 记录

---

### 4. `validate_rollout_dataset.py`
用途：**中间格式 rollout 数据检验**

作用：
- 用于检查 `collect_rollout_dataset.py` 和 `collect_teleop_rollout_dataset.py` 生成的中间数据是否可用
   - 检查 `episode_xxxxx` 目录结构是否完整
   - 检查图片、`steps.jsonl`、`meta.json` 是否一致
   - 检查 state / action 维度、字段、NaN 等问题
- 中间格式数据进入导出前必须跑一次
- 不直接用于训练，只做数据质量把关

---

### 5. `convert_to_lerobot_v21.py`
用途：**训练格式导出（LeRobot v2.1-style local export）**

作用：
- 将中间格式 rollout 数据集导出为当前 openpi 可训练的本地格式
- 自动生成：
  - `data/chunk-xxx/episode_xxxxxx.parquet`
  - `videos/chunk-xxx/image/episode_xxxxxx.mp4`
  - `videos/chunk-xxx/wrist_image/episode_xxxxxx.mp4`
  - `meta/info.json`
  - `meta/stats.json`
  - `meta/tasks.jsonl`
  - `meta/episodes.jsonl`
  - `meta/episodes_stats.jsonl`
- 所有进入 openpi 训练的数据，都先通过此脚本导出

---

### 6. `collect_scripted_rollout_dataset.py`
用途：**scripted teacher / teacher 数据采集**

作用：
- 进行 **scripted expert 自动采集**
- 自动执行接近、抓取、抬升，生成 teacher 数据
- 输出同样的中间格式：
  - `episode_xxxxx/frames_main/frames_wrist/steps.jsonl/meta.json`
- 用来大规模采集成功抓取样本，供后续导出和微调

---

## 主流程

### A. 采集 teacher 数据
使用：

- `collect_scripted_rollout_dataset.py`

输出：
- 中间格式 teacher 数据集

### B. 检查中间格式数据
使用：

- `validate_rollout_dataset.py`

### C. 导出训练格式
使用：

- `convert_to_lerobot_v21.py`

输出：
- LeRobot v2.1-style local dataset

### D. 训练前计算 norm stats
在 openpi 目录中运行 `compute_norm_stats.py`

### E. 微调训练
在 openpi 中使用目标 config 进行训练

### F. 在线展示 / 评测
- 展示：`run_isaac_client_with_rgb_and_wrist.py`
- rollout 留档：`collect_rollout_dataset.py`

---

## 分工总结

- `camera_probe.py`：相机诊断
- `run_isaac_client_with_rgb_and_wrist.py`：在线展示 / 固定轮数成功率评测
- `collect_rollout_dataset.py`：模型 rollout 采集
- `validate_rollout_dataset.py`：中间数据质检
- `convert_to_lerobot_v21.py`：训练格式导出
- `collect_scripted_rollout_dataset.py`：scripted teacher 数据采集


---

## 命令

下面这些命令按当前脚本整理。路径需要按照实际修改。

### 1. 相机探针：`camera_probe.py`

运行主视角探针：

```bash
/opt/conda/envs/isaaclab/bin/python /root/gpufree-data/isaac_client/camera_probe.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1
```

无缓冲输出：

```bash
/opt/conda/envs/isaaclab/bin/python -u /root/gpufree-data/isaac_client/camera_probe.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1
```

---

### 2. 在线展示 / 固定轮数评测：`run_isaac_client_with_rgb_and_wrist.py`

运行 20 轮、每轮 150 步，自动保存视频和成功率汇总：

```bash
/opt/conda/envs/isaaclab/bin/python -u /root/gpufree-data/isaac_client/run_isaac_client_with_rgb_and_wrist.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --episodes 20 \
  --steps 150 \
  --state-mode official \
  --enable_cameras \
  --video_fps 20 \
  --stop_on_success
```

输出目录默认在：

```text
/root/gpufree-data/isaac_client/infer_runs/run_时间戳/
```

包含：
- 每个 episode 的 `main.mp4`
- 每个 episode 的 `wrist.mp4`
- 每个 episode 的 `summary.json`
- 总 `summary.json`（含成功率）

---

### 3. 模型 rollout 采集：`collect_rollout_dataset.py`

采集 10 条 rollout：

```bash
/opt/conda/envs/isaaclab/bin/python -u /root/gpufree-data/isaac_client/collect_rollout_dataset.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --episodes 10 \
  --steps_per_episode 120 \
  --state-mode official \
  --enable_cameras \
  --dataset_root /root/gpufree-data/isaac_client/datasets/rollout_dataset_probe
```

---

### 4. 中间格式数据质检：`validate_rollout_dataset.py`

检查 rollout 数据集：

```bash
/opt/conda/envs/isaaclab/bin/python /root/gpufree-data/isaac_client/validate_rollout_dataset.py \
  --dataset_root /root/gpufree-data/isaac_client/datasets/rollout_dataset_probe
```

检查 scripted teacher 数据集：

```bash
/opt/conda/envs/isaaclab/bin/python /root/gpufree-data/isaac_client/validate_rollout_dataset.py \
  --dataset_root /root/gpufree-data/isaac_client/datasets/scripted_rollout_dataset_v3
```

---

### 5. 训练格式导出：`convert_to_lerobot_v21.py`

将中间格式 teacher 数据导出为 LeRobot v2.1-style local format：

```bash
/opt/conda/envs/isaaclab/bin/python /root/gpufree-data/isaac_client/convert_to_lerobot_v21.py \
  --src_root /root/gpufree-data/isaac_client/datasets/scripted_rollout_dataset_v3 \
  --out_root /root/gpufree-data/isaac_client/datasets/lerobot_v21_export_scripted_v3 \
  --fps 50 \
  --chunks-size 1000 \
  --overwrite
```

---

### 6. scripted teacher 数据采集：`collect_scripted_rollout_dataset.py`

采 200 条 scripted teacher 数据：

```bash
/opt/conda/envs/isaaclab/bin/python -u /root/gpufree-data/isaac_client/collect_scripted_rollout_dataset.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --episodes 200 \
  --max_attempts 200 \
  --steps_per_episode 120 \
  --state-mode official \
  --enable_cameras \
  --abort_object_xy_shift 0.12 \
  --abort_object_drop_delta 0.08 \
  --dataset_root /root/gpufree-data/isaac_client/datasets/scripted_rollout_dataset_v2
```

---

## openpi 常用命令

### 1. 计算 norm stats

```bash
cd /root/gpufree-data/openpi_ws/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/gpufree-data/openpi_cache

python -m uv run scripts/compute_norm_stats.py --config-name pi05_isaac_lift_scripted_v2
```

### 2. 复制 `norm_stats.json` 到 assets 目录

```bash
mkdir -p /root/gpufree-data/openpi_ws/openpi/assets/pi05_isaac_lift_scripted_v2/isaac_lift_scripted_v2

cp /root/gpufree-data/isaac_client/datasets/lerobot_v21_export_scripted_v3/norm_stats.json \
   /root/gpufree-data/openpi_ws/openpi/assets/pi05_isaac_lift_scripted_v2/isaac_lift_scripted_v2/norm_stats.json
```

### 3. 训练

```bash
cd /root/gpufree-data/openpi_ws/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/gpufree-data/openpi_cache
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python -m uv run scripts/train.py pi05_isaac_lift_scripted_v2 \
  --exp-name lift_scripted_v3_s1 \
  --overwrite
```

从已有 checkpoint 继续训练，用：

```bash
cd /root/gpufree-data/openpi_ws/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/gpufree-data/openpi_cache
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python -m uv run scripts/train.py pi05_isaac_lift_scripted_v2 \
  --exp-name lift_scripted_v3_s1 \
  --resume
```

### 4. 启动微调后的服务端

```bash
cd /root/gpufree-data/openpi_ws/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/gpufree-data/openpi_cache

python -m uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_isaac_lift_scripted_v2 \
  --policy.dir=/root/gpufree-data/openpi_ws/openpi/checkpoints/pi05_isaac_lift_scripted_v2/lift_scripted_v3_s1/11999
```
---

## 主线命令顺序

### teacher 数据 -> 导出 -> 训练 -> 展示

1. scripted teacher 采集：

```bash
/opt/conda/envs/isaaclab/bin/python -u /root/gpufree-data/isaac_client/collect_scripted_rollout_dataset.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --episodes 200 \
  --max_attempts 1600 \
  --steps_per_episode 120 \
  --state-mode official \
  --enable_cameras \
  --abort_object_xy_shift 0.12 \
  --abort_object_drop_delta 0.08 \
  --dataset_root /root/gpufree-data/isaac_client/datasets/scripted_rollout_dataset_v3
```

2. 质检：

```bash
/opt/conda/envs/isaaclab/bin/python /root/gpufree-data/isaac_client/validate_rollout_dataset.py \
  --dataset_root /root/gpufree-data/isaac_client/datasets/scripted_rollout_dataset_v3
```

3. 导出：

```bash
/opt/conda/envs/isaaclab/bin/python /root/gpufree-data/isaac_client/convert_to_lerobot_v21.py \
  --src_root /root/gpufree-data/isaac_client/datasets/scripted_rollout_dataset_v3 \
  --out_root /root/gpufree-data/isaac_client/datasets/lerobot_v21_export_scripted_v3 \
  --fps 50 \
  --chunks-size 1000 \
  --overwrite
```

4. 算 norm stats：

```bash
cd /root/gpufree-data/openpi_ws/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/gpufree-data/openpi_cache

python -m uv run scripts/compute_norm_stats.py --config-name pi05_isaac_lift_scripted_v2
```

5. 复制 stats：

```bash
mkdir -p /root/gpufree-data/openpi_ws/openpi/assets/pi05_isaac_lift_scripted_v2/isaac_lift_scripted_v2

cp /root/gpufree-data/isaac_client/datasets/lerobot_v21_export_scripted_v3/norm_stats.json \
   /root/gpufree-data/openpi_ws/openpi/assets/pi05_isaac_lift_scripted_v2/isaac_lift_scripted_v2/norm_stats.json
```

6. 训练：

```bash
cd /root/gpufree-data/openpi_ws/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/gpufree-data/openpi_cache
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python -m uv run scripts/train.py pi05_isaac_lift_scripted_v2 \
  --exp-name lift_scripted_v3_s1 \
  --overwrite
```

7. 启动服务：

```bash
cd /root/gpufree-data/openpi_ws/openpi
source .venv/bin/activate
export OPENPI_DATA_HOME=/root/gpufree-data/openpi_cache

python -m uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_isaac_lift_scripted_v2 \
  --policy.dir=/root/gpufree-data/openpi_ws/openpi/checkpoints/pi05_isaac_lift_scripted_v2/lift_scripted_v3_s1/11999
```

8. 用展示客户端看推理效果：

```bash
/opt/conda/envs/isaaclab/bin/python -u /root/gpufree-data/isaac_client/run_isaac_client_with_rgb_and_wrist.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --episodes 20 \
  --steps 150 \
  --state-mode official \
  --enable_cameras \
  --video_fps 20 \
  --stop_on_success
```
