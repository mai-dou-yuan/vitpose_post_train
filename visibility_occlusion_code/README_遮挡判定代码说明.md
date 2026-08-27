# 关节点遮挡判定代码说明

这份代码用于根据 3D 手部关节点、相机内参和畸变参数，自动生成 wrist-camera 视角下的 joint-level visibility labels。

遮挡标签不是人工标注数据集，而是通过代码自动判定。

## 1. 文件说明

```text
utils/visibility_proxy.py
```

核心遮挡判定代码。主要功能：

```text
1. 将 GT 3D joints 投影到 2D 图像；
2. 判断关节点是否在图像视野内；
3. 使用手部几何 proxy 判断 self-occlusion；
4. 输出 visible / self_occluded / out_of_view 标签。
```

```text
tools/generate_visibility_labels.py
```

生成遮挡标签的入口脚本。输入 prediction/GT npz，输出带 visibility_label 的 npz。

```text
tools/visualize_visibility_labels.py
```

遮挡标签可视化脚本。用于检查 visible / self_occluded / out_of_view 标签是否合理。

```text
tools/evaluate_occlusion_metrics.py
```

按 visible joints、self-occluded joints、occluded fingertips 计算指标。

```text
tools/evaluate_visibility_ratio_groups.py
```

按每帧 visible joint ratio 分组计算指标。

```text
tools/apply_visibility_labels.py
```

将一份参考 visibility labels 复用到其他 baseline 的 prediction npz 上，用于公平对比。

```text
tools/export_per_joint_metrics.py
```

导出 per-joint error，支持 visible / occluded 分组。

```text
utils/pose_metrics.py
```

指标计算工具，被遮挡指标脚本调用。

## 2. 标签定义

```text
0 = visible
1 = self_occluded
2 = out_of_view
3 = uncertain
```

当前实验中主要使用：

```text
visible
self_occluded
out_of_view
```

`uncertain` 保留为扩展标签，当前自动判定默认不会主动产生。

## 3. 输入 NPZ 需要包含的字段

生成遮挡标签时，输入 npz 至少需要：

```text
gt_pose: [N, 21, 3]
cam_k: [N, 3, 3]
dist_coeffs: [N, 5] 或 [N, 1, 5]
```

如果要后续评估预测误差，还需要：

```text
pred_pose: [N, 21, 3]
```

## 4. 核心方法

遮挡判定不是通过预测误差反推，而是只基于 GT 和相机参数。

流程：

```text
1. 使用 cv2.projectPoints 将 GT 3D joints 投影到图像平面；
2. 若深度无效、在相机后方或投影点超出图像范围，则标记为 out_of_view；
3. 对 in-view 关节，从相机原点向目标关节发射射线；
4. 将手部骨架构造成 proxy 几何体：
   - finger bones: capsules
   - palm bones: capsules
   - joints: spheres
5. 若射线在到达目标关节前先击中其他手部 proxy，且深度差超过阈值，则标记为 self_occluded；
6. 否则标记为 visible。
```

## 5. 保守版参数

当前实验采用保守参数：

```text
finger_radius = 5
joint_radius = 5
palm_radius = 10
depth_margin = 10
image_size = 336
```

含义：

```text
finger_radius / joint_radius / palm_radius 控制 proxy 几何体粗细；
depth_margin 控制判定遮挡所需的最小前后深度差；
保守参数会减少误判为遮挡的关节。
```

## 6. 生成遮挡标签命令

```bash
python tools/generate_visibility_labels.py \
  --input results/predictions/test_users_4_8_13_predictions.npz \
  --output results/predictions/test_users_4_8_13_with_visibility_conservative.npz \
  --image-size 336 \
  --finger-radius 5 \
  --joint-radius 5 \
  --palm-radius 10 \
  --depth-margin 10
```

输出文件会在原有字段基础上新增：

```text
visibility_label: [N, 21]
projected_2d: [N, 21, 2]
in_view_mask: [N, 21]
visible_joint_ratio: [N]
```

## 7. 统计标签数量

```bash
python - <<'PY'
import numpy as np

p = 'results/predictions/test_users_4_8_13_with_visibility_conservative.npz'
d = np.load(p, allow_pickle=True)
labels = d['visibility_label']
names = {0: 'visible', 1: 'self_occluded', 2: 'out_of_view', 3: 'uncertain'}

for k, v in names.items():
    print(v, int((labels == k).sum()), 'ratio', float((labels == k).mean()))

print('frames', labels.shape[0], 'joints', labels.shape[1])
print('visible_joint_ratio mean', float(np.nanmean(d['visible_joint_ratio'])))
print('visible_joint_ratio min/max', float(np.nanmin(d['visible_joint_ratio'])), float(np.nanmax(d['visible_joint_ratio'])))
PY
```

当前 users 4/8/13 保守版标签统计：

```text
visible 20922 ratio 0.4901
self_occluded 19719 ratio 0.4619
out_of_view 2052 ratio 0.0481
uncertain 0 ratio 0.0000
frames 2033
joints 21
visible_joint_ratio mean 0.5210
```

## 8. 可视化检查

```bash
python tools/visualize_visibility_labels.py \
  --npz results/predictions/test_users_4_8_13_with_visibility_conservative.npz \
  --out-dir results/visibility_vis_conservative \
  --num 20 \
  --target-user-ids 4,8,13
```

可视化用于人工确认自动遮挡标签是否合理。

## 9. 遮挡指标评估

```bash
python tools/evaluate_occlusion_metrics.py \
  --input results/predictions/test_users_4_8_13_with_visibility_conservative.npz \
  --output-json results/metrics/users_4_8_13_occlusion_metrics_conservative.json
```

会输出：

```text
all_in_view
visible
occluded
occluded_fingertips
```

## 10. 对 baseline 复用同一份遮挡标签

为了公平比较，不同模型需要使用同一份 visibility labels。

例如 HaMeR：

```bash
python tools/apply_visibility_labels.py \
  --input results/predictions/hamer_predictions.npz \
  --reference results/predictions/test_users_4_8_13_with_visibility_conservative.npz \
  --output results/predictions/hamer_with_visibility.npz
```

WiLoR / SimpleHand / EgoPoseFormer 同理。

## 11. 依赖

主要依赖：

```text
numpy
opencv-python / cv2
```

评估指标脚本还需要：

```text
scipy
```

具体依赖以原项目环境为准。

## 12. 注意事项

```text
1. 不要用预测误差大小定义遮挡，否则会形成循环论证；
2. out_of_view 和 self_occluded 应分开统计；
3. 仅用 2D 投影点是否在图像内，不能判断 self-occlusion；
4. 当前方法是基于 GT 3D joints + camera parameters + hand proxy geometry 的自动 proxy label；
5. 若用于论文，应说明这是 automatic proxy visibility labels。
```
