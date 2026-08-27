# Full DexYCB random input visualization

脚本默认以 seed 42 从过滤后的官方 s0 train split 选取 100 组，并优先覆盖不同的
subject/sequence/camera。每组包含：

1. `01_original_crop_box.jpg`：原始 RGB、GT 2D joints 和实际 crop 框；
2. `02_unaugmented_crop.jpg`：未增强 crop 和同步 joints；
3. `03_augmented_crop.jpg`：训练增强 crop 和同步 joints；
4. `04_segmentation_crop.jpg`：crop 内 segmentation id 255；
5. `00_panel.jpg`：上述四图横向对照。

```bash
conda run -n vit python \
  experiments_graphormer_dexycb_light_fastvit/sample_visualization/visualize_random_samples.py \
  --setup s0 --split train --count 100 --seed 42
```

可使用 `--config`、`--setup`、`--split`、`--count`/`--num-samples`、`--seed` 和
`--output-dir` 调整运行。脚本生成每 10 组一页的 overview、JSON manifest 和 CSV
manifest。manifest 记录 subject、sequence、camera、frame、源路径、crop 参数和 crop
内手部像素数，并拒绝任何来自本地 sampled/splits 子集的路径。
