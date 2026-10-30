# 格雷码邻近行匹配与重建

独立入口为 `D:/vscode/visual information/project/code/reconstruct_gray_neighborhood.py`，适用于 `gray_phase_c*_v1` 序列。它允许在邻近行内按双轴投影坐标搜索对应点，保留唯一性、全局一对一和左右反向检查，并用左右实际二维坐标三角化。原有重建入口的默认行为不受本次新增文件影响。

扩大搜索能缓解纵向校正偏差导致的匹配失败，但不会修正相机内外参。该入口的所有输出均标记为 `experimental_geometry_unverified`；点云名称含 `EXPERIMENTAL`。使用 OpenCV 的 `triangulatePoints`，输出先位于校正后左相机坐标系，再转换到原始左相机坐标系。[OpenCV 三角化说明](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## 当前扫描缓存的对照结果

当前原始序列已不在工作目录中，因此本次使用上一轮的解码缓存；未重新解码或降低灰码、光照和调制度阈值。缓存记录的标定路径和处理缩放均已核对；旧缓存没有内嵌标定哈希，这一限制写入 QC。新输出保存了当前标定哈希和缓存哈希。

两组均采用投影坐标距离 1.5 px、唯一性差值 0.1 px、精确反向一致性、深度 300～1500 mm、点云步长 2，以及每台相机最大 2 个处理像素的重投影误差，只有行搜索半径不同。

| 指标 | 仅同一行 | 上下各 4 行 |
| --- | ---: | ---: |
| 双向一致对应 | 880 | 232,356 |
| 几何与深度过滤后 | 880 | 224,708 |
| 抽样点云 | 252 | 56,337 |
| 接受点的原图重投影误差中位数 | 约 0 px | 约 3.00 px |

这是同一新匹配器的对照。此前原入口输出 266 点；新入口的全局唯一性与精确反向检查更严格，因此同一行基线为 252 点。仅同一行时，横向双目模型可以构造零重投影残差，这本身不证明对应点正确。

扩展结果中 199,210 个有效点来自右图比左图低 3 行的匹配。若把几何误差上限恢复到 1 个处理像素，扩展结果中仅 639 个有效像素能保留，说明大量新增点依赖允许更大的几何残差，尚未获得精度修复。

所选鸭身内部区域共 16,450 个处理像素，扩展后仅有 76 个有效点。主要增加的是墙面和底座，鸭身细条纹调制度不足的问题仍需分别处理。

进一步使用棋盘已知的正确角点对应、完全绕过格雷码匹配，在原标定下三角化：A/A2 图中的 27 mm 方格，横纵相邻角点距离中位数约为 24.75～25.26 mm；B 组约为 26.74～26.86 mm。这是基于用户提供方格尺寸的几何一致性检查，说明即使找到了正确对应点，当前几何仍会造成尺寸偏差。

缓存重建没有原始白光颜色，点云使用中性灰色；深度预览采用伪彩色。

## 复现缓存实验

PowerShell 命令如下。输出目录必须是新目录，避免覆盖已有结果。

```powershell
$projectDir = 'D:\vscode\visual information\project'

python "$projectDir\code\reconstruct_gray_neighborhood.py" `
  --decoded "$projectDir\processed\scan_20261030_1544_27mm\fixed_mask\decoded_projector_xy.npz" `
  --calibration "$projectDir\calibration\stereo\recalibration_20261030_27mm\stereo_calibration.npz" `
  --output "$projectDir\processed\gray_neighborhood_20261030\radius_4_repeat" `
  --row-radius 4 `
  --max-reprojection-error-px 2
```

如果有完整原始格雷码序列，将 `--decoded` 替换为 `--view` 并传入该次 `repeat` 目录。原始序列模式使用真实左白光图给点云着色。该入口不处理散斑协议。

## 参数和输出

- `--row-radius`：默认 0，单位是处理图像像素。`--scale 0.5` 时，4 行对应原图约 8 行。半径应按观察到的残差选择；过大的范围会增加歧义。
- `--match-tolerance-px`：默认 1.5，单位是二维投影坐标的欧氏距离，与相机图像行数无关。
- `--uniqueness-margin-px`：默认 0.1，最佳和次佳合格候选的距离差必须大于该值；相同编码的歧义候选会被拒绝。
- `--lr-tolerance-px`：默认 0，左右反向必须回到同一个二维相机像素。
- `--max-reprojection-error-px`：默认 1，单位是每台相机的处理图像像素。本次实验显式设为 2；在半分辨率下相当于每台相机最大 4 个原图像素。
- `--point-stride`：默认 2；只影响导出的点云抽样，不改变深度有效像素。
- `--left-color`：可为缓存模式提供尺寸完全一致、已经校正的左彩色图。缺省使用中性灰色。

输出包括两种左相机坐标系下的实验 PLY、`depth_mm.npy`、`right_xy_px.npy`、`vertical_offset_px.npy`、`matching_score_projector_px.npy`、`reprojection_error_px.npy`、有效掩膜、深度预览和 `qc.json`。`vertical_offset_px` 定义为右图行号减左图行号；重投影误差图记录两台相机误差的较大值。

跨行匹配必须保留实际右图行号，不能只保存横向视差后直接使用原 Q 矩阵。

## 验证

新增 17 项测试覆盖搜索边界、两轴编码、视差范围、重复投影 x 候选、歧义拒绝、跨行目标唯一性、二维反向检查、穷举对照、缩放、负深度、缓存路径及哈希关联、几何过滤，以及完整 38 帧合成序列的端到端重建和实际图像尺寸检查。当前工作区全部 46 项本地测试通过。独立审查另进行了 1,000 组随机输入与穷举结果的对照，未发现差异。真实数据的点数、PLY 负载和投影残差另作核对。结果和对照图保存在 `D:/vscode/visual information/project/processed/gray_neighborhood_20261030`。
