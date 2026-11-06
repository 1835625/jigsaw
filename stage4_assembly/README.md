# Stage 4 — 数字拼接与人工摆放

`assemble.py` 根据 Stage 3 布局，将 Stage 2 的真实 RGBA 碎片拼回画布，
输出原始/颜色校正版、接缝诊断、质量报告和逐步人工摆放计划。
默认接入 `step2`、`step3` 已提交的数据，不重新运行求解器。

## 1. 运行

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python stage4_assembly/assemble.py \
  --reference stage1_capture_calibration/output/reference/reference.png
```

默认读取 `stage2_piece_extraction/output/pieces.json` 及其相对路径下的 RGBA/mask，
布局来自 `stage3_feature_matching/quad_puzzle/examples/output.json`，
输出到 `stage4_assembly/output/`。这些是仓库样本的结果。
仓库保留了这组样例的 PNG、报告与人工操作表；上述命令可重新生成。
新一批数据请使用独立输出目录，避免覆盖样例。
新一批数据必须显式指定对应输入：

```bash
python stage4_assembly/assemble.py \
  --pieces /path/to/stage2/output/pieces.json \
  --placements /path/to/stage3/result.json \
  --reference /path/to/reference.png \
  --config stage4_assembly/config/assembly.example.json \
  --output /path/to/assembly
```

| 参数 | 含义 |
| --- | --- |
| `--pieces` | 完整版 `pieces.json`；精简版 `pieces_base_edge.json` 缺少图像路径，不能替代 |
| `--placements` / `--assignments` | Stage 3 位姿或参考图分配 JSON |
| `--reference` | 可选：整图朝向判别、颜色统计与评价 |
| `--layout` | 参考图分配模式必需的 `jigsaw_layout.json` |
| `--orientation auto\|0\|90\|180\|270` | 整图顺时针旋转；auto 只在提供参考图时判别朝向 |
| `--no-color-correction` | 保留拍摄颜色，仍可评价/判别朝向 |
| `--no-steps` | 不生成逐步 PNG，仍输出完整计划及总览 |
| `--config` | JSON 参数，可覆盖部分默认值；未知参数报错 |

不提供参考图时保持求解器朝向，不做校色，SSIM/PSNR 为不可用。
样本求解器朝向与原图相差 180°，复现正向效果请使用带参考图的命令。
参考图只选择整图朝向，不改变碎片相对布局。

退出码：0 = 成功生成结果（可能含复核项）；1 = 已生成结果且
`fail_on_manual_review=true`、存在复核项；2 = 输入/运行/读写错误。
同时读取报告的 `status/warnings`，不能只靠退出码判断是否免复核。
未知/无解布局、重复/遗漏 ID、错误单位、空掩码、奇异/镜像单应会被拒绝。
同名产物会覆盖；重跑会清理过期步骤图和可选比较图，不清理其他文件。

## 2. 两种输入契约

### 已实现的 Stage 3 几何位姿

接受 `polygon-assembly-result/v2`、`polygon-assembly-result/v1`、
`quad-assembly-result/v1`，状态须为 `feasible/approximate`。
`placements[].id` 须恰好覆盖 Stage 2 的 `piece_id`，单位须与
`pieces.json.coordinate_system` 一致，裁剪比例须为 1。

**`rotationDeg/origin` 作用于基边局部坐标，origin 是 P0 的落点，不是 PNG 左上角。**
Stage 2 PNG 有留白，基边坐标还可能有负 x。令
`B=rectified_to_base_edge_matrix`、`C=rectified_to_piece_matrix`、
`T` 为 Stage 3 的旋转平移、`G` 为整图朝向变换，则：

```text
H_PNG_to_assembly = G @ T @ B @ inverse(C)
H_table_to_assembly = H_PNG_to_assembly @ C
```

用 `image_corners_px` 交叉检查转换；缺少矩阵时，可由 PNG 角点和基边角点的
恒定偏移恢复裁剪平移，不猜测所有留白都是 20 px。
画布保持实物几何尺度，评价时显式缩放参考图。
本模式不需要 P 编号，计划的 `source_piece_id=null`，不会编造对应关系。

### SIFT 参考图 assignments

```bash
python stage4_assembly/assemble.py \
  --pieces stage2_piece_extraction/output/pieces.json \
  --assignments stage3_feature_matching/output/assignments.json \
  --layout stage1_capture_calibration/output/cut_pdf/jigsaw_layout.json \
  --reference stage1_capture_calibration/output/reference/reference.png
```

JSON 使用 `schema_version="1.0"`、`assignments` 数组，各项包含
`scan_piece_id/source_piece_id/homography_piece_to_source/confidence/status`。
默认单应从 PNG 像素映射到参考图；`H_piece_to_source`、早期的 `H_scan_to_source`
为同义字段，每条只能提供一个。若单应从整张俯视图出发，必须指定
`transform_source_frame="rectified_px"`（默认 `piece_image_px`），程序会合成裁剪逆变换。
`accepted/review` 可重建，review 显式保留。源编号须对 layout 的全部多边形形成一一映射，
`unassigned` 须为空。目标已在原图坐标中，不再旋转整图；参考图尺寸须与 layout 一致。

## 3. 拼接、颜色与质量

RGB 先预乘 alpha，再变换颜色/alpha 并解预乘，避免透明黑边插值污染。
纹理由目标多边形限制，重叠处按掩码距离羽化并归一化权重。
另可选 `blend_mode="overwrite"` 作对照；不支持的融合模式直接报错。

Lab 均值/方差校正强度默认 0.5；标准差比限制在 0.8–1.25，均值差限制为
各 Lab 通道 ±10，再乘强度；低方差区域不放大噪声。
同时输出未校色的 `assembled_raw.png`。
默认 `edge_erode_px=0`，避免再次腐蚀制造缝隙；可配置腐蚀去纸边，
但真实覆盖会减少并反映在报告中。羽化只混合实际重叠区域，不修正几何错位。
**参考图从不用于缺口填充，缺口保留黑色并输出覆盖掩码。**

有参考图时，同时评价原始/校色版的全画布 RGB PSNR、仅覆盖区 RGB PSNR、
全画布灰度 SSIM。PSNR 完全相同以 `identical=true, psnr_db=null` 表示无穷大；
无参考图则 `available=false`。灰度 SSIM 不衡量色彩恢复，不能与彩色 SSIM 混用。
未覆盖区参与全画布评价。每条接缝报告两侧平均 Delta E76、梯度幅值差，
以及参考图同位置的颜色差/梯度残差；真实图案也有颜色边界，这些值不是错缝率。
若没有有效两侧样本，则样本数为 0、指标为 null。
连续几何缺口/重叠/越界与栅格覆盖分别统计，前者不受取整影响。

## 4. 人工顺序与坐标

打开 `output/placement_guide.md`：

- `scan_locations.png` 标出原桌面取片位置，原俯视图可读时生成。
- `piece_index.png` 是按摆放顺序排列的碎片正面缩略图。
- `placement_order.png` 标出步骤号、S 编号与目标位置。
- `steps/01.png..16.png` 是逐步装配，绿边为新增碎片；表格步骤号可打开图片。

从最靠近目标左上角的碎片开始，每次优先选与已放碎片公共边总长最大的下一片。
默认允许 8 px 测量偏差，仅接触一个角不算公共边。邻接图断开时显式另起锚点。
这是确定性人工对齐顺序，不声称操作次数或搬运距离最优。

`placement_plan.json` 保留 README §6.3 的 `slot` 数组。
每步含目标多边形/中心、PNG 单应、原桌面位置、从原桌面朝向转过的角度、
已放邻片、复核标记和步骤图路径。角度正值顺时针，范围 `[-180,180)`，
不同于 Stage 3 相对于基边的 `rotationDeg`。
几何模式且 `px_per_mm` 为有限、正、各向同性密度时，额外输出相对目标外框的毫米坐标。
样本外框约 **253.38×142.87 mm**：在空白区自定左上角，x 向右、y 向下。
目标坐标轴须与原俯视图平行；整体转动外框时，人工转角还须叠加该转角。
原照片中的碎片若已被转动，角度须重新核对。参考图分配模式不从扫描密度猜原图毫米比例。
本阶段只供人工摆放，无机械臂、抓取、手眼标定或机器人轨迹代码。

## 5. 结果与验证

样本 16/16 块；连续外框 2027.016623×1142.954974 px，栅格 2028×1143。
提供参考图时选整图顺时针 180°：

| 指标 | 未校色 | 保守校色 |
| --- | ---: | ---: |
| RGB PSNR | 18.6555 dB | 20.4931 dB |
| 灰度 SSIM | 0.73938 | 0.74444 |

连续几何覆盖率 99.446573%，栅格覆盖率 99.551813%；后者受栅格化和 alpha 插值影响。
实拍印刷件的颜色/对比度损失、手剪误差、上游近似布局使 SSIM 未达早期建议的 0.85。
报告为 `review`：Stage 3 近似解未通过严格几何验收，Stage 2 对 S06 有复核标记。
数字重建完成不表示上游误差消失。

```bash
python -m unittest discover -s tests -p 'test_stage4_assembly.py' -v
python -m unittest discover -s tests -v
```

测试覆盖独立像素真值、裁剪偏移、透明边、两种协议、桌面角度、缺口不补图、
严格输入拒绝、重复运行及实拍回归。
详见 [`docs/REPRODUCTION_STAGE4.md`](../docs/REPRODUCTION_STAGE4.md)。

## 6. 几何与 SIFT 的统一对比

本分支的 `stage3_feature_matching/match_sift.py` 已实现 SIFT + RANSAC +
匈牙利一一分配。它使用参考图切出的 P 模板、已知源多边形和实拍 S 碎片，
输出 PNG 到参考图的单应；几何求解只使用局部片形，保持旋转和平移。

在仓库根目录执行：

```bash
python stage4_assembly/compare_methods.py
```

默认复用已有几何位姿和 SIFT `assignments.json`，不重复运行 Stage 3 搜索。
如需重新匹配，先运行 `python stage3_feature_matching/match_sift.py`。
全部输入路径可用 `--pieces/--geometry/--sift/--layout/--reference` 指定，
输出目录可用 `--output` 指定。

产物位于 `output/method_comparison/`：

- `comparison_raw.png`、`comparison_corrected.png`：参考图与两方案的横向对比。
- `comparison_report.json`、`comparison.md`：共同评价网格上的指标与复核项。
- `geometry/`、`sift/`：按原生坐标运行第四阶段，分别生成图像、诊断和人工计划。
- 各方案的 `*_raw.png`、`*_corrected.png`、`*_coverage.png`：共同网格上的图像和 Alpha 覆盖。

比较时直接把同一批 RGBA 碎片渲染到参考图的 1280×720 网格，使用相同羽化和校色参数。
几何布局仅增加一个整体外框归一化，单片位姿不优化；其整体朝向由参考图辅助判断。
原生毫米坐标计划保存在单独目录，共同网格仅供图像评价。

必须同时查看三种量：

1. `raster`：根据变换后的 Alpha 统计真实纹理覆盖，不用 RGB 非零判断覆盖。
2. `warped_measured_geometry`：实测 PNG 角点经过实际刚体/单应变换后，计算缺口、重叠和越界。
3. `target_geometry`：用于裁剪纹理的目标轮廓。SIFT 的目标来自已知切割布局，
   其接近 100% 的覆盖率不能当作匹配精度或复原正确率。

两种方案在原图、切割先验和变换自由度方面不同；这不是相同无参考条件下的算法排名。
校色后的指标另受参考图统计量辅助，未校色指标必须保留。
SIFT 的单应可含尺度、剪切和透视变化，其参考像素坐标不能直接当作实物毫米摆放坐标。
完整数据、环境及限制见 [对比复现记录](../docs/REPRODUCTION_STAGE4_COMPARISON.md)。

验证：`python -m unittest discover -s tests -v`，包含 4 个新增对比测试。

