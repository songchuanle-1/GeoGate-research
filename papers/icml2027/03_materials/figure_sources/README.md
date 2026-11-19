# 6 张配图：编辑文件和数据

修改图中文字最快的入口是 `svg/`。编译论文时使用 `02_manuscript/figures` 中的 PDF，无需运行绘图脚本。图 1 从冻结的数值表重新绘制；另外 5 张图的来源按与现稿 PDF 的 SHA-256 一致选取。

| 现稿中的图 | 文件名 | 对应绘图数据和入口 |
| --- | --- | --- |
| 正文图 1：空间注意力 | `fig1_three_phase` | `attention_profile.py` 读取 `profiles_method/frozen/attention_profiles.source.csv`，与独立存储的曲线数组逐值比对。[预览](fig1_preview.png) |
| 正文图 2：GeoGate 执行流程 | `fig2_geogate_method` | `profiles_method/architecture.py` |
| 正文图 3：分层配对回放 | `fig_layer_replay_paired_final` | `quantitative/figures/paired_replay_*.source.csv` |
| 正文图 4：任务迁移 | `fig_transfer_effects` | `quantitative/figures/transfer_effects.source.csv` |
| 附录图 S1：历史层曲线 | `fig_layer_curve` | `quantitative/figures/historical_layer_curve.source.csv` |
| 附录图 S2：剪枝与效率概览 | `fig_teaser` | `quantitative/figures/overview.source.csv` |

在装有 NumPy、Matplotlib 和 Times New Roman 的环境中，运行 `python attention_profile.py` 将同步更新图 1 的正文 PDF、SVG 和预览。图 2 的来源为 `profiles_method`，其余四张图的来源为 `quantitative`。这些子目录保留历史导出行为；请在临时副本中运行其 `regenerate.py`，只将选定的图放回稿件目录。
