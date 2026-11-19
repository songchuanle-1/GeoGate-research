# 换设备继续 GeoGate

登录有私有仓库权限的 GitHub 账号后，在新设备运行：

```bash
git clone https://github.com/songchuanle-1/GeoGate-research.git
cd GeoGate-research
git switch -c experiment/my-first-ablation
python -m analysis.flops_estimate --prune-layer 14 --keep-ratio 0.10
```

最后一行只需 Python 标准库。默认长度下应看到理论 LM FLOPs 约下降 **42.1%**、初始 prompt KV 约下降 **78.5%**。这个检查不运行模型。

| 任务 | 入口 |
| --- | --- |
| 看 ICML 当前稿 / 继续改稿 | [PDF](../../papers/icml2027/01_submission/GeoGate_ICML2027.pdf) · [main.tex](../../papers/icml2027/02_manuscript/main.tex)。在 `02_manuscript` 中用 pdfLaTeX、BibTeX、pdfLaTeX 两次。附录已合入正文。 |
| 找图 1 数据和绘图程序 | [attention_profile.py](../../papers/icml2027/03_materials/figure_sources/attention_profile.py) 从同目录下的冻结数据重画三个面板，输出到 ICML 稿的 `figures/`。 |
| 找 CVPR 旧版 | [正文](../../papers/cvpr2027/main.pdf) · [补充](../../papers/cvpr2027/supplement.pdf)；分别编译同目录的 `main.tex`、`supplement.tex`。两种会议格式属于同一项研究。 |
| 跑 LLaVA-3D / 3D-LLaVA 的 ScanQA | [图像 token 入口](../../adaptive_token/run_scanqa_geogate.py) · [超点 token 入口](../../adaptive_token/run_scanqa_geogate_3dllava.py)。这两个模型的上游包都叫 `llava`，请分别建环境。 |
| 找逐层对照、固定集合删除、跨任务或计时实验 | [16 组实验入口清单](experiment_index.json)。`additional_source_sha256` 记录新收录脚本的原始文件哈希。`analysis/score_mmscan_geogate.py` 的机器专用 Java 路径已去除，支持 `GEOGATE_JAVA_BIN`、`JAVA_HOME` 或 `PATH`。 |

**第一次跑 GPU 实验：** 按[依赖说明](../reproduction.md)准备对应上游模型、权重、场景和问题文件。先用同一份小规模问题文件跑通 native 和 L14/R10，再换完整评测集。以 LLaVA-3D 的 Linux 环境为例，先将 `BACKBONE`、`MODEL`、`QUESTIONS` 和 `SCENES` 设为本机相应路径：

```bash
export PYTHONPATH="$PWD:${BACKBONE:?set BACKBONE}${PYTHONPATH:+:$PYTHONPATH}"
run_dir="outputs/scanqa-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/code-commit.txt"
python -m pip freeze > "$run_dir/environment.txt"
for layer in -1 14; do
  python -m adaptive_token.run_scanqa_geogate \
    --model-path "${MODEL:?set MODEL}" --question-file "${QUESTIONS:?set QUESTIONS}" \
    --video-folder "${SCENES:?set SCENES}" --answers-file "$run_dir/layer_${layer}.jsonl" \
    --prune-layer "$layer" --keep-ratio 0.10 --seed 0 || break
done
```

`-1` 是未剪枝基线，`14` 是完整跑完 L14 后再剪枝。用同一输入、提示和种子配对；这个示例并不替代论文里专门的共享输入协议。高级脚本有些依赖历史清单、修改过的上游实现或场景资产，源码齐备不代表这些依赖齐备。

在另一台机器继续时，先提交代码、配置及可共享的汇总结果，`git push -u origin HEAD`；再在目标设备 `git fetch origin` 并切到同一实验分支。`outputs/` 默认忽略，模型权重、场景、逐条预测和本机配置不会随 Git 同步。
