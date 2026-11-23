# 换设备继续 GeoGate

要接着做 **11 月 22 日修订稿**的实验，先读[原始待办清单（P1–P4）](GeoGate_pending_experiments_zh.md)。[修订说明](GeoGate_revision_plan_zh.md)记录为什么改稿；[当前 PDF（含附录）](../../papers/icml2027/01_submission/GeoGate_ICML2027.pdf)和[对应的 LaTeX 压缩包](../../papers/icml2027/01_submission/GeoGate_ICML2027_source_revised.zip)可直接下载。四项均待做。只有一次实验机会时，先做 **P1：在未参与选参的场景上比较等分析成本的早剪和晚剪**。

登录有私有仓库权限的 GitHub 账号后，在新设备运行：

```bash
git clone https://github.com/songchuanle-1/GeoGate-research.git
cd GeoGate-research
git switch -c experiment/p1-heldout-cost
python -m unittest tests.test_cost_controls_v4 -v
python -m analysis.flops_estimate --prune-layer 14 --keep-ratio 0.10
```

两条 Python 命令只需标准库：前者将逐输入的等成本预算与穷举解核对，后者检查默认 R10 配置的理论 LM FLOPs 与初始 prompt KV（约下降 **42.1%** 和 **78.5%**）。这里没有加载模型或评测质量。已有副本且当前在干净 `main` 时，先执行 `git pull --ff-only origin main`，再建实验分支。

### 准备哪项实验

| 优先级 | 在新服务器上的第一步 | 何时可以报告结果 |
| --- | --- | --- |
| **P1：新场景早晚剪** | 准备 LLaVA-3D-7B、授权场景的原生 20 帧输入、问题和参考答案。先只用场景与问题 ID 审查重叠，冻结划分、协议和容差，再提取逐题长度。[预算](../../analysis/freeze_cost_controls_v4.py)、[四条件共享输入 runner](../../adaptive_token/run_cost_controls_v4.py) 和[评分脚本](../../analysis/score_cost_controls_v4.py)均按旧 ScanQA 的 **4,675 题**校验；应在新分支按冻结清单适配三处，包括 smoke 场景和统计中的场景数。不要只删数量检查。 | 原生、L2、L8、L14/128 四组逐题配齐并通过输入、保留数和成本偏差核对；给出 L14−L2 与 L14−L8 的场景聚类配对 EM 区间。 |
| **P2：固定保留集合** | 从[固定删除 runner](../../analysis/run_fixed_deletion_v21.py)复用索引与深度校验，另建只保留同一 top 10% 的分支。 | 排名来源 L8/L14/L20 × 执行层 L8/L14/L20 的九格和同流程原生基线齐全，逐题索引一致。只解释离线质量。 |
| **P3：三方法质量与耗时** | 并排检查[共享输入质量 runner](../../adaptive_token/run_shared_input_quality_v3.py)与[计时 runner](../../adaptive_token/run_scanqa_timing_boundaries_v4.py)，先确认官方 3DZip 栈可运行，再接入同时产出答案和完整应用时间的新流程。 | 同一批请求、硬件、热身和轮换顺序下，原生、官方 3DZip、GeoGate 都有 EM、CIDEr 和分会话耗时；计入预处理、视觉/压缩、选择/缓存与生成。 |
| **P4：OpenEQA-HM3D 保质** | 按[冻结协议](OPENEQA_HM3D_V14_PROTOCOL_20260913.md)，准备官方问题 JSON、授权 RGB-D、位姿、内参和可锁定版本的官方 LLM-Match 评分器。用下方命令先做 CPU 资产检查，再按协议依次试跑。 | 63 个 episode 预检全部通过、两次分阶段试跑通过、557 对输出通过[结构审计](../../analysis/audit_openeqa_hm3d_v14.py)后取得官方评分。用[汇总脚本](../../analysis/score_openeqa_hm3d_v14.py)按 episode 重采样；95% 区间下界 ≥ −2.0 LLM-Match 分才达到非劣效标准。 |

P4 的资产预检只用 NumPy 和 Pillow。将 `OPENEQA_JSON` 设为官方 JSON 文件，`EPISODES` 设为本机包含 `hm3d-v0/` 的数据目录。输出保存在本机；非零退出时先修复输入，不启动模型。

```bash
python -m analysis.preflight_openeqa_hm3d_v14 \
  --dataset "${OPENEQA_JSON:?set OPENEQA_JSON}" \
  --episode-root "${EPISODES:?set EPISODES}" \
  --out "outputs/openeqa-$(date +%Y%m%d-%H%M%S)/preflight.json"
```

每次正式运行保存 Git 提交、上游模型和权重版本、环境、数据清单及哈希、实验协议、实际命令、逐题输出和汇总表。原始场景、模型权重和逐题预测保留在获授权的存储中；跨设备同步实验分支和可共享的汇总结果。

### 论文与已有代码

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
