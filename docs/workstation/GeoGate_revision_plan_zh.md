# GeoGate 投稿前修改意见：不新增实验版

> **执行说明：** 下文保留为改稿前的审稿建议，图表编号均是旧稿编号。摘要、引言、固定集合小节、外部比较与结论已按本轮修改；保留原论文标题。新增实验均未执行，后续请使用单独的[待做实验清单](GeoGate_pending_experiments_zh.md)。

**适用稿件：** 当前 `main.tex` 及其 `sec/` 正文和附录。下文表号、图号均指修改前的 PDF；调整顺序后沿用原有 LaTeX 标签，不要手工固定编号。

**建议先做一个选择：把这篇论文写成“如何判断晚剪是否真的有用”的实证研究，让 GeoGate 作为被检验的具体方案。** 原稿的优点恰恰是做了一个可能推翻自身解释的对照，并如实报告了结果。现在摘要写了算法、成本和迁移，却没有写这个发现；引言也把已有评分器上的调度方案列在第一贡献。优先解决这个错位，不必再增加一页限制说明。

本轮全部修改只使用已经发表在稿件中的结果，不需要 GPU、原始预测、重新评分或远程服务器。它能让贡献和证据更匹配，**不能替代独立验证，也不保证提高审稿分数**。

## 一、先改三处：摘要、第一贡献、表 5 的位置

### 1. 标题与摘要：让读者先看到科学问题

当前标题已有 *Evaluating*，可以保留。若希望更直接，建议改为 **Evaluating Late-Layer Token Pruning in 3D Multimodal Models: Token Selection versus Deletion Depth**。对应修改 `main.tex` 中的 `\icmltitle` 和短标题 `\icmltitlerunning`。不建议用 *optimal*、*lossless* 或 *geometry-aware*：前两者没有相应结果，最后一个也不是实际使用的评分规则。

**`sec/0_abstract.tex` 可参考的英文摘要**（可直接放入现有 `abstract` 环境；匿名材料说明按实际发布内容另行保留）：

```latex
Pruning at different layers changes both when spatial tokens are removed and which tokens attention selects. We study these two effects in 3D multimodal models using GeoGate, a training-free schedule that applies FastV's scorer after block L14 and compacts the prompt cache before decoding. On LLaVA-3D, deleting each layer's own top-ranked tokens causes the greatest excess damage over random deletion at L14 among the three layers tested. Yet when the same L14-selected tokens are deleted at L8 rather than L14, exact match drops by a further 1.50 points on ScanQA validation. Thus the own-set peak does not identify the most damaging deletion depth. On the same validation split used for schedule selection, retaining 128 tokens at L14 outperforms selected L2 and L8 controls by 1.22 and 1.03 exact-match points at matched analytical language-model cost. At 10\% retention, analytical language-model FLOPs fall by 42.1\%, but application time falls by only 0.8--1.4\% on one H100. On SQA3D validation, the same schedule reduces 3D-LLaVA exact match by 0.95 points. These results distinguish the value of a selected token set from the choice of when to prune it.
```

这里保留两个不同预算：**128 个**对应等分析成本对照；**10%** 对应默认配置、迁移和系统测量。不要为了简洁合并成同一个配置。若摘要需要减字，先删实现细节，不删开发集限定或表 5 的对照。

### 2. 引言：把三个贡献改成三个有答案的问题

在 `sec/1_intro.tex` 保留 FastV、HiMAP、PyramidDrop 的定位。现有第三段已经指出“同一评分规则不等于同一 token 集合”，不必再加背景。把它与下列三项连接起来，替换目前以 *A training-free late-layer pruning schedule* 为首项的贡献列表：

1. **固定身份后，删除深度还重要吗？** 仍然重要。对 L14、L20 选出的两个集合，L8 删除比 L14 删除造成更大损失，L14 又比 L20 更有害。L8 选出的集合没有明确排序。**贡献是识别出比较中的混杂，并用固定集合检验它**，不是发现所有 token 的统一最佳层。
2. **如果目标是保留少量 token，晚剪在相同计算预算下值得吗？** 在用于选参的 LLaVA-3D ScanQA 验证集上，L14/128 优于两个指定的等分析成本早剪对照。证明的是这组比较；不是完整的质量–成本前沿，也不是跨域验证。
3. **开发集优势能否转化为跨任务质量与应用收益？** 不总能。正文保留一个迁移反例和一对成本数字即可，其余放回对应表。保留全层 KV 压缩这一实现特点，但不把它单独计作已验证的质量贡献。

### 3. 实验：先解释表 5，再展示调度成绩

`sec/5_experiments.tex` 的最小改法：把现有 **Token-Set Relevance and Deletion Depth** 整个小节移到 **Evaluation Setup** 后、**Quality under Controlled Comparisons** 前。把现有图 3 及其解释放在等成本表附近；官方 3DZip 放在这两者之后，作为不同问题的对照。不需要重画图，也不需要增加新表。保留 `tab:fixedset`、`fig:layercurve`、`tab:costmatched` 等原标签。

表 5 前后建议采用下列叙述，解决最容易出现的符号误读：

> **English draft:** Selecting and deleting at the same layer changes two things at once: the tokens selected and the depth at which they are removed. We therefore replay each deletion set at three depths. For the L14-selected set, deletion at L8 causes 1.50 more EM points of damage than deletion at L14; the L20-selected set gives the same ordering. None of the three source sets meets the prespecified criterion for an L14-specific damage maximum.

紧接一句：*This intervention deletes the selected 10% of tokens; it does not test whether retaining only those tokens is sufficient.* 这里每层自选结果使用“相对随机删除的额外损伤”，固定集合结果使用“相对原生模型的损伤”。**不要将两张表的数相减，声称得到了 token 身份的贡献百分比。** 表 5 当前列方向是 L14 减 L8，因此表中是 **−1.50**；英文句子换了比较方向，写 **1.50 more at L8** 是一致的。

## 二、逐条回应审稿意见：现在具体能改什么

| 审稿疑问 | 本轮改稿动作 | 不应写成 |
|---|---|---|
| 是否只在选参集上赢？ | 在表 3 的结果句中同时写出模型、ScanQA validation、L14/128、L2/L8 和 matched **analytical LM cost**。在跨域段开头明确：外部对照是原生模型，不是等成本早剪。 | “跨域证实晚剪优于早剪”；或者把同一验证集上的留一场景分析称作独立复现。 |
| 为什么一定是 L14？ | `sec/3_attention_analysis.tex` 保留 L13→L14 的跃升；图 3 旁写明 L14 相对 L15/L16/L18 的优势未得到证实。外部分析中，234/234 个场景出现正向跃升，但跃升强度并未成为经验证的质量预测指标。 | “L14 是最优层”；“相邻层等价”；或“注意力回升可自动确定安全剪枝点”。 |
| 负结果对读者有什么用？ | 在 SQA3D 段先写两折均未通过预设的 −1 EM 非劣效门槛，再写合并估计 −0.95；指出合并结果不是第三个独立验证。在 MMScan 保留 ΔEM = −0.061、95% CI [−0.298, 0.177]，与 SQA3D 分开解释。 | “两折都显著变差”或“MMScan 无损”。未通过非劣效、检出下降、未通过优效是三个不同判断。 |
| 与 3DZip 比究竟赢在哪里？ | 表 4 保留 **等 128 个空间 token** 时 +2.91 EM、+8.24 CIDEr 的实证优势。表 7 单独保留相对原生模型的一张 H100 应用计时。两处都点名对手，避免使用未限定的 *more efficient*。 | 用表 4 的质量和表 7 的速度拼成“比官方 3DZip 又快又准”。 |
| 复现性是否足够？ | 在 `sec/supp/repro_index.tex` 保留三层区分：论文包可重建 PDF；已保存答案可供何种核查取决于实际开放内容；历史 matched-input QA 运行脚本版本未找回。摘要中的代码声明应与匿名仓库实际可下载内容一致。 | “所有主表可一键复现”；也不要反过来写成“所有实验代码均已丢失”。 |

## 三、结论怎么收，哪些文字可以删

**`sec/6_conclusion.tex` 首段可改为：**

> On the ScanQA validation split used for selection, delaying pruning improved exact match over two selected early schedules at matched analytical LM cost. The fixed-set intervention gives a different lesson: a layer can select tokens whose removal matters without being the depth at which removing them is most damaging. We therefore recommend reporting the source of the token ranking separately from the layer of intervention. Whether late pruning beats cost-matched early pruning on unseen scenes remains open.

随后用两句话交代迁移损失与实测应用时间；保留不完整的历史训练记录、复现边界在附录中的入口。**删减优先顺序：** 先精简表 1 中与后文重复的说明，再合并对“分析成本不是运行时间”的重复提醒；保留表 3 和表 5 的区间、符号定义、选择所用数据。不要通过删掉负结果来腾版面。

## 四、只靠改稿解决不了的问题

以下三项不要列为本轮待办，也不要在回复审稿人时承诺已经完成。若今后只有一次计算资源，优先第一项。

1. **新场景上的等分析成本早晚剪对照。** 需要在未参与选参的场景上冻结 L2/L8/L14 的策略并配对比较。它直接检验摘要的正向发现；现有 MMScan 原生对照不能替代。
2. **固定保留集合的跨层干预。** 已有结果固定的是被删除的 10%。将完全相同的 10% 作为唯一保留集合，在不同深度施加，是另一个问题。即使未来做，也应注明使用晚层排名指导早层保留是离线诊断，不是可部署的在线算法。
3. **原生、官方 3DZip、GeoGate 的同流程质量与耗时。** 需要同输入、同生成配置、相同计时边界，并在同一流程记录答案和时间。现在可报告等 token 质量优势，不能排出三者实际延迟的名次。

## 五、交稿前的无 GPU 检查

- [ ] 只读标题、摘要和贡献第一条，就能说出固定集合实验改变了哪一种解释，而不只是记住“L14、10%”。
- [ ] 全稿搜索 `128`、`10\%`、`matched`、`optimal`、`preserv`、`speedup`；逐项确认预算、比较对象、数据划分和计时范围。
- [ ] 表 5 的表头、表注和正文对损伤正负号的解释相同；任何“删除”结论都没有被改写为“保留充分”。
- [ ] 每个数只和同一实验流程的原生基线比较。外部结果没有被误写成早剪对照；事后分析没有被改写成预设检验。
- [ ] 重排后只需编译论文并查看 PDF：正文结尾、图表首次引用、表 5 新位置、参考文献和页数。图表浮动以 PDF 为准，不用缩字号或改模板来塞内容。

**时间有限时，到这里就够：改摘要、改引言三条贡献、前移固定集合小节、在结论写明未验证的外部早晚剪对照。** 这四步对应上轮弱拒的核心理由，且不依赖任何新实验。
