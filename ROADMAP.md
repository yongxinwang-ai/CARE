# Implementation of Elite Sampling for GRPO in EasyR1

## My proposal

**Primal–Dual Budgeter for Adaptive Multimodal Reasoning under Dual Budgets:
Unifying Direct Answer, Textual CoT, and Visual CoT with Optimal Stopping**

---

# 1. Motivation（为何需要双预算自适应）

* **过度视觉/文本开销**：多阶段启发式路由容易“又裁剪又长思考”，不必要地放大视觉 token 和 CoT token。
* **统一停机的缺口**：现有做法缺少一个把“要不要继续想/继续看”**一次性**决定的原则，难以对齐延迟/SLA。
* **Ground-R1 的契机**：Ground-R1 已经验证**两阶段 RL**可在**无 bbox 监督**下学会 zoom/crop、迭代细化与不确定性感知，并给出 `<think>/<box>/<answer>` 的可解释轨迹；我们只需在**动作层**引入“增益−成本”的最优停机，即可自然避免过度推理与过度视觉。

---

# 2. Contributions（本文要做什么）

1. **双预算最优停机框架**：同时约束**视觉预算**（视觉 token/裁剪/绘制）与**文本预算**（CoT/答案 token），单回合贪心选择动作，满足“**最大增益−λᵥ·视觉成本−λₜ·文本成本**”。
2. **动作级增益/成本学习**：为 `ANS / TEXT-k / CROP(b) / DRAW(op) / TOOL(t)` 提供轻量估计头，预测**信息增益 Δ**与**两类成本**，推理时**拍卖式**选择。
3. **与 Ground-R1 无缝结合**：保持其**两阶段 rollouts + 组归一 GRPO**不变，仅把奖励改为“**准确−双成本**”，证据仍写进 `<think>/<box>` 可视轨迹。
4. **可行性与理论保证**：给出对偶更新**逼近目标预算**的论证，并利用“多次 zoom 收益递减”的经验事实，给出贪心停机的**近似最优**理由（§5）。

---

# 3. Method（核心方法）

## 3.1 带双预算的约束优化

对策略 $\pi$：在**视觉预算** $B_v$ 与**文本预算** $B_t$ 下最大化正确率：

$$
\max_\pi \ \mathbb{E}[\text{Acc}]
\quad \text{s.t.} \quad
\mathbb{E}[\text{VisCost}] \le B_v,\ 
\mathbb{E}[\text{TextCost}] \le B_t.
$$

拉格朗日松弛（双对偶）：

$$
\max_\pi \ \mathbb{E}\!\left[\text{Acc} - \lambda_v(\text{VisCost}-B_v) - \lambda_t(\text{TextCost}-B_t)\right],\ \lambda_v,\lambda_t \ge 0.
$$

**单回合最优停机**：每一步，对候选动作 $a$ 预测**一步信息增益** $\Delta(a)$、**视觉成本** $c_v(a)$、**文本成本** $c_t(a)$。若

$$
\max_a\big[\Delta(a) - \lambda_v c_v(a) - \lambda_t c_t(a)\big] \le 0,
$$

则 **STOP**；否则执行性价比最高的动作。

> 与 Ground-R1 的两阶段“先证据裁剪、再答案合成”兼容：`CROP/DRAW` 发生在“证据”步骤，`TEXT-k/ANS` 发生在“答案/推理”步骤。

## 3.2 动作与成本模型（更细的记账）

**动作集合**

* `ANS`：直接作答（短答案）。
* `TEXT-k`：追加 $k$ 个 CoT token。
* `CROP(b)`：对原图裁剪框 $b$ 并回读高分辨率区域。
* `DRAW(op)`：在 ROI 上**画辅助线/平行/垂线/角标/测量**（矢量原语，渲染回读）。
* `TOOL(t)`：OCR/表格/检测等（可选，成本更高）。

**成本定义**（单位化到“token 等价成本”）

$$
\begin{aligned}
c_v(a) &= \underbrace{\Delta\#\text{visual tokens}}_{\text{重编码 patch 数}}
+ \underbrace{\kappa_{\text{crop}}\cdot \mathbb{I}[a=\text{CROP}]}_{\text{裁剪开销}}
+ \underbrace{\kappa_{\text{draw}}\cdot n_{\text{prims}}}_{\text{绘制/测量}} ,\\
c_t(a) &= \underbrace{\Delta\#\text{text tokens}}_{\text{新生成的 CoT/答案}}.
\end{aligned}
$$

> 说明：矢量 `DRAW` 若只反馈少量标量（角度/长度），可把渲染回读做成**低分辨率贴图**或延后渲染，从而使 $c_v$ 远小于再次 `CROP`。

**推荐初值（可调）**

* 视觉 patch 1 个=1，`CROP` 固定费 $\kappa_{\text{crop}}=32$，`DRAW` 每个原语 $\kappa_{\text{draw}}=4$。
* 文本 token 1 个=1（或按实际 token 价差做缩放）。

## 3.3 信息增益 Δ 的学习（多信号融合）

训练三个小头，推理时**加权和**：

* **熵降** $\Delta_{\text{ent}}$：预测执行 $a$ 后答案熵的降低。
* **一致性提升** $\Delta_{\text{cons}}$：候选少采样一致率↑或“答案↔证据注意力对齐”↑（可由 `<box>`/证据耦合信号监督）。
* **任务增益** $\Delta_{\text{task}}$：几何测角/读表/OCR 等**命中特征**带来的 logit 提升。

> 监督数据来自离线/在线轨迹：执行前后真实熵差、一致率差与命中事件作为标签；再做**分位回归+温度缩放**校准，抑制过乐观估计。

## 3.4 对偶更新（自适应 λ₍ᵥ,ₜ₎）

一次问题结束统计总成本 $C_v,C_t$，做

$$
\lambda_v \leftarrow [\lambda_v+\eta_v(C_v-B_v)]_+,\quad
\lambda_t \leftarrow [\lambda_t+\eta_t(C_t-B_t)]_+ .
$$

* **超支**→对应 λ 上升→下次更易早停；**节省**→λ 下降→允许多一步。
* SLA（严格/常规/宽松）映射到 $(B_v,B_t)$ 与 $(\lambda_v^0,\lambda_t^0)$。

## 3.5 训练：与 Ground-R1 对齐

* **SFT 预热（几小时）**：用已有/生成的可视 CoT 轨迹，拟合 $\hat\Delta,\hat c_v,\hat c_t$，并做**反事实剪枝**（能删的动作删，Δ 置 0）得“最小证据老师”。
* **R1 风格 RL（组归一 GRPO）**：沿用 Ground-R1 的**两阶段 rollouts**与**组归一优势**，但把轨迹奖励改为

  $$
  r = \text{Acc} + r_{\text{format}} - \lambda_v \cdot \text{VisCost} - \lambda_t \cdot \text{TextCost},
  $$

  让“**更省且对**”的轨迹优势变大；回合末更新 $\lambda_v,\lambda_t$。

## 3.6 推理伪代码（单回合停机）

```text
Input: q, image v, budgets (Bv,Bt), duals (λv,λt)
s ← encode(q, low-res(v)); H ← ∅; Cv←0; Ct←0
repeat:
  A ← {ANS, TEXT-k, CROP(b1..bK), DRAW(op1..opM), TOOL(t1..tL)}
  for a in A:  predict Δ(a), cv(a), ct(a)
  a* ← argmax_a [ Δ(a) − λv·cv(a) − λt·ct(a) ]
  if Δ(a*) − λv·cv(a*) − λt·ct(a*) ≤ 0: return finalize_answer(H)
  (H, s, Cv, Ct) ← execute(a*, H, s, Cv, Ct)   # 写入 <think>/<box>/矢量原语
end
# (离线) λv,λt ← dual updates using (Cv,Bv) and (Ct,Bt)
```

---

# 4. 可行性与证据

* **行为支撑**：Ground-R1 已证实**自动 zoom/再定位、迭代细化、不确定性感知**可通过 RL 自发涌现，无需 bbox 监督——这给 PDB 的“动作层优化”坚实基础。
* **分布特性**：Ground-R1 的推理统计显示**78% 样本两次 zoom 内解决，5% 无需 zoom**，符合“边际收益递减”假设，有利于**贪心停机**近似最优。
* **工程复用**：直接复用其**两阶段接口与 GRPO**，仅增补（Δ, c）估计头与对偶控制，训练与硬件规模按其实现复用（如 G1=4、G2=2 的配置）。

**证明要点（附录可展开）**

* KKT 条件 + 对偶上升 ⇒ 长期平均 $\mathbb{E}[\text{VisCost}],\mathbb{E}[\text{TextCost}]$ 贴近 $(B_v,B_t)$。
* 若 $\Delta$ 近似子模（经验上多次 crop/画线收益递减），则预算下的贪心给出 $(1-1/e)$ 近似。

---

# 5. 实验设计

## 5.1 数据与任务

* **VisCoT 训练子集**：按 Ground-R1 做法使用 1/50（≈8k）样本做预热与 RL；**评测**用 VisCoT 测试拆分（Doc/Text、Chart、FGVQA、关系等）。
* **通用基准**：MME / MM-Vet / SEED-Bench / MME-RWL / RealWorldQA / POPE。
* **几何子集（自建）**：含角度/垂直/平行/测长，提供少量“辅助线模板”以充实 `DRAW` 空间。

## 5.2 指标

* **二维前沿**：Accuracy vs (**VisCost**, **TextCost**)，报告 2D 帕累托与 AUC。
* **单轴曲线**：Accuracy\@VisBudget、Accuracy\@TextBudget。
* **选择性**：Over-think（可直答/短 CoT 却触发视觉）、Under-think（该用视觉却停在文本）。
* **可解释度**：轨迹长度、裁剪/绘制分布、答案-证据对齐分。

## 5.3 对比

* **Ground-R1（原版）**：无双预算停机。
* **启发式路由**：M0→M1→M2 阈值法。
* **R1-系方法**：Vision-R1 / LMM-R1（无显式证据成本）。
* **FAST/Grounded baselines**：仅快慢门控或仅证据引导，无对偶停机。

## 5.4 消融

* 一 λ（合并成本） vs **双 λ**；
* Δ 由（熵降 / 一致性 / 任务）不同组合；
* `DRAW` 矢量回读 vs 渲染回读的成本差异；
* 候选数 K（ROI/top-K 原语）；
* G1/G2 不同（复现实验表明 G1=4,G2=2 最优）。

## 5.5 实现细节（建议初值）

* **预算设定（默认 SLA=“常规”）**：$B_v=512$ 视觉等价 token、$B_t=160$ 文本 token。
* **对偶初值/步长**：$\lambda_v^0=0.02,\ \lambda_t^0=0.01$；$\eta_v,\eta_t\in[10^{-3},10^{-2}]$（按超支率自适应增减）。
* **动作成本表**（可起步参考）：

  * 每 16×16 patch = 1；`CROP` 固定费 32；`DRAW` 原语 4；`TEXT-k`：每 token = 1。
* **Rollouts**：沿用 Ground-R1 的 G1=4、G2=2；温度 1；max resp len 512。


## Current Tasks
action set当前只包含直接回答和cot两种形式，其他的action暂时不需要实现


### Active Development


## Development Workflow

1. **Task Planning**

- Study the existing codebase and understand the current state
- Update `ROADMAP.md` to include the new task
- Priority tasks should be inserted after the last completed task

2. **Task Creation**

- Study the existing codebase and understand the current state
- Create a new task file in the `/tasks` directory
- Name format: `XXX-description.md` (e.g., `001-db.md`)
- Include high-level specifications, relevant files, acceptance criteria, and implementation steps
- Refer to last completed task in the `/tasks` directory for examples. For example, if the current task is `012`, refer to `011` and `010` for examples.
- Note that these examples are completed tasks, so the content reflects the final state of completed tasks (checked boxes and summary of changes). For the new task, the document should contain empty boxes and no summary of changes. Refer to `000-sample.md` as the sample for initial state.

3. **Task Implementation**

- Follow the specifications in the task file
- Implement features and functionality
- Update step progress within the task file after each step
- Stop after completing each step and wait for further instructions

4. **Roadmap Updates**

- Mark completed tasks with ✅ in the roadmap
- Add reference to the task file (e.g., `See: /tasks/001-db.md`)

