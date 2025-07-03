# Cursor Prompt：三维超声体积 → 多增强 → Prompt 生成与筛选（Teacher 仅 MedSAM2）

---

## 1. 数据准备与预处理
**Prompt**：
> 给定一份三维超声体积，
> - 线性归一化灰度值至 [0,1]，可选中值滤波抑制斑点噪声；
> - 若输入为射频 (RF) 信号，先进行包络检测与对数压缩，再归一化；
> - 重采样至 1 mm³（或 0.5 mm）体素间距，并裁剪/填充至 128×128×128 体素；
> - 将标注对齐为体素级二值或多类格式；
> - 按 1:1 比例混合有标签与无标签样本，生成多样化 batch。

---

## 2. 多重几何增强
**Prompt**：
> 对预处理后的体积随机采样 N 种几何变换：旋转 (90°/180°/270°)、翻转、平移、缩放、弹性形变、仿射扭曲等；
> 生成 N 个增强体积，并保留正反向坐标映射。

---

## 3. Student 推理与粗掩膜生成
**Prompt**：
> 将每个增强体积依次输入 Student（Slicing SAM-3D）模型，输出前景概率图，并阈值化生成粗掩膜 $M_{coarse}=\mathbf{1}[p>0.5]$。

---

## 4. Prompt 生成与坐标还原

### 4.1 掩膜提示
**Prompt**：
> 使用 $M_{coarse}$ 作为掩膜提示，输入 Teacher（MedSAM2）模型。

### 4.2 点提示
**Prompt**：
> 在 $M_{coarse}$ 内采样若干质心或随机点 $(z,y,x)$，作为点提示输入 Teacher（MedSAM2）模型。

---

## 5. Teacher 伪标签生成
**Prompt**：
> 并行使用多种提示（掩膜提示 + 点提示）在 MedSAM2 Teacher 上解码，生成 N 个伪标签掩膜 $\{M_i\}_{i=1}^N$。

---

## 6. 方差筛选
**Prompt**：
> 对所有伪标签 $\{M_i\}$，在每个体素位置 $v$ 计算方差：
> \[\mathrm{Var}(v)=\frac{1}{N}\sum_{i}(M_i(v)-\bar M(v))^2, \bar M(v)=\frac{1}{N}\sum_i M_i(v).\]
> 设置阈值 $u_{th}$，保留 $\mathrm{Var}(v)<u_{th}$ 的体素，得到高置信伪标签 $M$。

---

## 7. 区域一致性与损失计算
**Prompt**：
> 计算：
> - 平凡监督损失 $L_{sup}$：交叉熵 + Dice；
> - 一致性损失 $L_{con}$：学生预测与 EMA Teacher 预测 MSE；
> - 区域一致性损失 $L_{rs}$：在高置信区域 $M$ 上，学生预测 $p_s$ 与伪标签 $p_f$ 的 MSE；
> - 总损失 $L_{total}=L_{sup}+\lambda L_{con}+\beta L_{rs}$，其中 $\lambda,\beta$ 随训练迭代 ramp-up。

---

## 8. EMA 教师模型更新
**Prompt**：
> 使用 EMA 更新：
> $$\theta' \leftarrow \alpha\theta' + (1-\alpha)\theta, \quad \alpha\approx0.99.$$  
> 将学生当前权重融入教师模型。

---

## 9. 训练循环
**Prompt**：
> 重复以下步骤直至收敛：
> 1. 采样带标注 $(x_L,y_L)$ 与无标签 $x_U$；
> 2. 对 $x_L$ / $x_U$ 应用弱 / 强增强；
> 3. 学生模型与 Teacher 并行推断，获取预测与伪标签；
> 4. 计算并反向更新学生参数；
> 5. EMA 更新教师模型；
> 6. 在验证集监控性能。

---

## 10. 最终推理
**Prompt**：
> 对新输入三维超声体，使用 EMA 教师模型或学生模型执行单次前向，直接输出分割结果，无需额外提示。


---

## 5. 区域一致性损失（高置信伪标签）

`L_rs-con = Σ M_conf·||P_s − P_f||² / Σ M_conf`&#x20;

---

## 6. **总损失与权重调度**

```math
L_total = L_sup + λ(t)·L_con + β(t)·L_rs-con
λ(t)=λ_max·sigmoid(t/T_r)  (Ramp-up,  T_r≈10 epochs)
β(t)=β_max·exp(−t/T_d)     (Ramp-down,T_d≈40 epochs)
```

典型 `λ_max=1`, `β_max=0.3`。Phase-1 全程启用。

---

## 7. Mean-Teacher 更新

`θ′ = 0.99·θ′ + 0.01·θ`   （每 iteration 同步）。验证集最高 Dice 的 θ′ 即最终模型。

---

## 8. 训练循环

1. 抽 `(x_L, y_L)` & `x_U` → 强增广给 Student，弱增广给 Teacher。
2. Student / Teacher 前向；生成 `M_coarse`、伪标签、`M_conf`。
3. 计算 `L_sup`, `L_con`, `L_rs-con`；反向只更新 Student。
4. **EMA** 同步 Teacher。
5. 循环到 Epoch 止，按 `poly_lr` 更新学习率。

---

## 9. 推理

* 切片滑窗 → Student 推理 → 重组 3D 掩膜。
* 显存预算（FP16）：`Student+TinySAM < 7 GB`；若加 MedSAM2，需 `~11 GB` + `torch.no_grad()`。
* 可选医生交互：MedSAM2 点/框 Prompt 精修 → 导出 `nii.gz` 或 `dcmseg`。

---

> **备忘**
>
> * 若显存不足：Patch Sliding + Grad Accum + AMP。
> * Prompt 抖动大：调高 `u_th` 或增大 N。

```

---

