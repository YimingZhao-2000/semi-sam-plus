## 🌀 残差“双 Conv3D”结构

> **公式**
> $y = g + \varphi(g)$，其中
> $\varphi = \text{Conv3D} \to \text{IN} \to \sigma \to \text{Conv3D} \to \text{IN}$

* **梯度永不消失**：$\nabla_g y = I + \nabla_g \varphi$，包含单位阵。
* **显存友好**：不膨胀通道数，只需两次 3-D 卷积。



### 🔹 总体五步流程

```
3D 体积  ──► 逐切片 (Slice) ──► 冻结的 SAM ViT-B 编码器
      ──► 堆叠为 Slice-Embedding F (H/16 × W/16 × D × 256)
      ──► 4 层「残差双卷积 + 上采样 ×2」轻量 3D 解码器
      ──► 输出与原尺寸一致的掩膜
```

### 🔹 轻量 3-D 解码器：单层公式

```
g   : 输入 3-D 特征 (C, D, H, W)
r1  = IN( W2 * σ( IN(W1 * g) ) )
r2  = IN( W0 * g )
g'  = U2( σ( r1 + r2 ) )    # U2 = 三维双线性上采样 ×2
```

* **W0 / W1 / W2**：3×3×3 Conv
* **IN**：InstanceNorm3d
* **σ**：LeakyReLU
* 计算 / 显存 皆 **O(H·W·D)**，无自定义 CUDA。

### 🔹 复合损失（单尺度）

$$
\boxed{ L = \text{CE} + (1 - \text{SoftDice}) }
$$

* **CE**：交叉熵
* **SoftDice**：

  $$
  \text{SoftDice}= \frac{2\langle Y,\hat Y\rangle}{\|Y\|_2^2+\|\hat Y\|_2^2}
  $$

### 🔹 多尺度深监督

* 4 级解码输出都计算上式损失，系数：

  $$
  \alpha_4 : \alpha_3 : \alpha_2 : \alpha_1 = 1 : \tfrac12 : \tfrac14 : \tfrac18,\;
  \sum\alpha_\ell = 1
  $$
* 总损失：

  $$
  L_{\text{tot}} = \sum_{\ell=1}^{4} \alpha_\ell L^{(\ell)}
  $$

---



## 🎯 Dice + Cross-Entropy 复合损失

```math
L^{(`)} = -\sum_{k,n} Y_{k,n}\ln \hat Y_{k,n}
          + \frac{1 - 2\langle Y,\hat Y\rangle}{\|Y\|_2^2 + \|\hat Y\|_2^2}
```

* 第 1 项 = **交叉熵**（≈ KL 散度）
* 第 2 项 = **Soft-Dice**；当 $\|Y\|=\|\hat Y\|$ 时就是余弦相似度。
* 线性可加：$\nabla(\lambda_1 L_1 + \lambda_2 L_2)=\lambda_1\nabla L_1+\lambda_2\nabla L_2$。

---

## 🔀 多尺度深监督（4 级）

```math
L_{\text{tot}} = \sum_{`=1}^4 \alpha_{`} L^{(`)},\quad
(\alpha_4,\alpha_3,\alpha_2,\alpha_1)=\tfrac1{15}(8,4,2,1)
```

> **链式法则**
> $\displaystyle \nabla_\theta L_{\text{tot}}   = \sum_{`} \alpha_{`}
>     \frac{\partial L^{(`)}}{\partial \hat Y^{(`)}}
>     \nabla_\theta f^{(`)}_\theta$

* **粗分辨率** → 给出全局梯度方向
* **细分辨率** → 精修边界，所有梯度在共享参数处自动累加，收敛稳定。

---

### ✅ 使用建议

1. **把本段 Markdown 放入 `docs/loss_and_decoder.md`** 以便 Cursor 自动索引。
2. 如需代码示例，在同一文件追加 PyTorch 伪代码（模块名保持注释关键词 “DiceLoss” / “ResDualConv3d” 等）。

> *以上内容完全来自内部 PDF：公式与编号已核对一致，可直接查阅原文以验证。*
