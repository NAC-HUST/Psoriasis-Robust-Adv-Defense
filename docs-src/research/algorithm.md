# 算法详解

SAMOO 算法的详细技术说明。

## 🧮 算法原理

### 多目标优化框架

SAMOO 在两个目标间寻求 Pareto 最优解：

**目标 1**: 攻击成功率最大化

$$\max f_{attack}(\mathbf{x}_{adv}) = P(\text{misclassify}(\mathbf{x}_{adv}))$$

**目标 2**: 扰动最小化

$$\min f_{sparse}(\mathbf{x}_{adv}) = \|\mathbf{x}_{adv} - \mathbf{x}_{orig}\|_0$$

### Pareto 前沿

在多个非支配解中选择最优方案。

---

完整算法文档待补充...
