"""
0–9 手写数字识别（NumPy 手写神经网络 + 学习向详细注释）
============================================================

这份代码做两件事：
1. 用 sklearn 的 digits（8×8 手写数字）训练一个多层感知机（MLP）
2. 对你自己的图片做预测（OpenCV 预处理成 8×8 灰度）

为什么不用现成的 keras.fit？
  为了把「前向传播 / 损失 / 反向传播 / 梯度下降 / 正则化」都摊开写在代码里。
  读代码时请对照注释里的公式，比只调库更能建立直觉。

阅读路线（建议按这个顺序）：
  1) Softmax + 交叉熵损失          -> 模型「错得有多惨」怎么量化
  2) 前向传播 forward               -> 输入怎么变成 10 类概率
  3) 反向传播 backward              -> 梯度从哪来
  4) 梯度下降 / mini-batch SGD      -> 参数怎么更新
  5) L1 / L2 正则化                 -> 怎么抑制过拟合
  6) 训练曲线 train vs val          -> 欠拟合 / 过拟合怎么从曲线看出来
  7) predict_image                  -> 真实图片怎么喂进网络

依赖:
  pip install -r requirements.txt

用法:
  # 训练（默认会画损失曲线并保存权重）
  python digit_recognize.py --train

  # 对比：无正则 / L2 / 故意欠拟合（隐藏层很小）
  python digit_recognize.py --train --demo-fit

  # 对一张图识别（白底黑字或黑底白字都尽量兼容）
  python digit_recognize.py --predict your_digit.png

  # 指定已保存的权重
  python digit_recognize.py --predict your_digit.png --weights models/digit_mlp.npz
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 标准库
# ---------------------------------------------------------------------------
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 第三方库
# numpy      : 张量运算；本项目的「深度学习框架」就是它
# matplotlib : 画训练/验证损失，观察欠拟合与过拟合
# cv2        : 读你自己的图片并缩放到 8×8
# sklearn    : 只借数据集 + 划分训练/验证；模型本身是手写的
# ---------------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DEFAULT_WEIGHTS = MODEL_DIR / "digit_mlp.npz"
CURVE_PATH = ROOT / "output" / "digit_train_curves.png"


# ===========================================================================
# 一、核心概念速览（先建立地图，再看实现）
# ===========================================================================
# 1) 监督学习分类任务
#    输入 x：一张数字图拉平后的向量（这里 8×8=64 维）
#    标签 y：0~9 的类别
#    目标 ：学一个函数 f(x; W) ≈ P(y|x)
#
# 2) 损失函数 Loss
#    衡量「当前参数下，预测和真值差多少」。训练 = 尽量把 Loss 变小。
#    本项目用「Softmax + 交叉熵（Cross-Entropy）」——多分类标准搭配。
#
# 3) 梯度下降 Gradient Descent
#    参数更新：W <- W - lr * ∂Loss/∂W
#    lr（学习率）太大：震荡/发散；太小：学得很慢。
#    Mini-batch SGD：每次只用一小批样本估梯度，更快、有噪声反而利于跳出尖锐最小值。
#
# 4) 正则化 Regularization
#    L2：在 Loss 上加 λ/2 * Σ W²，让权重大的参数「吃亏」→ 权重趋向变小、更平滑
#    L1：加 λ * Σ |W|，推动部分权重变 0 → 更稀疏（特征选择感）
#    直觉：约束模型复杂度，减轻过拟合。
#
# 5) 欠拟合 Underfitting vs 过拟合 Overfitting
#    欠拟合：模型太弱 / 训练不够 → 训练集和验证集都差
#    过拟合：死记训练样本 → 训练集很好，验证集变差（两曲线张开）
#    常用对策：更多数据、降复杂度、Dropout、早停、L1/L2、数据增强
# ===========================================================================


# ===========================================================================
# 工具函数：One-hot、Softmax、交叉熵
# ===========================================================================

def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    把类别标签变成 one-hot 矩阵。

    例子:
        y = [3, 0]  ->
        [[0,0,0,1,0,0,0,0,0,0],
         [1,0,0,0,0,0,0,0,0,0]]

    为什么要用 one-hot？
        交叉熵里要用「真实类位置为 1、其余为 0」的目标分布。
    """
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Softmax：把任意实数分数变成「和为 1 的概率分布」。

        softmax(z)_i = exp(z_i) / Σ_j exp(z_j)

    数值稳定技巧:
        先减每行最大值，再 exp。数学上等价，但能避免 exp 爆炸成 inf。
    """
    # keepdims=True 保持形状以便广播
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """
    多分类交叉熵（对 batch 取平均）。

        CE = - (1/N) Σ_n Σ_k y_nk * log(p_nk)

    当 y 是 one-hot 时，内层其实只留下「真实类那一项」：
        CE = - (1/N) Σ_n log(p_n, y_n)

    eps：防止 log(0)。
    """
    p = np.clip(probs, eps, 1.0)
    return float(-(y_true * np.log(p)).sum(axis=1).mean())


# ===========================================================================
# 二、手写两层 MLP（多层感知机）
# ===========================================================================
# 结构（默认）:
#   输入 64 维（8×8 像素拉平）
#     -> Linear(64, H) + ReLU
#     -> Linear(H, 10)          # 10 个类别的 logits
#     -> Softmax -> 概率
#
# ReLU(z) = max(0, z)
#   作用：引入非线性。如果全是线性层叠在一起，整体仍等价于一个线性变换，
#   表达能力不够（对「弯弯曲曲的决策边界」无能为力）。
# ===========================================================================

@dataclass
class TrainConfig:
    """训练超参数：把「旋钮」集中放一起，方便做对比实验。"""

    hidden_size: int = 64          # 隐藏层宽度；太小易欠拟合，太大易过拟合
    epochs: int = 80               # 把全部训练数据过几遍
    batch_size: int = 64           # mini-batch 大小
    lr: float = 0.2                # 学习率 learning rate
    l1: float = 0.0                # L1 系数 λ1；0 表示不用
    l2: float = 1e-4               # L2 系数 λ2；常用从小的正数开始试
    seed: int = 42                 # 固定随机种子，结果可复现
    val_ratio: float = 0.2         # 验证集比例


class DigitMLP:
    """
    两层全连接网络 + Softmax 分类头。

    参数矩阵形状约定（行向量样本 batch）：
        X      : (N, Din)
        W1     : (Din, H)
        b1     : (H,)
        W2     : (H, 10)
        b2     : (10,)
    """

    def __init__(self, din: int = 64, hidden: int = 64, num_classes: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # 权重初始化（很重要，却常被忽略）
        # He 初始化思路：ReLU 网络常用 N(0, 2/fan_in)，避免一开局就饱和/消失。
        # 若全初始化成 0：隐藏层神经元对称，梯度相同，学不动（对称坍缩）。
        # ------------------------------------------------------------------
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / din), size=(din, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, num_classes))
        self.b2 = np.zeros(num_classes, dtype=np.float64)

        # 前向时缓存中间量，反向传播要用（这就是「计算图」的手工版）
        self.cache: dict[str, np.ndarray] = {}

    # ----------------------------------------------------------------------
    # 前向传播 Forward
    # ----------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X -> 概率 (N, 10)

        逐步:
            z1 = X @ W1 + b1
            a1 = ReLU(z1)
            logits = a1 @ W2 + b2
            probs = softmax(logits)
        """
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)  # ReLU
        logits = a1 @ self.W2 + self.b2
        probs = softmax(logits)

        self.cache = {"X": X, "z1": z1, "a1": a1, "logits": logits, "probs": probs}
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回预测类别（概率最大的那一类）。"""
        probs = self.forward(X)
        return probs.argmax(axis=1)

    def loss(
        self,
        probs: np.ndarray,
        y_true_oh: np.ndarray,
        l1: float = 0.0,
        l2: float = 0.0,
    ) -> float:
        """
        总损失 = 数据损失（交叉熵） + 正则项

            L = CE(probs, y) + λ1 * Σ|W| + (λ2/2) * Σ W²

        注意：一般不对 bias 做很强的正则；这里为简单起见只正则 W1/W2。
        「除以 2」是为了求导后 L2 梯度正好是 λ2 * W，写起来干净。
        """
        data_loss = cross_entropy(probs, y_true_oh)
        reg = 0.0
        if l1 > 0:
            reg += l1 * (np.abs(self.W1).sum() + np.abs(self.W2).sum())
        if l2 > 0:
            reg += 0.5 * l2 * ((self.W1 ** 2).sum() + (self.W2 ** 2).sum())
        return data_loss + reg

    # ----------------------------------------------------------------------
    # 反向传播 Backward：手动求梯度
    # ----------------------------------------------------------------------
    def backward(self, y_true_oh: np.ndarray, l1: float = 0.0, l2: float = 0.0) -> dict[str, np.ndarray]:
        """
        根据缓存的前向结果，计算 ∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2。

        Softmax + 交叉熵 的组合有一个非常漂亮的结果：
            ∂CE/∂logits = (probs - y_true) / N
        这就是为什么大家总说「这两货天生一对」。

        ReLU 的局部梯度：
            dReLU/dz = 1 (z>0) else 0
        反向时：上游梯度乘这个掩码即可。
        """
        X = self.cache["X"]
        z1 = self.cache["z1"]
        a1 = self.cache["a1"]
        probs = self.cache["probs"]
        n = X.shape[0]

        # ---- 输出层梯度 ----
        dlogits = (probs - y_true_oh) / n          # (N, 10)
        dW2 = a1.T @ dlogits                       # (H, 10)
        db2 = dlogits.sum(axis=0)                  # (10,)

        # ---- 反传到隐藏层 ----
        da1 = dlogits @ self.W2.T                  # (N, H)
        dz1 = da1 * (z1 > 0)                       # ReLU 掩码
        dW1 = X.T @ dz1                            # (Din, H)
        db1 = dz1.sum(axis=0)                      # (H,)

        # ---- 正则化对梯度的贡献 ----
        # L2: ∂(λ2/2 Σ W²)/∂W = λ2 * W
        # L1: ∂(λ1 Σ |W|)/∂W = λ1 * sign(W)；W=0 处次梯度取 0
        if l2 > 0:
            dW1 = dW1 + l2 * self.W1
            dW2 = dW2 + l2 * self.W2
        if l1 > 0:
            dW1 = dW1 + l1 * np.sign(self.W1)
            dW2 = dW2 + l1 * np.sign(self.W2)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """
        梯度下降参数更新（SGD 一步）:

            θ <- θ - lr * ∂L/∂θ

        这就是「学习」在代码里的全部形态：沿着损失下降的方向挪一点点参数。
        """
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    # ----------------------------------------------------------------------
    # 存取权重
    # ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: Path) -> None:
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]


# ===========================================================================
# 三、数据准备
# ===========================================================================

def load_digit_data(val_ratio: float = 0.2, seed: int = 42):
    """
    加载 sklearn digits：1797 张 8×8 灰度手写数字（像素值约 0~16）。

    预处理:
        1) 拉平成 64 维
        2) 缩放到 0~1（除以最大值）—— 让梯度尺度更友好
        3) 划分 train / val
           验证集作用：模拟「没见过的数据」，用来发现过拟合。
    """
    ds = load_digits()
    X = ds.data.astype(np.float64)           # (N, 64)
    y = ds.target.astype(np.int64)           # (N,)
    X = X / X.max()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed, stratify=y
    )
    # stratify=y：各类比例在 train/val 里大致一致，避免某类全挤在一边
    return X_train, X_val, y_train, y_val


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# ===========================================================================
# 四、训练循环（Mini-batch SGD）
# ===========================================================================

def train_mlp(cfg: TrainConfig, verbose: bool = True) -> tuple[DigitMLP, dict]:
    """
    完整训练流程，并返回历史曲线，便于观察欠/过拟合。

    每个 epoch:
        1. 打乱训练集（降低样本顺序带来的偏差）
        2. 按 batch 做：forward -> loss/backward -> step
        3. 在整个 train/val 上评估 loss 与 accuracy
    """
    X_train, X_val, y_train, y_val = load_digit_data(cfg.val_ratio, cfg.seed)
    y_train_oh = one_hot(y_train)
    y_val_oh = one_hot(y_val)

    model = DigitMLP(din=X_train.shape[1], hidden=cfg.hidden_size, seed=cfg.seed)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "config": asdict(cfg),
    }

    n = X_train.shape[0]
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(1, cfg.epochs + 1):
        # ---- mini-batch 训练 ----
        idx = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            batch = idx[start : start + cfg.batch_size]
            xb = X_train[batch]
            yb = y_train_oh[batch]

            probs = model.forward(xb)
            # loss 在这里主要是概念完整；真正驱动更新的是 backward 的梯度
            _ = model.loss(probs, yb, l1=cfg.l1, l2=cfg.l2)
            grads = model.backward(yb, l1=cfg.l1, l2=cfg.l2)
            model.step(grads, lr=cfg.lr)

        # ---- epoch 结束：全量评估（监控用，不更新参数）----
        train_probs = model.forward(X_train)
        val_probs = model.forward(X_val)
        train_loss = model.loss(train_probs, y_train_oh, l1=cfg.l1, l2=cfg.l2)
        val_loss = model.loss(val_probs, y_val_oh, l1=cfg.l1, l2=cfg.l2)
        train_acc = accuracy(y_train, train_probs.argmax(axis=1))
        val_acc = accuracy(y_val, val_probs.argmax(axis=1))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs):
            print(
                f"epoch {epoch:3d}/{cfg.epochs}  "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}  "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

    return model, history


# ===========================================================================
# 五、可视化：从曲线读欠拟合 / 过拟合
# ===========================================================================

def plot_history(history: dict, save_path: Path = CURVE_PATH, title: str = "Digit MLP") -> None:
    """
    画 train/val 的 loss 与 accuracy。

    怎么读图（超重要）:
      - 两条 loss 都很高、降不动          -> 欠拟合（模型弱 / lr 不合适 / 训太少）
      - train_loss 持续降，val_loss 回升  -> 过拟合（开始死记训练样本）
      - 两条一起稳步下降并接近            -> 拟合较健康
      - 加 L2 后 val 更好、train 略差     -> 正则在起作用（用一点训练误差换泛化）
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"训练曲线已保存: {save_path.resolve()}")


def demo_under_over_fitting() -> None:
    """
    三个对照实验，帮助「看见」欠拟合 / 过拟合 / 正则化。

    A) 欠拟合：隐藏层极小（hidden=4），表达能力不够
    B) 易过拟合：隐藏层很大 + 不设正则 + 多训几轮
    C) L2 正则：同样大网络，加 L2，看验证集是否更稳

    注意：digits 数据集本身不大也不极难，过拟合不一定非常夸张，
    但趋势一般仍能从 train/val 差距上看出来。
    """
    print("\n===== 对照实验 A：故意欠拟合（hidden=4）=====")
    under_cfg = TrainConfig(hidden_size=4, epochs=60, lr=0.2, l1=0.0, l2=0.0)
    _, hist_under = train_mlp(under_cfg)
    plot_history(hist_under, ROOT / "output" / "curve_underfit.png", title="Underfit hidden=4")

    print("\n===== 对照实验 B：大网络 + 无正则（更容易过拟合）=====")
    over_cfg = TrainConfig(hidden_size=256, epochs=120, lr=0.15, l1=0.0, l2=0.0)
    _, hist_over = train_mlp(over_cfg)
    plot_history(hist_over, ROOT / "output" / "curve_overfit.png", title="Overfit-prone no-reg")

    print("\n===== 对照实验 C：大网络 + L2 正则 =====")
    l2_cfg = TrainConfig(hidden_size=256, epochs=120, lr=0.15, l1=0.0, l2=2e-3)
    _, hist_l2 = train_mlp(l2_cfg)
    plot_history(hist_l2, ROOT / "output" / "curve_l2.png", title="Large net + L2")

    def gap(h):
        return h["train_acc"][-1] - h["val_acc"][-1]

    print("\n----- 结果解读（最后一轮 train_acc - val_acc，越大越像过拟合）-----")
    print(f"A 欠拟合:   train={hist_under['train_acc'][-1]:.3f} val={hist_under['val_acc'][-1]:.3f} gap={gap(hist_under):+.3f}")
    print(f"B 无正则:   train={hist_over['train_acc'][-1]:.3f} val={hist_over['val_acc'][-1]:.3f} gap={gap(hist_over):+.3f}")
    print(f"C L2正则:   train={hist_l2['train_acc'][-1]:.3f} val={hist_l2['val_acc'][-1]:.3f} gap={gap(hist_l2):+.3f}")
    print(
        "提示：A 通常 train/val 都不高；B 的 gap 往往大于 C；"
        "C 用一点点训练精度换更稳的验证表现。"
    )


# ===========================================================================
# 六、自定义图片预测
# ===========================================================================

def preprocess_digit_image(path: str) -> np.ndarray:
    """
    把任意输入图变成模型能吃的 (1, 64) 向量。

    步骤:
      1) 灰度读取
      2) 自动判断「黑底白字 / 白底黑字」，统一成「黑底白字」
         （和 digits 数据集更接近：笔画亮、背景暗）
      3) 缩放到 8×8
      4) 归一化到 0~1，并 reshape 成 (1, 64)

    实用建议:
      - 尽量让数字居中、四周留边
      - 笔画清晰、不要太细
      - 一张图里只放一个数字
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    # 若平均亮度偏高，多半是白底黑字 -> 反色
    if float(img.mean()) > 127:
        img = 255 - img

    # 轻微阈值，抑制底噪；再 resize
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)

    # digits 原始像素大约 0~16，我们训练时除以 max 变成 0~1
    # 这里直接 /255 到 0~1，尺度与训练一致即可
    x = img.astype(np.float64) / 255.0
    return x.reshape(1, -1)


def predict_image(path: str, weights: Path = DEFAULT_WEIGHTS, hidden_size: int = 64) -> tuple[int, np.ndarray]:
    """
    加载权重并对单张图预测。

    返回:
        pred  : 预测数字 0~9
        probs : 10 类概率，便于看模型有多「犹豫」
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"找不到权重 {weights}。请先运行: python digit_recognize.py --train"
        )

    model = DigitMLP(din=64, hidden=hidden_size)
    model.load(weights)
    x = preprocess_digit_image(path)
    probs = model.forward(x)[0]
    pred = int(probs.argmax())
    return pred, probs


# ===========================================================================
# 命令行
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="0-9 数字识别（手写 MLP + 教学注释）")
    p.add_argument("--train", action="store_true", help="训练模型并保存权重/曲线")
    p.add_argument("--demo-fit", action="store_true", help="欠拟合/过拟合/L2 对照实验")
    p.add_argument("--predict", type=str, help="对一张图片做数字识别")
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="权重路径")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层大小（需与训练一致）")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--l1", type=float, default=0.0, help="L1 正则系数")
    p.add_argument("--l2", type=float, default=1e-4, help="L2 正则系数")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo_fit:
        demo_under_over_fitting()
        return

    if args.train:
        cfg = TrainConfig(
            hidden_size=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            l1=args.l1,
            l2=args.l2,
        )
        print("配置:", json.dumps(asdict(cfg), ensure_ascii=False))
        model, history = train_mlp(cfg)
        MODEL_DIR.mkdir(exist_ok=True)
        model.save(args.weights)
        meta = {"hidden_size": cfg.hidden_size, "din": 64, "num_classes": 10}
        Path(args.weights).with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"权重已保存: {Path(args.weights).resolve()}")
        plot_history(history)
        print(
            f"最终: train_acc={history['train_acc'][-1]:.3f}  "
            f"val_acc={history['val_acc'][-1]:.3f}"
        )
        return

    if args.predict:
        # 若旁边有 meta json，就用里面的 hidden_size，避免和训练不一致
        meta_path = Path(args.weights).with_suffix(".json")
        hidden = args.hidden
        if meta_path.exists():
            hidden = json.loads(meta_path.read_text(encoding="utf-8")).get("hidden_size", hidden)

        pred, probs = predict_image(args.predict, Path(args.weights), hidden_size=hidden)
        print(f"预测数字: {pred}")
        print("各类概率:")
        for d, p in enumerate(probs):
            bar = "#" * int(p * 40)
            print(f"  {d}: {p:.4f}  {bar}")
        return

    print("用法示例:")
    print("  python digit_recognize.py --train")
    print("  python digit_recognize.py --train --demo-fit")
    print("  python digit_recognize.py --predict your_digit.png")


if __name__ == "__main__":
    main()
