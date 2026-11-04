"""
基础图像处理操作示例（带学习向详细注释）
================================================

这份代码覆盖 OpenCV / NumPy / Pillow 最常用的图像操作。
阅读建议：
1. 先建立概念：数字图像 = 多维数组（矩阵），每个格子是一个像素。
2. OpenCV 默认颜色顺序是 BGR（蓝绿红），不是日常说的 RGB。
3. 几乎所有操作都是：输入数组 -> 变换 -> 输出新数组（尽量不原地改）。

依赖:
  pip install -r requirements.txt

用法:
  python basic_image_ops.py your_image.jpg
  # 结果会写到当前目录下的 output/ 文件夹
"""

# ---------------------------------------------------------------------------
# 标准库
# sys     : 读命令行参数（sys.argv），控制程序退出
# Path    : 比字符串路径更方便的路径对象（拼接、创建目录、取绝对路径）
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 第三方库
# cv2     : OpenCV，计算机视觉主力库（读图、滤波、边缘、形态学……）
# numpy   : 数值计算；图像在内存里就是 numpy.ndarray
# PIL     : Pillow，偏“图像编辑”风格（增强、滤镜），和 OpenCV 互补
# ---------------------------------------------------------------------------
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ===========================================================================
# 1. 读写与基础信息
# ===========================================================================
# 核心记忆：
# - 彩色图 shape = (高 H, 宽 W, 通道 C)，注意先高后宽！
# - 灰度图 shape = (H, W)
# - dtype 常见 uint8，像素取值 0~255
# ===========================================================================

def load_image(path: str) -> np.ndarray:
    """
    读取图像文件，返回 BGR 格式的 numpy 数组。

    参数:
        path: 图片路径，如 "photo.jpg"

    返回:
        img: ndarray，彩色一般为 (H, W, 3)，通道顺序 B,G,R

    注意:
        cv2.imread 失败时不会抛异常，而是返回 None，所以要自己检查。
        路径含中文时，部分 OpenCV 版本可能读失败，可用 np.fromfile + imdecode。
    """
    # flags 默认是 cv2.IMREAD_COLOR：丢弃透明通道，强制 3 通道 BGR
    img = cv2.imread(path)
    if img is None:
        # 路径不存在、格式不支持、权限问题等都会走到这里
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """
    把 ndarray 写成图片文件。

    参数:
        path: 输出路径；扩展名决定编码格式（.jpg / .png …）
        img : uint8 图像数组；彩色需为 BGR

    提示:
        jpg 有损压缩，适合照片；png 无损，适合线稿/掩码。
        返回值 True/False 表示是否写成功，正式项目建议检查返回值。
    """
    cv2.imwrite(path, img)


def image_info(img: np.ndarray) -> dict:
    """
    查看图像基本元信息，调试时非常有用。

    返回字典字段:
        height / width / channels / dtype / shape

    学习点:
        img.ndim == 2 说明是单通道（灰度或二值）
        img.ndim == 3 说明有通道维，常见 channels=3（BGR）或 4（BGRA）
    """
    # shape[:2] 兼容灰度 (H,W) 和彩色 (H,W,C)
    h, w = img.shape[:2]
    # 灰度没有“通道维”，我们约定 channels=1
    channels = 1 if img.ndim == 2 else img.shape[2]
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(img.dtype),  # 例如 'uint8'
        "shape": img.shape,       # 原始 shape，方便对照
    }


# ===========================================================================
# 2. 颜色空间转换
# ===========================================================================
# 为什么要换颜色空间？
# - 灰度：降维，很多算法（阈值、边缘）先要求单通道
# - HSV ：色调/饱和度/亮度分离，做颜色阈值（如抽红色区域）更稳
# - RGB ：给 matplotlib、Pillow、浏览器显示用（它们默认 RGB）
# ===========================================================================

def to_gray(img: np.ndarray) -> np.ndarray:
    """
    BGR 彩色图 -> 灰度图。

    原理（简化）:
        灰度 ≈ 对人眼敏感度加权的通道混合，不是简单平均。
        OpenCV 内部按标准系数把 B,G,R 合成一个亮度值。

    返回:
        (H, W) 的 uint8 数组
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_hsv(img: np.ndarray) -> np.ndarray:
    """
    BGR -> HSV。

    通道含义（OpenCV 量化方式）:
        H (Hue)        色调，0~179（不是 0~360！）
        S (Saturation) 饱和度，0~255
        V (Value)      明度，0~255

    用途示例:
        用 inRange 按 H/S 范围抠出某种颜色。
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """
    BGR -> RGB。

    常见坑:
        用 matplotlib.pyplot.imshow 显示 OpenCV 图前，必须转 RGB，
        否则红蓝通道对调，肤色会发蓝。
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def split_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把彩色图拆成三个单通道图：B, G, R。

    返回:
        (b, g, r)，每个都是 (H, W)

    学习点:
        img[:, :, 0] 也是 B 通道，但 split 语义更清晰。
        合并回去用 cv2.merge([b, g, r])。
    """
    return cv2.split(img)


# ===========================================================================
# 3. 几何变换
# ===========================================================================
# 共同点：改变像素“落在哪里”，不改变“像素值含义”。
# OpenCV 仿射变换统一用 2x3 矩阵 + warpAffine。
# ===========================================================================

def resize(img: np.ndarray, width: int | None = None, height: int | None = None,
           scale: float | None = None) -> np.ndarray:
    """
    缩放图像。

    三种用法（三选一思路）:
        1) scale=0.5           -> 长宽都变成一半
        2) width=320           -> 宽固定，高按比例算
        3) height=240          -> 高固定，宽按比例算
        也可同时给 width 和 height（可能改变长宽比，会拉伸）

    interpolation 插值:
        INTER_AREA   缩小常用，抗锯齿较好（本函数默认）
        INTER_LINEAR 通用默认，放大缩小都行
        INTER_CUBIC  放大更平滑，稍慢
        INTER_NEAREST 最近邻，快但锯齿明显，掩码常用它保标签
    """
    h, w = img.shape[:2]

    # 情况 A：按比例缩放；dsize 传 None，用 fx/fy
    if scale is not None:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if width is None and height is None:
        raise ValueError("请指定 width、height 或 scale")

    # 情况 B：只给一边，另一边按原始长宽比推算，避免变形
    if width is None:
        # 新宽 = 旧宽 * (新高 / 旧高)
        width = int(w * height / h)
    if height is None:
        height = int(h * width / w)

    # dsize 的顺序是 (宽, 高)，和 numpy shape 的 (高, 宽) 相反！
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def rotate(img: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
    """
    绕图像中心旋转。

    参数:
        angle: 角度制；正值 = 逆时针（OpenCV 约定）
        scale: 旋转同时缩放，1.0 表示不缩放

    步骤:
        1) 算旋转中心（通常取图像中心）
        2) getRotationMatrix2D 得到 2x3 仿射矩阵
        3) warpAffine 按矩阵重采样像素

    注意:
        输出画布大小这里仍用原 (w,h)，旋转后边角可能被裁掉。
        若要“完整装下旋转后的图”，需要另行计算新宽高。
    """
    h, w = img.shape[:2]
    # 注意中心是 (x, y) = (宽/2, 高/2)
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # 第三个参数是输出图像大小 (宽, 高)
    return cv2.warpAffine(img, matrix, (w, h))


def flip(img: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """
    翻转图像（镜像）。

    OpenCV flip code:
        1  : 水平翻转（左右镜像，自拍预览常用）
        0  : 垂直翻转（上下颠倒）
       -1  : 同时水平和垂直翻转（等价旋转 180° 的一种实现）
    """
    code = {"horizontal": 1, "vertical": 0, "both": -1}[mode]
    return cv2.flip(img, code)


def crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    裁剪矩形 ROI（Region of Interest，感兴趣区域）。

    参数:
        x, y: 左上角坐标（x 向右，y 向下）
        w, h: 裁剪宽、高

    重要:
        numpy 切片是 img[行, 列] = img[y:y+h, x:x+w]
        .copy() 很关键：不加 copy 时，子数组可能仍共享原图内存，
        改裁剪结果可能“写回”原图。
    """
    return img[y : y + h, x : x + w].copy()


def translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    平移图像。

    仿射矩阵含义:
        [[1, 0, dx],
         [0, 1, dy]]
        新坐标 x' = x + dx
        新坐标 y' = y + dy

    dx>0 向右移，dy>0 向下移。
    移出画布的部分会被裁掉，空出区域默认填 0（黑色）。
    """
    h, w = img.shape[:2]
    # 必须是 float32，warpAffine 要求
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (w, h))


# ===========================================================================
# 4. 亮度 / 对比度 / 阈值
# ===========================================================================
# 亮度对比度：改像素值大小
# 阈值：把灰度图变成黑白二值图（前景/背景），是分割、OCR 预处理常用步骤
# ===========================================================================

def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.0,
                               beta: int = 0) -> np.ndarray:
    """
    线性调整：output = saturate(alpha * input + beta)

    参数:
        alpha: 对比度增益。1.0 不变；>1 对比更强；0~1 更灰
        beta : 亮度偏置。>0 变亮；<0 变暗

    convertScaleAbs:
        先按公式计算，再取绝对值并转到 uint8，同时做饱和裁剪到 0~255，
        避免溢出变成奇怪的回绕值。
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def threshold_binary(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    """
    固定阈值二值化。

    规则（THRESH_BINARY）:
        像素 > thresh  -> 255（白）
        像素 <= thresh -> 0（黑）

    适用:
        光照均匀、目标和背景灰度差明显的图。

    返回:
        二值图 (H, W)，只有 0 和 255
        （函数还返回实际阈值，这里用 _ 忽略）
    """
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Otsu 自动阈值二值化。

    思想:
        在 0~255 里自动找一个阈值，让“类间方差”最大
        （前景、背景分得尽可能开）。

    用法细节:
        thresh 参数传 0 即可，真正阈值由 Otsu 算出来；
        需要同时打开 THRESH_BINARY 和 THRESH_OTSU 两个标志。
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    自适应阈值：每个像素用“局部邻域”算自己的阈值。

    为何需要:
        一张图左边亮右边暗时，全局一个阈值会失效；
        自适应阈值对光照不均更稳（文档扫描、阴影场景）。

    本函数参数含义:
        maxValue=255
        方法=ADAPTIVE_THRESH_GAUSSIAN_C : 邻域加权平均作阈值
        blockSize=11 : 邻域边长，必须是奇数
        C=2 : 从均值里再减一点，调节粗细/敏感度
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


# ===========================================================================
# 5. 滤波与去噪
# ===========================================================================
# 滤波 = 用邻域信息平滑当前像素。
# 代价通常是：噪声少了，细节/边缘也可能糊了。
# 双边滤波试图“去噪但保边”，计算更贵。
# ===========================================================================

def blur_gaussian(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    高斯模糊：邻域按高斯权重加权平均（中间权重大，越远越小）。

    参数:
        ksize: 核边长，必须是正奇数（3/5/7…）
        sigmaX=0 表示让 OpenCV 根据 ksize 自动估算标准差

    用途:
        降噪、Canny 前预处理、降低细节以便检测大结构。
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def blur_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    中值滤波：取邻域像素排序后的中位数。

    特点:
        对椒盐噪声（随机黑白脏点）效果很好；
        比高斯更能保住边缘（但不是专门的保边滤波器）。

    注意:
        ksize 必须是奇数。
    """
    return cv2.medianBlur(img, ksize)


def blur_bilateral(img: np.ndarray) -> np.ndarray:
    """
    双边滤波：空间近 + 颜色近 的像素才参与平均。

    参数（本函数写死的示例值）:
        d=9        邻域直径
        sigmaColor=75  颜色空间滤波 sigma，越大越容易跨颜色平滑
        sigmaSpace=75  坐标空间滤波 sigma，越大越远的点也能影响

    效果直觉:
        平坦区域更干净，物体轮廓仍相对清晰；比高斯慢。
    """
    return cv2.bilateralFilter(img, 9, 75, 75)


# ===========================================================================
# 6. 边缘与形态学
# ===========================================================================
# 边缘：灰度变化剧烈的地方（物体轮廓常在这里）
# 形态学：主要针对二值图，用“结构元素(kernel)”做膨胀/腐蚀等集合运算
# ===========================================================================

def edges_canny(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """
    Canny 边缘检测（经典、实用）。

    流程概要:
        1) 降噪（内部会做高斯）
        2) 算梯度强度与方向
        3) 非极大值抑制（细化边缘）
        4) 双阈值滞后连接：
           - > high      : 强边缘，保留
           - < low       : 丢弃
           - low~high    : 弱边缘，仅当连到强边缘才保留

    调参经验:
        high/low 大约保持 2:1 ~ 3:1；噪声多就整体调高。
    """
    return cv2.Canny(gray, low, high)


def edges_sobel(gray: np.ndarray) -> np.ndarray:
    """
    Sobel 梯度边缘。

    做法:
        gx = 对 x 方向求导（垂直边缘更敏感）
        gy = 对 y 方向求导（水平边缘更敏感）
        幅值 mag = sqrt(gx^2 + gy^2)

    为何用 CV_64F:
        导数有正有负；若直接用 uint8，负值会被截断，信息丢失。
        先在 float64 算完幅值，再 clip 回 0~255 的 uint8 便于显示/保存。
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # dx=1, dy=0
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # dx=0, dy=1
    mag = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(mag, 0, 255))


def morph_erode(binary: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """
    腐蚀：邻域内“全都是前景”才保留前景 -> 前景变瘦、小白噪点易消失。

    kernel:
        结构元素，这里用全 1 的矩形小核。
    iterations:
        重复次数，越大腐蚀越狠。
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(binary, kernel, iterations=iterations)


def morph_dilate(binary: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """
    膨胀：邻域内“有一个前景”就变前景 -> 前景变胖、小黑洞易被填上。

    记忆口诀:
        腐蚀变瘦，膨胀变胖。
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(binary, kernel, iterations=iterations)


def morph_open(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    开运算 = 先腐蚀后膨胀。

    典型用途:
        去掉前景上的小白点/细毛刺，同时大致保持主体面积。
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def morph_close(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    闭运算 = 先膨胀后腐蚀。

    典型用途:
        填前景内部小黑洞、连接近距离断裂。
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# ===========================================================================
# 7. 直方图与均衡化
# ===========================================================================
# 直方图：统计每个灰度值出现了多少次。
# 均衡化：把灰度分布“拉开”，让暗图/灰蒙蒙的图对比更强。
# ===========================================================================

def equalize_hist(gray: np.ndarray) -> np.ndarray:
    """
    全局直方图均衡化。

    效果:
        整体对比度增强；但在本来就很亮/暗的局部可能过曝或放大噪声。
    限制:
        只接受单通道图。彩色图通常对 Y/V/L 通道均衡，再合并回去。
    """
    return cv2.equalizeHist(gray)


def clahe(gray: np.ndarray, clip_limit: float = 2.0,
          tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    CLAHE：限制对比度的自适应直方图均衡。

    相对全局均衡的改进:
        1) 把图划成小块（tile），每块各自均衡 —— 更适应当地光照
        2) clipLimit 限制对比度，减轻噪声被过度放大

    参数:
        clip_limit: 越大增强越猛（也更容易噪）
        tile_grid : 例如 (8,8) 表示划分成 8x8 个小块
    """
    # 先创建算法对象，再 apply 到具体图像（OpenCV 常见两段式 API）
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe_obj.apply(gray)


# ===========================================================================
# 8. 绘图标注
# ===========================================================================
# 检测/跟踪结果可视化时，就是在原图上画框、点、文字。
# 记住：OpenCV 颜色仍是 BGR；(0,255,0) 是绿色，(0,0,255) 是红色。
# ===========================================================================

def draw_shapes(img: np.ndarray) -> np.ndarray:
    """
    在图上画线、矩形、圆、文字的最小示例。

    学习点:
        先 img.copy()，避免把标注画脏原图；
        坐标都是 (x, y)，厚度 thickness=2；
        putText 不支持中文（需要 FreeType 或其他方案）。
    """
    out = img.copy()

    # 线：起点 -> 终点，绿色
    cv2.line(out, (20, 20), (200, 20), (0, 255, 0), 2)

    # 矩形：左上 (40,40) 到右下 (180,140)，蓝色（BGR）
    cv2.rectangle(out, (40, 40), (180, 140), (255, 0, 0), 2)

    # 圆：圆心、半径，红色；thickness=2 画圈，-1 则实心填充
    cv2.circle(out, (110, 90), 30, (0, 0, 255), 2)

    # 文字：内容、左下角位置、字体、字号缩放、颜色、粗细
    cv2.putText(out, "Hello", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


# ===========================================================================
# 9. Pillow 补充：增强与滤镜
# ===========================================================================
# OpenCV 偏“算法/矩阵”；Pillow 偏“修图滤镜”。
# 注意：Pillow 图像对象 Image，和 OpenCV 的 ndarray 不是同一种类型。
# 互转:
#   ndarray(RGB) -> Image.fromarray(rgb)
#   Image -> np.array(img) 后再按需要 RGB<->BGR
# ===========================================================================

def pil_enhance(path: str, brightness: float = 1.2, contrast: float = 1.2,
                sharpness: float = 1.5) -> Image.Image:
    """
    用 Pillow 连续调节亮度、对比度、锐度。

    enhance(factor) 含义:
        1.0 = 原图
        >1  增强
        0~1 减弱
        0   极端情况（如亮度全黑）

    返回:
        PIL.Image 对象（可用 .save() 存盘）
    """
    img = Image.open(path)
    # 每一步都返回新图；链式增强很常见
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def pil_filters(path: str) -> dict[str, Image.Image]:
    """
    套用 Pillow 内置卷积类滤镜，返回名字->图像的字典。

    滤镜直觉:
        BLUR         模糊
        SHARPEN      锐化
        EDGE_ENHANCE 强化边缘
        EMBOSS       浮雕感
        CONTOUR      轮廓感
    """
    img = Image.open(path)
    return {
        "blur": img.filter(ImageFilter.BLUR),
        "sharpen": img.filter(ImageFilter.SHARPEN),
        "edge_enhance": img.filter(ImageFilter.EDGE_ENHANCE),
        "emboss": img.filter(ImageFilter.EMBOSS),
        "contour": img.filter(ImageFilter.CONTOUR),
    }


# ===========================================================================
# Demo：对一张图跑一遍并保存结果
# ===========================================================================
# 这是“总控函数”：把上面积木串起来，方便你一眼看到每种效果。
# 真正做项目时，通常只挑选其中几步组成自己的流水线。
# ===========================================================================

def run_demo(input_path: str, out_dir: str = "output") -> None:
    """
    读取一张图，依次执行多种基础操作，把结果存到 out_dir。

    参数:
        input_path: 输入图片路径
        out_dir   : 输出文件夹（不存在会自动创建）
    """
    # Path 对象支持 / 拼接，跨平台比字符串 + "\\" 更干净
    out = Path(out_dir)
    out.mkdir(exist_ok=True)  # exist_ok=True：已存在也不报错

    # ---------- 读入 & 看信息 ----------
    img = load_image(input_path)
    print("图像信息:", image_info(img))

    # ---------- 颜色 ----------
    gray = to_gray(img)
    save_image(str(out / "01_gray.jpg"), gray)

    # ---------- 几何 ----------
    save_image(str(out / "02_resized.jpg"), resize(img, scale=0.5))
    save_image(str(out / "03_rotated.jpg"), rotate(img, 45))
    save_image(str(out / "04_flipped.jpg"), flip(img, "horizontal"))

    # ---------- 亮度对比度 ----------
    # alpha=1.3 对比加强，beta=30 整体提亮
    save_image(str(out / "05_bright.jpg"), adjust_brightness_contrast(img, alpha=1.3, beta=30))

    # ---------- 滤波 ----------
    save_image(str(out / "06_blur.jpg"), blur_gaussian(img, 9))

    # ---------- 边缘 / 阈值 / 均衡（这些通常基于灰度） ----------
    save_image(str(out / "07_canny.jpg"), edges_canny(gray))
    save_image(str(out / "08_otsu.jpg"), threshold_otsu(gray))
    save_image(str(out / "09_clahe.jpg"), clahe(gray))

    # ---------- 可视化标注 ----------
    save_image(str(out / "10_draw.jpg"), draw_shapes(img))

    # ---------- 裁剪：取大致中心的一半宽高区域 ----------
    h, w = img.shape[:2]
    cw, ch = w // 2, h // 2                 # 裁剪宽高 = 原图一半
    # 左上角放在 (w/4, h/4)，这样裁出来的块大致居中
    save_image(str(out / "11_crop.jpg"), crop(img, w // 4, h // 4, cw, ch))

    # ---------- Pillow 增强 ----------
    enhanced = pil_enhance(input_path)
    # Pillow 用 save；OpenCV 用 imwrite。别混。
    enhanced.save(out / "12_pil_enhance.jpg")

    # resolve() 打出绝对路径，方便你去文件夹里找结果
    print(f"结果已保存到: {out.resolve()}")


# ===========================================================================
# 脚本入口
# ===========================================================================
# 只有“直接运行本文件”时才会进这里；
# 若被其他文件 import，只会加载函数，不会自动跑 demo。
# ===========================================================================

if __name__ == "__main__":
    # sys.argv[0] 是脚本名，sys.argv[1] 才是你传的图片路径
    if len(sys.argv) < 2:
        print("用法: python basic_image_ops.py <图片路径>")
        print("示例: python basic_image_ops.py photo.jpg")
        sys.exit(1)  # 非 0 表示异常退出，方便脚本/CI 判断失败
    run_demo(sys.argv[1])
