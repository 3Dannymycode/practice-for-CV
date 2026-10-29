"""
基础图像处理操作示例
依赖: pip install -r requirements.txt
用法: python basic_image_ops.py your_image.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# 1. 读写与基础信息
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """读取图像（BGR，OpenCV 默认）"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """保存图像"""
    cv2.imwrite(path, img)


def image_info(img: np.ndarray) -> dict:
    """查看图像基本信息"""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(img.dtype),
        "shape": img.shape,
    }


# ---------------------------------------------------------------------------
# 2. 颜色空间转换
# ---------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR -> 灰度"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_hsv(img: np.ndarray) -> np.ndarray:
    """BGR -> HSV"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR -> RGB（给 matplotlib / PIL 用）"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def split_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """拆分 B、G、R 通道"""
    return cv2.split(img)


# ---------------------------------------------------------------------------
# 3. 几何变换
# ---------------------------------------------------------------------------

def resize(img: np.ndarray, width: int | None = None, height: int | None = None,
           scale: float | None = None) -> np.ndarray:
    """缩放：指定宽/高，或按比例 scale"""
    h, w = img.shape[:2]
    if scale is not None:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if width is None and height is None:
        raise ValueError("请指定 width、height 或 scale")
    if width is None:
        width = int(w * height / h)
    if height is None:
        height = int(h * width / w)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def rotate(img: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
    """绕中心旋转（角度为度，逆时针为正）"""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, matrix, (w, h))


def flip(img: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """翻转: horizontal / vertical / both"""
    code = {"horizontal": 1, "vertical": 0, "both": -1}[mode]
    return cv2.flip(img, code)


def crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """裁剪矩形区域 (x, y, 宽, 高)"""
    return img[y : y + h, x : x + w].copy()


def translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """平移"""
    h, w = img.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (w, h))


# ---------------------------------------------------------------------------
# 4. 亮度 / 对比度 / 阈值
# ---------------------------------------------------------------------------

def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.0,
                               beta: int = 0) -> np.ndarray:
    """对比度 alpha（1.0 不变），亮度 beta（0 不变）"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def threshold_binary(gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    """固定阈值二值化"""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu 自动阈值"""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """自适应阈值（适合光照不均）"""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


# ---------------------------------------------------------------------------
# 5. 滤波与去噪
# ---------------------------------------------------------------------------

def blur_gaussian(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """高斯模糊"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def blur_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """中值滤波（去椒盐噪声较好）"""
    return cv2.medianBlur(img, ksize)


def blur_bilateral(img: np.ndarray) -> np.ndarray:
    """双边滤波（保边去噪）"""
    return cv2.bilateralFilter(img, 9, 75, 75)


# ---------------------------------------------------------------------------
# 6. 边缘与形态学
# ---------------------------------------------------------------------------

def edges_canny(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny 边缘检测"""
    return cv2.Canny(gray, low, high)


def edges_sobel(gray: np.ndarray) -> np.ndarray:
    """Sobel 边缘（X+Y 幅值）"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(mag, 0, 255))


def morph_erode(binary: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """腐蚀"""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(binary, kernel, iterations=iterations)


def morph_dilate(binary: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """膨胀"""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(binary, kernel, iterations=iterations)


def morph_open(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """开运算：先腐蚀后膨胀（去小白点）"""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def morph_close(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """闭运算：先膨胀后腐蚀（填小黑洞）"""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# ---------------------------------------------------------------------------
# 7. 直方图与均衡化
# ---------------------------------------------------------------------------

def equalize_hist(gray: np.ndarray) -> np.ndarray:
    """直方图均衡化（增强对比度）"""
    return cv2.equalizeHist(gray)


def clahe(gray: np.ndarray, clip_limit: float = 2.0,
          tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """自适应直方图均衡（CLAHE，通常比全局均衡更自然）"""
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe_obj.apply(gray)


# ---------------------------------------------------------------------------
# 8. 绘图标注
# ---------------------------------------------------------------------------

def draw_shapes(img: np.ndarray) -> np.ndarray:
    """在图上画线、矩形、圆、文字（示例）"""
    out = img.copy()
    cv2.line(out, (20, 20), (200, 20), (0, 255, 0), 2)
    cv2.rectangle(out, (40, 40), (180, 140), (255, 0, 0), 2)
    cv2.circle(out, (110, 90), 30, (0, 0, 255), 2)
    cv2.putText(out, "Hello", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
# 9. Pillow 补充：增强与滤镜
# ---------------------------------------------------------------------------

def pil_enhance(path: str, brightness: float = 1.2, contrast: float = 1.2,
                sharpness: float = 1.5) -> Image.Image:
    """用 Pillow 调节亮度 / 对比度 / 锐度"""
    img = Image.open(path)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def pil_filters(path: str) -> dict[str, Image.Image]:
    """常见 Pillow 滤镜"""
    img = Image.open(path)
    return {
        "blur": img.filter(ImageFilter.BLUR),
        "sharpen": img.filter(ImageFilter.SHARPEN),
        "edge_enhance": img.filter(ImageFilter.EDGE_ENHANCE),
        "emboss": img.filter(ImageFilter.EMBOSS),
        "contour": img.filter(ImageFilter.CONTOUR),
    }


# ---------------------------------------------------------------------------
# Demo：对一张图跑一遍并保存结果
# ---------------------------------------------------------------------------

def run_demo(input_path: str, out_dir: str = "output") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    img = load_image(input_path)
    print("图像信息:", image_info(img))

    gray = to_gray(img)
    save_image(str(out / "01_gray.jpg"), gray)
    save_image(str(out / "02_resized.jpg"), resize(img, scale=0.5))
    save_image(str(out / "03_rotated.jpg"), rotate(img, 45))
    save_image(str(out / "04_flipped.jpg"), flip(img, "horizontal"))
    save_image(str(out / "05_bright.jpg"), adjust_brightness_contrast(img, alpha=1.3, beta=30))
    save_image(str(out / "06_blur.jpg"), blur_gaussian(img, 9))
    save_image(str(out / "07_canny.jpg"), edges_canny(gray))
    save_image(str(out / "08_otsu.jpg"), threshold_otsu(gray))
    save_image(str(out / "09_clahe.jpg"), clahe(gray))
    save_image(str(out / "10_draw.jpg"), draw_shapes(img))

    # 裁剪中心区域（示例）
    h, w = img.shape[:2]
    cw, ch = w // 2, h // 2
    save_image(str(out / "11_crop.jpg"), crop(img, w // 4, h // 4, cw, ch))

    # Pillow
    enhanced = pil_enhance(input_path)
    enhanced.save(out / "12_pil_enhance.jpg")

    print(f"结果已保存到: {out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python basic_image_ops.py <图片路径>")
        print("示例: python basic_image_ops.py photo.jpg")
        sys.exit(1)
    run_demo(sys.argv[1])
