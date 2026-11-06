"""
BlazeFace 人脸检测（MediaPipe Tasks API，带学习向详细注释）
============================================================

这份代码做一件事：用人脸检测模型在图片 / 摄像头画面里找到“人脸在哪”。
阅读建议：
1. 先分清三层：读图 → 模型推理 → 画框可视化。
2. BlazeFace 是轻量级人脸检测器，适合实时；MediaPipe 负责加载模型并封装 API。
3. OpenCV 图像默认是 BGR；MediaPipe 要 RGB。转换这一步漏了，检测会乱。

依赖:
  pip install mediapipe opencv-python numpy
  # 或: pip install -r requirements.txt

首次运行若本地没有模型文件，会自动下载到 models/

用法:
  python blazeface_detect.py photo.jpg
  python blazeface_detect.py photo.jpg --show
  python blazeface_detect.py --camera
  python blazeface_detect.py --camera 1 --model 0 --conf 0.6
"""

# ---------------------------------------------------------------------------
# from __future__ import annotations
#   让类型注解以“字符串形式”推迟求值。好处：
#   - 可以写 list[FaceResult] 这类较新语法，且类可以前向引用自己
#   - 和 Python 3.9+ / 3.10+ 注解风格兼容更好
# ---------------------------------------------------------------------------
from __future__ import annotations

# ---------------------------------------------------------------------------
# 标准库
# argparse         : 解析命令行参数（比手写 sys.argv 更清晰）
# sys              : 平台判断（Windows）、异常退出码
# urllib.request   : 下载模型文件（首次运行）
# dataclass        : 用很少样板代码定义“纯数据”结果对象
# Path             : 路径拼接、建目录、取绝对路径
# ---------------------------------------------------------------------------
import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 第三方库
# cv2       : OpenCV，读图 / 摄像头 / 画框 / 显示窗口
# mediapipe : Google 的端侧 CV 工具包；这里用 Tasks API 做人脸检测
# numpy     : 图像本质是 ndarray；算均值判断黑帧等也会用到
# ---------------------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


# ===========================================================================
# 路径与模型配置
# ===========================================================================
# ROOT      : 本脚本所在目录（无论你从哪 cd 进来运行，模型都找这里）
# MODEL_DIR : 存放 .tflite 权重的文件夹
#
# BlazeFace 有两个常用变体（model_selection）:
#   0 short_range : 近距离（大约 2 米内），自拍 / 笔记本摄像头最常用
#   1 full_range  : 更远一点（大约 5 米内），房间监控类场景更合适
# 选错变体不会直接报错，但远脸 / 近脸漏检会明显变多。
# ===========================================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

# key = model_selection；value = (本地文件名, 官方下载 URL)
MODEL_URLS = {
    0: (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ),
    1: (
        "blaze_face_full_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
    ),
}

# MediaPipe Face Detector 默认返回 6 个关键点，顺序是约定好的。
# 注意：这里的 left/right 是“图像坐标系里的左右”，对自拍镜像后要自己心里换算。
KEYPOINT_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "right_ear",
    "left_ear",
)


# ===========================================================================
# 检测结果数据结构
# ===========================================================================
# 为什么不用“裸字典 / 元组”？
#   dataclass 字段有名字、有类型，读代码和 print 调试都更省事。
# 坐标约定：
#   (x1, y1) 左上角；(x2, y2) 右下角；单位是像素，原点在图像左上。
# ===========================================================================

@dataclass
class FaceResult:
    """单张脸的检测结果。"""

    conf: float  # 置信度，通常 0~1；越大越“像脸”
    x1: int      # 框左上 x
    y1: int      # 框左上 y
    x2: int      # 框右下 x
    y2: int      # 框右下 y
    # 关键点名 -> (x, y) 像素坐标；可能少于 6 个（极少数情况）
    keypoints: dict[str, tuple[int, int]]


# ===========================================================================
# 模型准备
# ===========================================================================

def ensure_model(model_selection: int = 0) -> Path:
    """
    确保本地有对应 BlazeFace 权重；没有就下载。

    参数:
        model_selection: 0=短距，1=全距

    返回:
        本地 .tflite 文件的 Path

    学习点:
        推理本身不需要每次联网；只有“第一次缺文件”才下载。
        把模型放进仓库 / 手动拷贝到 models/ 也可以，函数会直接复用。
    """
    name, url = MODEL_URLS[model_selection]
    MODEL_DIR.mkdir(exist_ok=True)  # 没有 models/ 就创建
    path = MODEL_DIR / name

    # st_size > 0：避免上次下载中断留下 0 字节空文件还当“已有模型”
    if path.exists() and path.stat().st_size > 0:
        return path

    print(f"下载 BlazeFace 模型: {name}")
    # urlretrieve：同步下载到本地路径；网络不通时会在这里报错
    urllib.request.urlretrieve(url, path)
    print(f"模型已保存: {path}")
    return path


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    """
    把框坐标裁剪进图像范围内。

    为什么需要？
        模型偶发给出贴边 / 略越界的框；直接拿去切片或画图可能越界。
        w-1 / h-1 是合法像素下标的最大值（0-based）。
    """
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def create_detector(
    model_selection: int = 0,
    min_confidence: float = 0.5,
) -> vision.FaceDetector:
    """
    创建 MediaPipe FaceDetector。

    参数:
        model_selection : 0 / 1，对应短距 / 全距模型
        min_confidence  : 低于该分数的候选脸会被丢掉（默认 0.5）

    返回:
        FaceDetector 对象；用完务必 .close() 释放资源

    Tasks API 分层直觉:
        BaseOptions      -> 模型文件路径、加速器等“底座”配置
        FaceDetectorOptions -> 检测阈值等任务相关配置
        create_from_options -> 真正构建可调用的检测器
    """
    model_path = ensure_model(model_selection)
    options = vision.FaceDetectorOptions(
        # model_asset_path 必须是字符串路径；Path 要先 str(...)
        base_options=BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=min_confidence,
    )
    return vision.FaceDetector.create_from_options(options)


# ===========================================================================
# 核心推理：BGR 图 -> 人脸列表
# ===========================================================================

def detect_faces(bgr: np.ndarray, detector: vision.FaceDetector) -> list[FaceResult]:
    """
    对一张 BGR 图像做人脸检测。

    参数:
        bgr      : OpenCV 读进来的图，shape=(H, W, 3)，通道顺序 B,G,R
        detector : create_detector(...) 得到的检测器

    返回:
        FaceResult 列表；无人脸时返回空列表（不是 None）

    流水线:
        1) BGR -> RGB（MediaPipe 吃 SRGB）
        2) 包成 mp.Image
        3) detector.detect(...)
        4) 把官方结果转成我们自己的 FaceResult（像素坐标）
    """
    h, w = bgr.shape[:2]

    # 关键坑：忘记转 RGB，红蓝通道反了，肤色特征错位，检测变差
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe 不直接吃 numpy；要用它的 Image 容器声明“这是 SRGB”
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    faces: list[FaceResult] = []
    if not result.detections:
        return faces

    for det in result.detections:
        # categories[0].score：该检测作为“人脸”类的置信度
        conf = float(det.categories[0].score) if det.categories else 0.0

        # Tasks Face Detector 的 bounding_box 已经是像素坐标（不是 0~1）
        # origin_x/y = 左上角；width/height = 宽高
        box = det.bounding_box
        x1, y1 = int(box.origin_x), int(box.origin_y)
        x2, y2 = int(box.origin_x + box.width), int(box.origin_y + box.height)
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)

        # 关键点：与框不同，keypoints 是归一化坐标（相对宽/高的 0~1）
        # 所以要乘 w / h 才变成像素
        kps: dict[str, tuple[int, int]] = {}
        for i, kp in enumerate(det.keypoints or []):
            name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kp_{i}"
            kps[name] = (int(kp.x * w), int(kp.y * h))

        faces.append(FaceResult(conf=conf, x1=x1, y1=y1, x2=x2, y2=y2, keypoints=kps))
    return faces


# ===========================================================================
# 可视化
# ===========================================================================

def draw_faces(bgr: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
    """
    在图上画人脸框、置信度文字、关键点。

    注意:
        先 bgr.copy()，避免把标注直接画脏原图（后面还可能要另存原图）。
        OpenCV 颜色是 BGR：(0,220,0) 偏绿，(0,0,255) 是红色。
    """
    vis = bgr.copy()
    for i, f in enumerate(faces):
        # 矩形：左上 -> 右下，厚度 2
        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 220, 0), 2)

        # 文字画在框上方；y-8 可能小于 0，用 max 防止出界
        cv2.putText(
            vis,
            f"face{i} {f.conf:.2f}",
            (f.x1, max(0, f.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,          # 字号缩放
            (0, 220, 0),   # 绿色
            2,             # 粗细
            cv2.LINE_AA,   # 抗锯齿，字更顺滑
        )

        # 关键点画成实心小圆（thickness=-1 表示填充）
        for pt in f.keypoints.values():
            cv2.circle(vis, pt, 2, (0, 0, 255), -1, cv2.LINE_AA)
    return vis


# ===========================================================================
# 图片模式
# ===========================================================================

def run_image(
    path: str,
    out_dir: str = "output",
    show: bool = False,
    model_selection: int = 0,
    min_confidence: float = 0.5,
) -> None:
    """
    对单张图片做人脸检测，保存标注图，可选弹窗显示。

    参数:
        path            : 输入图片路径
        out_dir         : 输出目录（默认 output/）
        show            : True 时用 cv2.imshow 弹窗，按任意键关闭
        model_selection : 0 短距 / 1 全距
        min_confidence  : 最小置信度阈值
    """
    # cv2.imread 失败返回 None，不会抛异常 —— 必须自己检查
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    detector = create_detector(model_selection, min_confidence)
    try:
        # try/finally：哪怕检测中途异常，也要关掉检测器，避免句柄泄漏
        faces = detect_faces(img, detector)
    finally:
        detector.close()

    print(f"检测到 {len(faces)} 张人脸")
    for i, f in enumerate(faces):
        print(
            f"  [{i}] conf={f.conf:.3f} box=({f.x1},{f.y1})-({f.x2},{f.y2}) "
            f"kps={f.keypoints}"
        )

    vis = draw_faces(img, faces)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    # stem = 去掉扩展名的文件名；例如 face.png -> blazeface_face.jpg
    save_path = out / f"blazeface_{Path(path).stem}.jpg"
    cv2.imwrite(str(save_path), vis)
    print(f"结果已保存: {save_path.resolve()}")

    if show:
        cv2.imshow("BlazeFace", vis)
        cv2.waitKey(0)          # 0 = 无限等待按键
        cv2.destroyAllWindows()


# ===========================================================================
# 摄像头打开与预热
# ===========================================================================

def _open_camera(cam_id: int) -> cv2.VideoCapture:
    """
    打开摄像头；Windows 优先 DirectShow，并预热几帧避免黑屏。

    参数:
        cam_id: 摄像头编号，笔记本内置一般是 0；外接可能是 1

    为什么 Windows 要用 CAP_DSHOW？
        默认后端有时打开慢、偶发黑屏；DirectShow 通常更稳。

    为什么要预热？
        刚打开时自动曝光 / 白平衡还没起来，前几帧经常全黑或极暗。
        用帧均值粗判亮度：mean 很小 ≈ 画面几乎全黑。
    """
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {cam_id}")

    # 不强制超高分辨率：部分笔记本相机会因设定失败而黑屏
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 最多读 30 帧做预热；亮度够了就提前停
    last = None
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            last = frame
            # frame.mean()：所有像素通道的平均亮度
            if float(frame.mean()) > 5.0:
                break

    if last is None:
        raise RuntimeError("摄像头打开了但读不到画面，请检查是否被其他程序占用")

    if float(last.mean()) < 2.0:
        # 常见原因：权限没开、被会议软件占用、镜头盖着、隐私设置拦截
        print(
            "警告：画面几乎全黑。请关闭占用摄像头的程序（PyCharm/浏览器/会议软件），"
            "并在 Windows 设置 → 隐私和安全性 → 相机 中允许桌面应用访问。",
            flush=True,
        )
    return cap


# ===========================================================================
# 摄像头实时模式
# ===========================================================================

def run_camera(
    cam_id: int = 0,
    model_selection: int = 0,
    min_confidence: float = 0.5,
) -> None:
    """
    打开摄像头，逐帧做人脸检测并实时显示。

    退出方式: 焦点在预览窗口时按 q

    主循环结构（几乎所有实时 CV demo 都长这样）:
        while True:
            读一帧 ->（可选预处理）-> 推理 -> 画结果 -> imshow -> 检查按键
    """
    print("正在打开摄像头...", flush=True)
    cap = _open_camera(cam_id)

    print("正在加载 BlazeFace 模型...", flush=True)
    detector = create_detector(model_selection, min_confidence)
    print("摄像头已启动：看向镜头即可实时框选人脸。按 q 退出。", flush=True)

    win = "BlazeFace (camera)"
    # WINDOW_NORMAL：允许用户拖拽改窗口大小（WINDOW_AUTOSIZE 则锁死）
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    def _bring_to_front() -> None:
        """
        Windows 下把预览窗口拉到前台。

        背景:
            有些环境窗口创建后藏在任务栏后面，用户以为“没打开”。
            用 Win32 API FindWindow + SetForegroundWindow 尝试置顶一次。
            失败就静默忽略 —— 不影响检测主流程。
        """
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes

            # 按窗口标题查找；必须和 namedWindow 的名字一致
            hwnd = ctypes.windll.user32.FindWindowW(None, win)
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # 9 = SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    try:
        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取摄像头帧失败", flush=True)
                break

            # flip code=1：水平镜像。自拍时“你动左边，画面左边动”，更符合照镜子直觉
            frame = cv2.flip(frame, 1)

            faces = detect_faces(frame, detector)
            vis = draw_faces(frame, faces)

            # 左上角 HUD：当前人脸数 + 退出提示
            cv2.putText(
                vis,
                f"faces: {len(faces)}  |  press q to quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, vis)

            # 第一帧显示后再抢焦点，避免窗口还没创建就 FindWindow 失败
            if frame_i == 0:
                _bring_to_front()
                print("预览窗口已打开，请看任务栏里的 BlazeFace (camera)", flush=True)
            frame_i += 1

            # waitKey(1)：等 1ms 并处理窗口事件；没有它窗口可能假死
            # & 0xFF：兼容某些系统返回值高位带杂讯的情况
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 无论正常按 q 还是中途异常，都释放摄像头和窗口
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


# ===========================================================================
# 命令行入口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    定义并解析命令行参数。

    设计点:
        image 用 nargs='?'：可省略，方便只写 --camera
        --camera 用 nargs='?' + const=0：
            --camera     -> 使用摄像头 0
            --camera 1   -> 使用摄像头 1
            不写 --camera -> args.camera is None
    """
    p = argparse.ArgumentParser(description="BlazeFace 人脸检测 (MediaPipe Tasks)")
    p.add_argument("image", nargs="?", help="输入图片路径")
    p.add_argument("--camera", nargs="?", const=0, type=int, help="打开摄像头，可选编号")
    p.add_argument("--show", action="store_true", help="弹窗显示检测结果")
    p.add_argument(
        "--model",
        type=int,
        choices=(0, 1),
        default=0,
        help="0=短距BlazeFace(~2m), 1=全距(~5m)",
    )
    p.add_argument("--conf", type=float, default=0.5, help="最小置信度")
    p.add_argument("--out", default="output", help="输出目录")
    return p.parse_args()


def main() -> None:
    """
    程序总入口：根据参数走摄像头模式或图片模式。
    """
    args = parse_args()

    # 摄像头模式优先：即使同时给了图片路径，有 --camera 也先开摄像头
    if args.camera is not None:
        run_camera(args.camera, args.model, args.conf)
        return

    if not args.image:
        print("用法: python blazeface_detect.py <图片> [--show]")
        print("      python blazeface_detect.py --camera")
        sys.exit(1)  # 非 0 退出码表示用法错误 / 失败

    run_image(args.image, args.out, args.show, args.model, args.conf)


# ===========================================================================
# 脚本入口
# ===========================================================================
# 直接运行本文件时执行 main()；
# 被其他文件 import 时只加载函数，不会自动打开摄像头 / 读图。
# ===========================================================================

if __name__ == "__main__":
    main()
