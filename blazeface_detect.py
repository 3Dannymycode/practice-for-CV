"""
BlazeFace 人脸检测（MediaPipe Tasks API）

依赖:
  pip install mediapipe opencv-python numpy

首次运行会自动下载短距/全距 BlazeFace 模型到 models/

用法:
  python blazeface_detect.py photo.jpg
  python blazeface_detect.py photo.jpg --show
  python blazeface_detect.py --camera
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

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

KEYPOINT_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "right_ear",
    "left_ear",
)


@dataclass
class FaceResult:
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    keypoints: dict[str, tuple[int, int]]


def ensure_model(model_selection: int = 0) -> Path:
    name, url = MODEL_URLS[model_selection]
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / name
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"下载 BlazeFace 模型: {name}")
    urllib.request.urlretrieve(url, path)
    print(f"模型已保存: {path}")
    return path


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def create_detector(
    model_selection: int = 0,
    min_confidence: float = 0.5,
) -> vision.FaceDetector:
    model_path = ensure_model(model_selection)
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=min_confidence,
    )
    return vision.FaceDetector.create_from_options(options)


def detect_faces(bgr: np.ndarray, detector: vision.FaceDetector) -> list[FaceResult]:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    faces: list[FaceResult] = []
    if not result.detections:
        return faces

    for det in result.detections:
        conf = float(det.categories[0].score) if det.categories else 0.0
        box = det.bounding_box  # 已是像素坐标
        x1, y1 = int(box.origin_x), int(box.origin_y)
        x2, y2 = int(box.origin_x + box.width), int(box.origin_y + box.height)
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)

        kps: dict[str, tuple[int, int]] = {}
        for i, kp in enumerate(det.keypoints or []):
            name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kp_{i}"
            # Tasks API 关键点是归一化坐标
            kps[name] = (int(kp.x * w), int(kp.y * h))

        faces.append(FaceResult(conf=conf, x1=x1, y1=y1, x2=x2, y2=y2, keypoints=kps))
    return faces


def draw_faces(bgr: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
    vis = bgr.copy()
    for i, f in enumerate(faces):
        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 220, 0), 2)
        cv2.putText(
            vis,
            f"face{i} {f.conf:.2f}",
            (f.x1, max(0, f.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        for pt in f.keypoints.values():
            cv2.circle(vis, pt, 2, (0, 0, 255), -1, cv2.LINE_AA)
    return vis


def run_image(
    path: str,
    out_dir: str = "output",
    show: bool = False,
    model_selection: int = 0,
    min_confidence: float = 0.5,
) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    detector = create_detector(model_selection, min_confidence)
    try:
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
    save_path = out / f"blazeface_{Path(path).stem}.jpg"
    cv2.imwrite(str(save_path), vis)
    print(f"结果已保存: {save_path.resolve()}")

    if show:
        cv2.imshow("BlazeFace", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _open_camera(cam_id: int) -> cv2.VideoCapture:
    """打开摄像头；Windows 优先 DirectShow，并预热几帧避免黑屏。"""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {cam_id}")

    # 不强制超高分辨率，部分笔记本相机会因此黑屏
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 预热：刚打开时常是黑帧，等自动曝光起来
    last = None
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            last = frame
            if float(frame.mean()) > 5.0:
                break
    if last is None:
        raise RuntimeError("摄像头打开了但读不到画面，请检查是否被其他程序占用")
    if float(last.mean()) < 2.0:
        print(
            "警告：画面几乎全黑。请关闭占用摄像头的程序（PyCharm/浏览器/会议软件），"
            "并在 Windows 设置 → 隐私和安全性 → 相机 中允许桌面应用访问。",
            flush=True,
        )
    return cap


def run_camera(
    cam_id: int = 0,
    model_selection: int = 0,
    min_confidence: float = 0.5,
) -> None:
    print("正在打开摄像头...", flush=True)
    cap = _open_camera(cam_id)

    print("正在加载 BlazeFace 模型...", flush=True)
    detector = create_detector(model_selection, min_confidence)
    print("摄像头已启动：看向镜头即可实时框选人脸。按 q 退出。", flush=True)

    win = "BlazeFace (camera)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    def _bring_to_front() -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes

            hwnd = ctypes.windll.user32.FindWindowW(None, win)
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
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
            frame = cv2.flip(frame, 1)  # 镜像，方便对着自己调
            faces = detect_faces(frame, detector)
            vis = draw_faces(frame, faces)
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
            if frame_i == 0:
                _bring_to_front()
                print("预览窗口已打开，请看任务栏里的 BlazeFace (camera)", flush=True)
            frame_i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    if args.camera is not None:
        run_camera(args.camera, args.model, args.conf)
        return
    if not args.image:
        print("用法: python blazeface_detect.py <图片> [--show]")
        print("      python blazeface_detect.py --camera")
        sys.exit(1)
    run_image(args.image, args.out, args.show, args.model, args.conf)


if __name__ == "__main__":
    main()
