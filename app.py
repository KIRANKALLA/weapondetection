import os
import time
import glob
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Weapon Detection using advanced deep learning algorithms", layout="wide")

# =========================
# Sidebar controls (single creation; no widgets in loops)
# =========================
st.sidebar.header("Model & Source")

model_path = st.sidebar.text_input(
    "Model path (.pt)",
    value=r"wd.pt",
    help="Absolute or relative path to your trained algorithm weights.",
    key="model_path",
)

use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=False, help="Requires CUDA-enabled PyTorch", key="use_gpu")

source_mode = st.sidebar.radio(
    "Choose source",
    options=[
        "Upload image(s)",
        "Local image path",
        "Upload a video",
        "Local video path",
        "Webcam",
    ],
    index=0,
    key="source_mode",
)

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.01, key="conf")
iou = st.sidebar.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.01, key="iou")
imgsz = st.sidebar.selectbox("Inference size (imgsz)", [320, 416, 512, 640, 960], index=3, key="imgsz")

# Skip-frames option (1 = no skip)
skip_n = st.sidebar.number_input(
    "Process every Nth frame (video/webcam)", min_value=1, max_value=10, value=2, step=1, key="skip_n"
)

# Inputs (declared once)
uploaded_images: List = []
uploaded_video = None
local_image_path = ""
local_video_path = ""
cam_index = 0

if source_mode == "Upload image(s)":
    uploaded_images = st.sidebar.file_uploader(
        "Upload image(s)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="uploader_images",
    )
elif source_mode == "Local image path":
    local_image_path = st.sidebar.text_input(
        "Image file OR folder path (reads *.jpg, *.jpeg, *.png, *.bmp, *.webp)",
        value=r"d:/datasets/weapons/sample.jpg",
        key="local_image_path",
    )
elif source_mode == "Upload a video":
    uploaded_video = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="uploader_video"
    )
elif source_mode == "Local video path":
    local_video_path = st.sidebar.text_input(
        "Video file path",
        value=r"e:/gun 2 video.mp4",
        help="Use a full path. For spaces, prefer raw string like r'e:/gun 2 video.mp4'.",
        key="local_video_path",
    )
else:
    cam_index = st.sidebar.number_input("Webcam index", min_value=0, value=0, step=1, key="cam_index")

start_clicked = st.sidebar.button("▶ Start", key="btn_start")

# =========================
# Utilities
# =========================
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str, want_gpu: bool):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    m = YOLO(weights_path)
    if want_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                m.to("cuda")
            else:
                st.warning("CUDA not available; running on CPU.")
        except Exception as e:
            st.warning(f"Could not move model to GPU: {e}")
    return m

def read_image_from_upload(upload) -> np.ndarray:
    """Read an uploaded image file_uploader object into a BGR numpy array."""
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR

def collect_local_images(path_str: str) -> List[str]:
    """Return list of image paths from a file or a directory."""
    if not path_str:
        return []
    if os.path.isdir(path_str):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path_str, ext)))
        return sorted(files)
    if os.path.isfile(path_str):
        return [path_str]
    return []

def infer_and_annotate_images(
    model: YOLO, images_bgr: List[Tuple[str, np.ndarray]], conf: float, iou: float, imgsz: int
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Run inference on list of (name, BGR image) and return (name, RGB annotated, summary dict).
    """
    out = []
    for name, bgr in images_bgr:
        res = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        annotated_bgr = res.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        counts = {}
        if res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            for cid in cls_ids:
                counts[cid] = counts.get(cid, 0) + 1

        out.append((name, annotated_rgb, {"detections": counts, "shape": annotated_rgb.shape}))
    return out

def open_video_capture(mode, uploaded_file, local_path_str, cam_idx):
    """
    Return (cv2.VideoCapture, cleanup_callback or None, opened_path_str or None).
    """
    cleanup = None
    opened_path = None

    if mode == "Upload a video":
        if not uploaded_file:
            st.warning("Please upload a video to start.")
            return None, None, None
        suffix = os.path.splitext(uploaded_file.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()
        opened_path = tfile.name
        cap = cv2.VideoCapture(opened_path)

        def _cleanup():
            try:
                os.unlink(opened_path)
            except Exception:
                pass

        cleanup = _cleanup

    elif mode == "Local video path":
        if not local_path_str or not os.path.exists(local_path_str):
            st.error("Invalid or missing local video path.")
            return None, None, None
        opened_path = local_path_str
        cap = cv2.VideoCapture(opened_path)

    else:  # Webcam
        cap = cv2.VideoCapture(int(cam_idx))
        opened_path = f"webcam:{cam_idx}"

    if not cap or not cap.isOpened():
        st.error("Failed to open video source. Check the path/index and permissions.")
        if cleanup:
            cleanup()
        return None, None, None

    return cap, cleanup, opened_path

# =========================
# Main UI
# =========================
st.title(" Weapon Detection and Tracking ")

with st.expander("Notes & Tips", expanded=False):
    st.markdown(
        """
- Renders with `st.image()` (no `cv2.imshow()`).
- Linux deps if needed: `sudo apt-get update && sudo apt-get install -y libgl1 ffmpeg`
- Lower `imgsz` (e.g., 320) and increase **Process every Nth frame** for more FPS.
- Enable **Use GPU** if your PyTorch is CUDA-enabled.
        """
    )

frame_area = st.empty()
stats_col1, stats_col2, stats_col3 = st.columns(3)

# =========================
# Run
# =========================
if start_clicked:
    try:
        model = load_model(st.session_state.model_path, st.session_state.use_gpu)
    except Exception as e:
        st.exception(e)
        st.stop()

    # ---------- IMAGE MODES ----------
    if source_mode in ("Upload image(s)", "Local image path"):
        images_to_process: List[Tuple[str, np.ndarray]] = []

        if source_mode == "Upload image(s)":
            if not uploaded_images:
                st.warning("Please upload one or more images.")
                st.stop()
            for up in uploaded_images:
                bgr = read_image_from_upload(up)
                if bgr is None:
                    st.warning(f"Could not read {up.name}")
                    continue
                images_to_process.append((up.name, bgr))
        else:  # Local image path
            paths = collect_local_images(local_image_path)
            if not paths:
                st.error("No images found at the provided path.")
                st.stop()
            for p in paths:
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    st.warning(f"Could not read: {p}")
                    continue
                images_to_process.append((os.path.basename(p), bgr))

        # Inference on images
        results = infer_and_annotate_images(
            model, images_to_process, st.session_state.conf, st.session_state.iou, st.session_state.imgsz
        )

        # Display results (grid)
        n = len(results)
        cols = st.columns(3) if n >= 3 else st.columns(max(1, n))
        for idx, (name, annotated_rgb, summary) in enumerate(results):
            with cols[idx % len(cols)]:
                st.image(annotated_rgb, caption=f"{name} | detections: {summary['detections']}", use_container_width=True)

        st.success(f"Processed {len(results)} image(s).")

    # ---------- VIDEO / WEBCAM MODES ----------
    else:
        cap, cleanup_cb, opened_path = open_video_capture(
            source_mode, uploaded_video, local_video_path, st.session_state.get("cam_index", 0)
        )
        if cap is None:
            st.stop()

        st.success(f"Opened source: {opened_path}")

        # FPS (for info only; we don't throttle)
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        if not fps_src or fps_src <= 0 or fps_src > 120:
            fps_src = 30.0

        frames = 0
        frame_idx = 0
        last_annotated = None
        t0 = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    st.info("End of stream or cannot read frame.")
                    break

                # Skip-frame logic: run YOLO only every Nth frame or if no previous result
                if frame_idx % st.session_state.skip_n == 0 or last_annotated is None:
                    results = model.predict(
                        frame,
                        conf=st.session_state.conf,
                        iou=st.session_state.iou,
                        imgsz=st.session_state.imgsz,
                        verbose=False,
                    )
                    annotated_bgr = results[0].plot()  # BGR
                    last_annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                # Display the latest annotated frame (reused for skipped frames)
                frame_area.image(last_annotated, channels="RGB", use_container_width=True)

                # Stats
                frames += 1
                frame_idx += 1
                elapsed = max(time.time() - t0, 1e-6)
                live_fps = frames / elapsed
                stats_col1.metric("Source FPS (approx.)", f"{fps_src:.1f}")
                stats_col2.metric("Processed frames", f"{frames}")
                stats_col3.metric("App FPS", f"{live_fps:.1f}")

                # Optional tiny sleep for UI responsiveness; comment for max throughput
                # time.sleep(0.001)

        finally:
            cap.release()
            if cleanup_cb:
                cleanup_cb()
            st.success("Processing finished.")
