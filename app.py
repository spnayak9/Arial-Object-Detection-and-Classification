
import io
import os
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ultralytics
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

st.set_page_config(page_title="YOLOv8 — Object Detection Demo", layout="wide")

# ---------- CONFIG ----------
MODEL_PATH = "Models/bestall.pt"  # place bestall.pt next to this script
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
CLASS_NAMES = None  # optional: ["Bird","drone", ...] — set if you want custom names
# ----------------------------

st.title("YOLOv8 Object Detection — Simple App")
st.write("Upload an image and see detections. Model runs on CPU by default for local safety.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_path_input = st.text_input("YOLO model path", MODEL_PATH)
    conf_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=float(DEFAULT_CONF), step=0.01)
    iou_thresh = st.slider("NMS IoU threshold", min_value=0.0, max_value=1.0, value=float(DEFAULT_IOU), step=0.01)
    device_choice = st.selectbox("Device", options=["cpu"], index=0, help="Use 'cpu' locally. If you have working GPU + compatible torch/CUDA, you can set 'cuda:0' manually.")
    show_raw = st.checkbox("Show raw YOLO results (debug)", value=False)
    st.markdown("---")
    st.write("Model file should be next to this script or provide full path above.")
    if st.button("Reload model"):
        # set a sentinel to reload cached model
        st.session_state.pop("_yolo_model_cache", None)
        st.experimental_rerun()

# --------- Model loader (cached) ----------
@st.cache_resource(show_spinner=True)
def load_yolo_model(path: str):
    path = Path(path)
    if not path.exists():
        st.error(f"YOLO model not found at: {path.resolve()}")
        return None
    if YOLO is None:
        st.error("Ultralytics package not installed. Run `pip install ultralytics`.")
        return None
    try:
        yolo = YOLO(str(path))
        return yolo
    except Exception as e:
        st.error("Failed to load YOLO model. See traceback below.")
        st.write(e)
        return None

yolo_model = load_yolo_model(model_path_input)

# --------- Image upload / input ----------
st.subheader("Input")
col_u1, col_u2 = st.columns([2,1])
with col_u1:
    uploaded = st.file_uploader("Upload one image (jpg/png) or multiple images", accept_multiple_files=True, type=["jpg","jpeg","png"])
    image_url = st.text_input("Or paste an image URL and press Enter", value="")

with col_u2:
    st.write("Tips")
    st.write("- Use small images for fast inference (<= 2MP).")
    st.write("- To process multiple, upload multiple files.")
    st.write("- Model runs on CPU by default to avoid CUDA mismatches.")

# gather images list
images = []
if uploaded:
    for up in uploaded:
        try:
            images.append((up.name, Image.open(up).convert("RGB")))
        except Exception as e:
            st.warning(f"Could not open {up.name}: {e}")

if image_url:
    try:
        import requests
        resp = requests.get(image_url, timeout=8)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            images.append(("url_image", img))
        else:
            st.warning("Could not fetch image from URL (status != 200).")
    except Exception as e:
        st.warning("Error fetching image URL: " + str(e))

if not images:
    st.info("Upload or provide an image URL to start.")
    st.stop()

# ---------- helpers ----------
def draw_detections(pil_img: Image.Image, detections, label_font=None):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default() if label_font is None else ImageFont.truetype(label_font, size=14)
    except Exception:
        font = ImageFont.load_default()
    for det in detections:
        x1,y1,x2,y2 = det["box"]
        cls = det["class_name"]
        conf = det["confidence"]
        color = (255, 0, 0) if str(cls).lower().startswith("bird") else (0, 160, 255)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        label = f"{cls} {conf:.2f}"
        # text bbox robust
        try:
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = bbox[2]-bbox[0]; text_h = bbox[3]-bbox[1]
        except Exception:
            text_w = len(label)*6; text_h = 12
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=(0,0,0))
        draw.text((x1+2, y1 - text_h - 2), label, fill=color, font=font)
    return img

def parse_yolo_result(res, conf_threshold=0.25, class_names=None):
    """
    Robustly parse a single ultralytics Results object (res)
    Returns list of detections: {'box':[x1,y1,x2,y2], 'class_name':str, 'confidence':float}
    """
    dets = []
    try:
        # Option A: res.boxes.data -> numpy Nx6 [x1,y1,x2,y2,conf,cls]
        if hasattr(res, "boxes") and hasattr(res.boxes, "data"):
            data = res.boxes.data.cpu().numpy() if hasattr(res.boxes.data, "cpu") else np.array(res.boxes.data)
            for row in data:
                x1,y1,x2,y2,conf,cls = row.tolist()
                if conf < conf_threshold: 
                    continue
                name = (class_names[int(cls)] if class_names and int(cls) < len(class_names) else f"class_{int(cls)}")
                dets.append({"box":[int(x1),int(y1),int(x2),int(y2)], "class_name":name, "confidence":float(conf)})
            return dets
        # Option B: res.boxes.xyxy + conf + cls
        if hasattr(res, "boxes"):
            xyxy = getattr(res.boxes, "xyxy", None)
            confs = getattr(res.boxes, "conf", None)
            clss = getattr(res.boxes, "cls", None)
            if xyxy is not None:
                xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)
                confs = confs.cpu().numpy() if confs is not None and hasattr(confs, "cpu") else (np.array(confs) if confs is not None else np.zeros(len(xyxy)))
                clss = clss.cpu().numpy() if clss is not None and hasattr(clss, "cpu") else (np.array(clss) if clss is not None else np.zeros(len(xyxy)))
                for i in range(len(xyxy)):
                    x1,y1,x2,y2 = xyxy[i].tolist()
                    conf = float(confs[i]) if len(confs)>0 else 0.0
                    cls = int(clss[i]) if len(clss)>0 else 0
                    if conf < conf_threshold: 
                        continue
                    name = (class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}")
                    dets.append({"box":[int(x1),int(y1),int(x2),int(y2)], "class_name":name, "confidence":conf})
                return dets
    except Exception as e:
        st.write("Error parsing YOLO result:", e)
    return dets

# ---------- Process & display ----------
for fname, pil_img in images:
    st.markdown(f"### Image: **{fname}**")
    col1, col2 = st.columns([1,1])

    # show original
    with col1:
        st.image(pil_img, caption="Original", use_container_width=True)

    # run detection
    with st.spinner("Running YOLO inference..."):
        if yolo_model is None:
            st.error("YOLO model not loaded. Check model path and ultralytics installation.")
            st.stop()
        try:
            # use yolo_model.predict or call; we pass device and conf/nms settings directly per-call if supported
            # ultralytics accepts kwargs like conf, iou, device on call
            results = yolo_model(pil_img, device=device_choice, verbose=False, conf=conf_threshold, iou=iou_thresh)
        except TypeError:
            # older ultralytics may not accept conf/iou kwargs; call default then filter
            results = yolo_model(pil_img, device=device_choice, verbose=False)

    # parse detections
    detections = []
    if results and len(results) > 0:
        r = results[0]
        detections = parse_yolo_result(r, conf_threshold=conf_threshold, class_names=CLASS_NAMES)

    # annotated image
    annotated = draw_detections(pil_img, detections)
    with col2:
        st.image(annotated, caption="Annotated", use_container_width=True)
        # download button
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(label="Download annotated image", data=buf, file_name=f"annotated_{fname}.png", mime="image/png")

    # show table of detections
    if detections:
        st.write("Detections")
        rows = []
        for i, d in enumerate(detections, start=1):
            x1,y1,x2,y2 = d["box"]
            rows.append({"#": i, "class": d["class_name"], "conf": f"{d['confidence']:.3f}", "x1":x1,"y1":y1,"x2":x2,"y2":y2})
        st.table(rows)
    else:
        st.write("No detections above threshold.")

    if show_raw:
        st.write("Raw YOLO results (object):")
        st.write(results[0] if results else None)

st.success("Done.")
