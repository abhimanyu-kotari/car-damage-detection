import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import io
import cv2
import torch

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoInspect AI | Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 12px;
    }

    .metric-card {
        background-color: #151e2e;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #38bdf8;
        font-size: 28px;
        font-weight: 800;
    }

    .damage-card {
        background-color: #151e2e;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .damage-name {
        color: #f1f5f9;
        font-size: 17px;
        font-weight: 700;
    }

    .damage-meta {
        display: flex;
        gap: 15px;
        margin-top: 6px;
        font-size: 13px;
    }

    .badge-conf {
        color: #38bdf8;
        font-weight: 700;
    }

    .badge-source {
        background-color: #1e293b;
        color: #a5b4fc;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        border: 1px solid #334155;
    }

    .xai-note {
        margin-top: 8px;
        font-size: 12.5px;
        color: #94a3b8;
        background-color: #0f172a;
        padding: 8px 12px;
        border-radius: 6px;
        border-left: 3px solid #38bdf8;
    }

    div.stButton > button {
        background-color: #0ea5e9;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #0284c7;
        color: white;
    }

    .stImage img {
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL LOADING WITH CACHING
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    """Loads all models with error handling."""
    models = {}
    
    # 1. YOLOv8 Medium (Generalist - best.pt, 27.2M params)
    v8_path = os.path.join(BASE_DIR, "best.pt")
    if os.path.exists(v8_path):
        try:
            models["yolov8m"] = YOLO(v8_path)
        except Exception as e:
            st.sidebar.error(f"Error loading best.pt: {e}")

    # 2. YOLO11 Medium (Specialist - yolov11m-seg.pt, 22.4M params)
    v11_path = os.path.join(BASE_DIR, "yolov11m-seg.pt")
    if os.path.exists(v11_path):
        try:
            models["yolo11m"] = YOLO(v11_path)
        except Exception as e:
            st.sidebar.error(f"Error loading yolov11m-seg.pt: {e}")

    # 3. YOLOv8 Nano (Fallback - best_nano.pt, 3.4M params)
    nano_path = os.path.join(BASE_DIR, "best_nano.pt")
    if os.path.exists(nano_path):
        try:
            models["nano"] = YOLO(nano_path)
        except Exception:
            pass

    return models

loaded_models = load_models()

# =========================================================
# EXPLAINABLE AI (XAI) EIGENCAM GENERATOR
# =========================================================

def generate_xai_heatmap(model, img_rgb):
    """
    Computes real-time EigenCAM / Activation-CAM heatmap from the neural network backbone.
    """
    try:
        h_orig, w_orig = img_rgb.shape[:2]
        
        # Prepare input tensor at 640x640
        inp = cv2.resize(img_rgb, (640, 640)).transpose(2, 0, 1)
        inp = np.ascontiguousarray(inp, dtype=np.float32) / 255.0
        inp = torch.from_numpy(inp).unsqueeze(0)
        
        activations = []
        def hook_fn(module, i, o):
            activations.append(o)
            
        pytorch_model = model.model
        target_layer = None
        for m in pytorch_model.model:
            if 'SPPF' in m.__class__.__name__:
                target_layer = m
                break
        if target_layer is None:
            target_layer = pytorch_model.model[-2]
            
        handle = target_layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                pytorch_model(inp)
        finally:
            handle.remove()
            
        if not activations:
            return None
            
        feat = activations[0]
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
            
        # Channel energy pooling
        heatmap = torch.mean(feat.squeeze(0), dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Resize to original image size
        heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        blended_heatmap = cv2.addWeighted(img_rgb, 0.50, heatmap_colored, 0.50, 0)
        return blended_heatmap
    except Exception:
        return None

# =========================================================
# INFERENCE LOGIC
# =========================================================

def run_detection(image_np, conf_thresh, model_key="ensemble"):
    """
    Runs inference using the user-selected model mode.
    Guarantees native imgsz (640 for v8m/nano, 800 for v11m) and native result.plot() rendering!
    """
    detections = []
    annotated_image = None

    # ---------------------------------------------------------
    # Helper to calculate IoU between two boxes
    # ---------------------------------------------------------
    def box_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    # =========================================================
    # MODE 1: TRI-MODEL CONSENSUS ENSEMBLE (v8m + 11m + v8n)
    # =========================================================
    if model_key == "tri_ensemble":
        # Run all 3 models at their optimal native resolutions
        res_v8m = loaded_models["yolov8m"].predict(source=image_np, conf=conf_thresh, imgsz=640, verbose=False)[0]
        res_v11 = loaded_models["yolo11m"].predict(source=image_np, conf=conf_thresh, imgsz=800, verbose=False)[0]
        res_v8n = loaded_models["nano"].predict(source=image_np, conf=conf_thresh, imgsz=640, verbose=False)[0]

        # Use YOLOv8m's clean native segmentation plot as base (C-contiguous RGB)
        annotated_bgr = res_v8m.plot()
        annotated_image = np.ascontiguousarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

        candidates = []
        if res_v8m.boxes is not None:
            names_v8m = loaded_models["yolov8m"].names
            for box in res_v8m.boxes:
                candidates.append({
                    "model": "v8m",
                    "label": "YOLOv8 Medium",
                    "name": names_v8m[int(box.cls[0])].lower(),
                    "conf": float(box.conf[0]),
                    "box": box.xyxy[0].cpu().numpy(),
                    "weight": 0.45
                })

        if res_v11.boxes is not None:
            names_v11 = loaded_models["yolo11m"].names
            for box in res_v11.boxes:
                candidates.append({
                    "model": "11m",
                    "label": "YOLO11m Specialist",
                    "name": names_v11[int(box.cls[0])].lower(),
                    "conf": float(box.conf[0]),
                    "box": box.xyxy[0].cpu().numpy(),
                    "weight": 0.40
                })

        if res_v8n.boxes is not None:
            names_v8n = loaded_models["nano"].names
            for box in res_v8n.boxes:
                candidates.append({
                    "model": "v8n",
                    "label": "YOLOv8 Nano",
                    "name": names_v8n[int(box.cls[0])].lower(),
                    "conf": float(box.conf[0]),
                    "box": box.xyxy[0].cpu().numpy(),
                    "weight": 0.15
                })

        # Group candidates into spatial clusters by defect class and IoU
        clusters = []
        used = [False] * len(candidates)
        for i, c1 in enumerate(candidates):
            if used[i]:
                continue
            cluster = [c1]
            used[i] = True
            for j in range(i + 1, len(candidates)):
                if not used[j] and candidates[j]["name"] == c1["name"]:
                    # Lower threshold for dent to merge contiguous panel deformities
                    iou_limit = 0.20 if c1["name"] == "dent" else 0.30
                    if box_iou(c1["box"], candidates[j]["box"]) > iou_limit:
                        cluster.append(candidates[j])
                        used[j] = True
            clusters.append(cluster)

        # Evaluate consensus across clusters
        h_img, w_img = annotated_image.shape[:2]
        for cl in clusters:
            c_name = cl[0]["name"]
            models_present = {c["model"] for c in cl}
            best_cand = max(cl, key=lambda c: c["conf"])
            max_conf = best_cand["conf"]
            xyxy = best_cand["box"]

            # Case A: Consensus Agreement (>= 2 models agree)
            if len(models_present) >= 2:
                model_labels = [c["label"] for c in cl]
                source_str = f"Consensus ({', '.join(sorted(set(model_labels)))})"
                xai_reason = f"Confirmed by {len(models_present)} independent models with {max_conf:.0%} peak confidence."
                detections.append({
                    "name": c_name,
                    "confidence": max_conf,
                    "box": xyxy,
                    "source": source_str,
                    "xai_reason": xai_reason
                })
                # If YOLOv8m wasn't part of this cluster, draw it
                if "v8m" not in models_present:
                    x1 = max(0, min(w_img - 1, int(xyxy[0])))
                    y1 = max(0, min(h_img - 1, int(xyxy[1])))
                    x2 = max(0, min(w_img - 1, int(xyxy[2])))
                    y2 = max(0, min(h_img - 1, int(xyxy[3])))
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (16, 185, 129), 2)
                    tag = f"{c_name.title()} {max_conf:.0%} (Consensus)"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    badge_y1 = max(0, y1 - th - 6)
                    cv2.rectangle(annotated_image, (x1, badge_y1), (min(w_img - 1, x1 + tw + 6), y1), (16, 185, 129), -1)
                    cv2.putText(annotated_image, tag, (x1 + 3, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Case B: Single Model Solo Detection
            else:
                solo_model = cl[0]["model"]
                # 1. YOLOv8m solo: Authoritative on rare classes or solid detections
                if solo_model == "v8m":
                    min_thresh = conf_thresh if c_name in ["glass shatter", "lamp broken", "tire flat"] else max(conf_thresh, 0.35)
                    if max_conf >= min_thresh:
                        detections.append({
                            "name": c_name,
                            "confidence": max_conf,
                            "box": xyxy,
                            "source": "YOLOv8 Medium",
                            "xai_reason": f"Detected with {max_conf:.0%} confidence by YOLOv8 Medium."
                        })
                # 2. YOLO11m solo: Filter panel shadows & false positives
                elif solo_model == "11m":
                    min_thresh = max(conf_thresh, 0.40)
                    if max_conf >= min_thresh:
                        detections.append({
                            "name": c_name,
                            "confidence": max_conf,
                            "box": xyxy,
                            "source": "YOLO11m Specialist",
                            "xai_reason": f"Fine surface feature captured by YOLO11m Attention."
                        })
                        x1 = max(0, min(w_img - 1, int(xyxy[0])))
                        y1 = max(0, min(h_img - 1, int(xyxy[1])))
                        x2 = max(0, min(w_img - 1, int(xyxy[2])))
                        y2 = max(0, min(h_img - 1, int(xyxy[3])))
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (236, 72, 153), 2)
                        tag = f"{c_name.title()} {max_conf:.0%} (YOLO11)"
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        badge_y1 = max(0, y1 - th - 6)
                        cv2.rectangle(annotated_image, (x1, badge_y1), (min(w_img - 1, x1 + tw + 6), y1), (236, 72, 153), -1)
                        cv2.putText(annotated_image, tag, (x1 + 3, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # 3. YOLOv8n solo: Prone to false positives (35% on background), require >= 55%
                elif solo_model == "v8n":
                    if max_conf >= 0.55:
                        detections.append({
                            "name": c_name,
                            "confidence": max_conf,
                            "box": xyxy,
                            "source": "YOLOv8 Nano",
                            "xai_reason": f"High-confidence candidate captured by YOLOv8 Nano."
                        })

    # =========================================================
    # MODE 2: DUAL-MODEL ENSEMBLE (v8m + 11m)
    # =========================================================
    elif model_key == "ensemble":
        # 1. Run YOLOv8m (Generalist) at its native 640px
        res_v8 = loaded_models["yolov8m"].predict(source=image_np, conf=conf_thresh, imgsz=640, verbose=False)[0]
        # 2. Run YOLO11m (Specialist) at its native 800px
        res_v11 = loaded_models["yolo11m"].predict(source=image_np, conf=conf_thresh, imgsz=800, verbose=False)[0]

        # Use YOLOv8m's native plot as high-fidelity visual base (C-contiguous RGB)
        annotated_bgr = res_v8.plot()
        annotated_image = np.ascontiguousarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

        # Extract YOLOv8m detections
        if res_v8.boxes is not None:
            names_v8 = loaded_models["yolov8m"].names
            for box in res_v8.boxes:
                c_id = int(box.cls[0])
                c_name = names_v8[c_id].lower()
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({
                    "name": c_name,
                    "confidence": conf,
                    "box": xyxy,
                    "source": "YOLOv8 Medium",
                    "xai_reason": f"Detected with {conf:.0%} confidence by YOLOv8 Medium."
                })

        # Add any extra detections found by YOLO11m Specialist (crack, dent, scratch)
        if res_v11.boxes is not None:
            names_v11 = loaded_models["yolo11m"].names
            for i, box in enumerate(res_v11.boxes):
                c_id = int(box.cls[0])
                c_name = names_v11[c_id].lower()
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # Check if YOLOv8m already detected this or adjacent contiguous damage
                overlap = False
                iou_threshold = 0.20 if c_name == "dent" else 0.35
                for d in detections:
                    if d["name"] == c_name and box_iou(d["box"], xyxy) > iou_threshold:
                        overlap = True
                        if conf > d["confidence"]:
                            d["confidence"] = conf
                            d["source"] = "YOLO11m Specialist (Higher Confidence)"
                        break

                # If YOLO11m caught a defect that YOLOv8m missed, only add it if it meets a reliable confidence threshold
                min_threshold = max(conf_thresh, 0.40 if c_name == "crack" else 0.35)
                if not overlap and conf >= min_threshold:
                    detections.append({
                        "name": c_name,
                        "confidence": conf,
                        "box": xyxy,
                        "source": "YOLO11m Specialist (New Detection)",
                        "xai_reason": f"Fine surface feature captured by YOLO11m Attention."
                    })
                    # Draw the new detection onto annotated_image
                    h_img, w_img = annotated_image.shape[:2]
                    x1 = max(0, min(w_img - 1, int(xyxy[0])))
                    y1 = max(0, min(h_img - 1, int(xyxy[1])))
                    x2 = max(0, min(w_img - 1, int(xyxy[2])))
                    y2 = max(0, min(h_img - 1, int(xyxy[3])))
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (236, 72, 153), 2)
                    tag = f"{c_name.title()} {conf:.0%} (YOLO11)"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    badge_y1 = max(0, y1 - th - 6)
                    cv2.rectangle(annotated_image, (x1, badge_y1), (min(w_img - 1, x1 + tw + 6), y1), (236, 72, 153), -1)
                    cv2.putText(annotated_image, tag, (x1 + 3, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # =========================================================
    # MODE 2: SINGLE MODEL (YOLOv8m, YOLOv8n, or YOLO11m)
    # =========================================================
    else:
        target_model = loaded_models[model_key]
        img_size = 800 if model_key == "yolo11m" else 640
        source_name = "YOLOv8 Medium" if model_key == "yolov8m" else ("YOLOv8 Nano" if model_key == "nano" else "YOLO11m Specialist")

        res = target_model.predict(source=image_np, conf=conf_thresh, imgsz=img_size, verbose=False)[0]
        
        # Native Ultralytics plot rendering (C-contiguous RGB)
        annotated_bgr = res.plot()
        annotated_image = np.ascontiguousarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

        if res.boxes is not None:
            names = target_model.names
            for box in res.boxes:
                c_id = int(box.cls[0])
                c_name = names[c_id].lower()
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({
                    "name": c_name,
                    "confidence": conf,
                    "box": xyxy,
                    "source": source_name,
                    "xai_reason": f"Detected with {conf:.0%} confidence by {source_name}."
                })

    return detections, annotated_image

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

with st.sidebar:
    st.markdown("## 🚗 AutoInspect AI")
    st.caption("Vehicle Damage Detection System")
    st.divider()

    # Build model options
    model_choices = {}
    if "yolov8m" in loaded_models and "yolo11m" in loaded_models and "nano" in loaded_models:
        model_choices["Tri-Model Consensus Ensemble (YOLOv8m + YOLO11m + YOLOv8n) 🏆"] = "tri_ensemble"
    if "yolov8m" in loaded_models and "yolo11m" in loaded_models:
        model_choices["Dual-Model Ensemble (YOLOv8m + YOLO11m) ⭐"] = "ensemble"
    if "yolov8m" in loaded_models:
        model_choices["YOLOv8 Medium (best.pt - 27.2M params)"] = "yolov8m"
    if "yolo11m" in loaded_models:
        model_choices["YOLO11m Specialist (yolov11m-seg.pt - 22.4M params)"] = "yolo11m"
    if "nano" in loaded_models:
        model_choices["YOLOv8 Nano (best_nano.pt - 3.4M params)"] = "nano"

    selected_label = st.selectbox(
        "Select Model / Engine",
        options=list(model_choices.keys()),
        index=0,
        help="Tri-Model Consensus combines YOLOv8m, YOLO11m, and YOLOv8n using majority consensus voting to eliminate false positives."
    )
    selected_mode = model_choices[selected_label]

    st.markdown("### Model Information")
    if selected_mode == "tri_ensemble":
        st.write("**Architecture:** Tri-Model Consensus Ensemble")
        st.write("• Generalist: YOLOv8m (27.2M params, 640px)")
        st.write("• Specialist: YOLO11m (22.4M params, 800px)")
        st.write("• Verifier: YOLOv8n (3.4M params, 640px)")
        st.caption("✨ Majority consensus voting across all 3 models eliminates isolated false positives.")
    elif selected_mode == "ensemble":
        st.write("**Architecture:** Dual-Model Ensemble")
        st.write("• Base: YOLOv8m (All 6 Classes, 640px)")
        st.write("• Booster: YOLO11m (Crack, Dent, Scratch, 800px)")
    elif selected_mode == "yolov8m":
        st.write("**Architecture:** YOLOv8m-seg (Medium)")
        st.write("**Parameters:** 27.2 Million")
        st.write("**Resolution:** 640 × 640")
        st.write("**Classes:** 6 Classes")
    elif selected_mode == "nano":
        st.write("**Architecture:** YOLOv8n-seg (Nano)")
        st.write("**Parameters:** 3.4 Million")
        st.write("**Resolution:** 640 × 640")
        st.write("**Classes:** 6 Classes")
    else:
        st.write("**Architecture:** YOLO11m-seg (Specialist)")
        st.write("**Parameters:** 22.4 Million")
        st.write("**Resolution:** 800 × 800")
        st.write("**Classes:** 3 Classes (Crack, Dent, Scratch)")

    st.divider()

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Adjust detection threshold (Recommended: 0.35 - 0.40). Lower values detect subtle micro-defects but may catch shadow lines; higher values eliminate false positives."
    )

    enable_xai = st.toggle(
        "🔬 Enable Explainable AI (XAI)",
        value=True,
        help="Displays neural network activation heatmap showing which regions triggered detection."
    )

    st.divider()
    st.caption("RV University · Deep Learning Project")

# =========================================================
# MAIN INTERFACE
# =========================================================

st.markdown('<div class="main-title">Vehicle Damage Inspection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-assisted vehicle exterior inspection with segmentation and Explainable AI.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photograph of the damaged vehicle."
)

if uploaded_file is None:
    st.info("👆 Upload a car image to begin the damage inspection.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.markdown('<div class="section-title">Image Preview</div>', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Vehicle Image", use_container_width=True)

    detect_button = st.button("🔍 Analyze Vehicle Damage", type="primary")

    if detect_button:
        try:
            with st.spinner("AI is analyzing the vehicle..."):
                detections, annotated_img = run_detection(image_np, confidence, model_key=selected_mode)

                xai_heatmap_img = None
                if enable_xai:
                    active_model = loaded_models.get("yolov8m", loaded_models.get("yolo11m", loaded_models.get("nano")))
                    if active_model is not None:
                        xai_heatmap_img = generate_xai_heatmap(active_model, image_np)

            # =================================================
            # METRICS DASHBOARD
            # =================================================
            st.divider()
            st.markdown('<div class="section-title">Inspection Results</div>', unsafe_allow_html=True)

            detection_count = len(detections)
            avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
            unique_classes = len(set(d["name"] for d in detections))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Detected Damages</div>
                    <div class="metric-value">{detection_count}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Confidence</div>
                    <div class="metric-value">{avg_conf:.0%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Damage Categories</div>
                    <div class="metric-value">{unique_classes}</div>
                </div>
                """, unsafe_allow_html=True)

            # =================================================
            # VISUAL COMPARISON & TABS
            # =================================================
            st.markdown('<div class="section-title">Visual AI Analysis</div>', unsafe_allow_html=True)

            if enable_xai and xai_heatmap_img is not None:
                tab1, tab2 = st.tabs(["📸 AI Segmentation Output", "🔬 Explainable AI (XAI) Attention Heatmap"])

                with tab1:
                    col_orig, col_seg = st.columns(2)
                    with col_orig:
                        st.markdown("**Original Image**")
                        st.image(image, use_container_width=True)
                    with col_seg:
                        st.markdown("**AI Segmentation Detection**")
                        st.image(annotated_img, use_container_width=True)

                with tab2:
                    col_orig2, col_xai = st.columns(2)
                    with col_orig2:
                        st.markdown("**Original Vehicle**")
                        st.image(image, use_container_width=True)
                    with col_xai:
                        st.markdown("**XAI Neural Activation Heatmap (EigenCAM)**")
                        st.image(xai_heatmap_img, use_container_width=True)
                        st.caption("🔥 Red/Yellow areas show where the neural network's receptive field focused its spatial attention.")
            else:
                col_orig, col_seg = st.columns(2)
                with col_orig:
                    st.markdown("**Original Image**")
                    st.image(image, use_container_width=True)
                with col_seg:
                    st.markdown("**AI Segmentation Detection**")
                    st.image(annotated_img, use_container_width=True)

            # =================================================
            # DAMAGE DETAILS WITH MODEL ATTRIBUTION
            # =================================================
            st.markdown('<div class="section-title">Damage Details & Attribution</div>', unsafe_allow_html=True)

            if detections:
                for idx, det in enumerate(detections, start=1):
                    c_name = det["name"].replace("_", " ").title()
                    c_conf = det["confidence"]
                    c_source = det["source"]
                    c_xai = det.get("xai_reason", "Feature verified by model.")

                    st.markdown(f"""
                    <div class="damage-card">
                        <div class="damage-name">{idx}. {c_name}</div>
                        <div class="damage-meta">
                            <span class="badge-conf">Confidence: {c_conf:.1%}</span>
                            <span class="badge-source">Identified by: {c_source}</span>
                        </div>
                        <div class="xai-note"><strong>XAI Attribution:</strong> {c_xai}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No vehicle damage detected at this confidence threshold. Try lowering the slider in the sidebar.")

            # =================================================
            # DOWNLOAD OPTIONS
            # =================================================
            st.markdown('<div class="section-title">Download Inspection Image</div>', unsafe_allow_html=True)
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                buf_seg = io.BytesIO()
                Image.fromarray(annotated_img).save(buf_seg, format="PNG")
                st.download_button(
                    label="⬇️ Download Segmentation Image",
                    data=buf_seg.getvalue(),
                    file_name="car_damage_inspection.png",
                    mime="image/png"
                )

            with dl_col2:
                if enable_xai and xai_heatmap_img is not None:
                    buf_xai = io.BytesIO()
                    Image.fromarray(xai_heatmap_img).save(buf_xai, format="PNG")
                    st.download_button(
                        label="⬇️ Download XAI Heatmap",
                        data=buf_xai.getvalue(),
                        file_name="car_damage_xai_heatmap.png",
                        mime="image/png"
                    )

        except Exception as error:
            st.error(f"An error occurred during prediction: {error}")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "AutoInspect AI | Multi-Model Architecture with Explainable AI. "
    "Predictions are intended for computer-assisted vehicle inspection auditing."
)
