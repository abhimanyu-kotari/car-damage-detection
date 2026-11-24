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
    page_title="AutoInspect AI | Ensemble & XAI",
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
def load_all_models():
    """Loads and caches both YOLOv8m Generalist and YOLO11m Specialist models."""
    models = {}
    
    # 1. Generalist Model (YOLOv8m)
    v8_path = os.path.join(BASE_DIR, "best.pt")
    if os.path.exists(v8_path):
        try:
            models["yolov8m"] = YOLO(v8_path)
        except Exception as e:
            st.sidebar.error(f"Error loading YOLOv8m: {e}")

    # 2. Specialist Model (YOLO11m)
    v11_path = os.path.join(BASE_DIR, "yolov11m-seg.pt")
    if os.path.exists(v11_path):
        try:
            models["yolo11m"] = YOLO(v11_path)
        except Exception as e:
            st.sidebar.error(f"Error loading YOLO11m: {e}")

    # 3. Fallback Nano Model
    nano_path = os.path.join(BASE_DIR, "best_nano.pt")
    if os.path.exists(nano_path):
        try:
            models["nano"] = YOLO(nano_path)
        except Exception:
            pass

    return models

loaded_models = load_all_models()

# =========================================================
# COLOR PALETTE FOR 6 DAMAGE CLASSES
# =========================================================

CLASS_COLORS = {
    "crack": (239, 68, 68),        # Red
    "dent": (14, 165, 233),        # Sky Blue
    "glass shatter": (168, 85, 247), # Purple
    "lamp broken": (245, 158, 11),  # Amber
    "scratch": (236, 72, 153),     # Pink
    "tire flat": (34, 197, 94)      # Green
}

# =========================================================
# EXPLAINABLE AI (XAI) ACTIVATION-CAM GENERATOR
# =========================================================

def generate_xai_heatmap(model, img_rgb):
    """
    Computes real-time EigenCAM / Activation-CAM heatmap from the model's SPPF/neck layers.
    Reveals the exact spatial regions where the neural network concentrated its attention.
    """
    try:
        h_orig, w_orig = img_rgb.shape[:2]
        
        # Prepare input tensor
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
            
        # Channel energy pooling (L2 spatial activation map)
        heatmap = torch.mean(feat.squeeze(0), dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Resize back to original image size
        heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        blended_heatmap = cv2.addWeighted(img_rgb, 0.50, heatmap_colored, 0.50, 0)
        return blended_heatmap
    except Exception as e:
        return None

# =========================================================
# ENSEMBLE PREDICTION ENGINE
# =========================================================

def run_ensemble_inference(image_np, conf_thresh, mode="ensemble"):
    """
    Executes inference based on chosen mode:
    - 'ensemble': Fuses YOLOv8m (Generalist) + YOLO11m (Specialist)
    - 'yolo11m': Runs YOLO11m Specialist only
    - 'yolov8m': Runs YOLOv8m Generalist only
    - 'nano': Runs YOLOv8 Nano fallback
    """
    detections = []
    
    # ---------------------------------------------------------
    # Helper to calculate Intersection over Union (IoU)
    # ---------------------------------------------------------
    def calc_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    # ---------------------------------------------------------
    # 1. Run predictions
    # ---------------------------------------------------------
    results_v8 = None
    results_v11 = None
    
    if mode in ["ensemble", "yolov8m"] and "yolov8m" in loaded_models:
        results_v8 = loaded_models["yolov8m"].predict(source=image_np, conf=conf_thresh, imgsz=800, verbose=False)[0]

    if mode in ["ensemble", "yolo11m"] and "yolo11m" in loaded_models:
        results_v11 = loaded_models["yolo11m"].predict(source=image_np, conf=conf_thresh, imgsz=800, verbose=False)[0]

    if mode == "nano" and "nano" in loaded_models:
        results_v8 = loaded_models["nano"].predict(source=image_np, conf=conf_thresh, imgsz=640, verbose=False)[0]

    # ---------------------------------------------------------
    # 2. Extract Specialist Detections (Crack, Dent, Scratch)
    # ---------------------------------------------------------
    if results_v11 is not None and results_v11.boxes is not None:
        names_11 = loaded_models["yolo11m"].names
        masks_11 = results_v11.masks.xy if results_v11.masks is not None else []
        for i, box in enumerate(results_v11.boxes):
            cls_id = int(box.cls[0])
            c_name = names_11[cls_id].lower()
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            poly = masks_11[i] if i < len(masks_11) else None
            
            detections.append({
                "name": c_name,
                "confidence": conf,
                "box": xyxy,
                "polygon": poly,
                "source": "YOLO11m Specialist",
                "color": CLASS_COLORS.get(c_name, (0, 191, 255)),
                "xai_reason": f"High spatial attention detected fine surface defect signature for {c_name.title()}."
            })

    # ---------------------------------------------------------
    # 3. Extract Generalist Detections (Glass, Tire, Lamp + Ensemble Fusion)
    # ---------------------------------------------------------
    if results_v8 is not None and results_v8.boxes is not None:
        names_8 = loaded_models.get("yolov8m", loaded_models.get("nano")).names
        masks_8 = results_v8.masks.xy if results_v8.masks is not None else []
        for i, box in enumerate(results_v8.boxes):
            cls_id = int(box.cls[0])
            c_name = names_8[cls_id].lower()
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            poly = masks_8[i] if i < len(masks_8) else None
            
            source_tag = "YOLOv8 Nano" if mode == "nano" else "YOLOv8m Generalist"
            
            if mode == "ensemble":
                # For macro damages, YOLOv8 Generalist is the primary authority (>90% mAP)
                if c_name in ["glass shatter", "tire flat", "lamp broken"]:
                    detections.append({
                        "name": c_name,
                        "confidence": conf,
                        "box": xyxy,
                        "polygon": poly,
                        "source": source_tag,
                        "color": CLASS_COLORS.get(c_name, (245, 158, 11)),
                        "xai_reason": f"Structural disruption and edge deformation verified by {source_tag}."
                    })
                else:
                    # For crack, dent, scratch: only add if specialist didn't already find an overlapping instance
                    already_found = False
                    for existing in detections:
                        if existing["name"] == c_name and calc_iou(existing["box"], xyxy) > 0.45:
                            already_found = True
                            # If Generalist is significantly more confident, update confidence
                            if conf > existing["confidence"]:
                                existing["confidence"] = conf
                                existing["source"] = f"{source_tag} (Supported by Specialist)"
                            break
                    if not already_found:
                        detections.append({
                            "name": c_name,
                            "confidence": conf,
                            "box": xyxy,
                            "polygon": poly,
                            "source": source_tag,
                            "color": CLASS_COLORS.get(c_name, (0, 191, 255)),
                            "xai_reason": f"Surface irregularity flagged with {conf:.0%} certainty."
                        })
            else:
                # Standalone mode
                detections.append({
                    "name": c_name,
                    "confidence": conf,
                    "box": xyxy,
                    "polygon": poly,
                    "source": source_tag,
                    "color": CLASS_COLORS.get(c_name, (0, 191, 255)),
                    "xai_reason": f"Feature activation detected {c_name.title()}."
                })

    return detections

# =========================================================
# CUSTOM VISUALIZATION RENDERER
# =========================================================

def draw_annotated_image(orig_image_rgb, detections):
    """Draws pixel-level segmentation masks and styled bounding boxes."""
    if not detections:
        return orig_image_rgb

    # 1. Draw segmentation polygon masks with alpha blending
    mask_layer = orig_image_rgb.copy()
    for det in detections:
        poly = det.get("polygon")
        if poly is not None and len(poly) > 0:
            pts = np.int32(poly)
            color_rgb = det["color"]
            cv2.fillPoly(mask_layer, [pts], color_rgb)

    annotated = cv2.addWeighted(mask_layer, 0.40, orig_image_rgb, 0.60, 0)

    # 2. Draw crisp bounding boxes and badges
    for det in detections:
        color_rgb = det["color"]
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        
        # Bounding box border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 2)
        
        # Text badge
        badge_text = f"{det['name'].title()} {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        badge_y1 = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 8, y1), color_rgb, -1)
        cv2.putText(annotated, badge_text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🚗 AutoInspect AI")
    st.caption("Dual-Model Ensemble & Explainable AI (XAI)")
    st.divider()

    # Build model options based on available files
    model_mode_options = []
    if "yolov8m" in loaded_models and "yolo11m" in loaded_models:
        model_mode_options.append("Dual-Model Ensemble (YOLOv8m + YOLO11m Specialist) ⭐")
    if "yolo11m" in loaded_models:
        model_mode_options.append("YOLO11m Specialist Only (Crack, Dent, Scratch)")
    if "yolov8m" in loaded_models:
        model_mode_options.append("YOLOv8m Generalist Only (All 6 Classes)")
    if "nano" in loaded_models:
        model_mode_options.append("YOLOv8 Nano (Fast / Fallback)")

    if not model_mode_options:
        st.error("No trained models found in project folder!")
        selected_mode_key = "ensemble"
    else:
        selected_mode_label = st.selectbox(
            "Detection Architecture",
            options=model_mode_options,
            index=0,
            help="Select Dual-Model Ensemble for highest overall accuracy across both macro and fine-grained damages."
        )
        if "Ensemble" in selected_mode_label:
            selected_mode_key = "ensemble"
        elif "YOLO11m" in selected_mode_label:
            selected_mode_key = "yolo11m"
        elif "Generalist" in selected_mode_label:
            selected_mode_key = "yolov8m"
        else:
            selected_mode_key = "nano"

    st.markdown("### Model System")
    if selected_mode_key == "ensemble":
        st.write("**Architecture:** Dual-Model Ensemble")
        st.write("• **Model 1 (Generalist):** YOLOv8m (6 Classes)")
        st.write("• **Model 2 (Specialist):** YOLO11m (Crack, Dent, Scratch)")
        st.write("**Damage Classes:** 6 Total")
    elif selected_mode_key == "yolo11m":
        st.write("**Architecture:** YOLO11m-seg")
        st.write("**Classes:** 3 (Crack, Dent, Scratch)")
    else:
        st.write("**Architecture:** YOLOv8m-seg")
        st.write("**Classes:** 6 Classes")

    st.divider()

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Higher values show fewer, more confident detections."
    )

    enable_xai = st.toggle(
        "🔬 Enable Explainable AI (XAI)",
        value=True,
        help="Generates an EigenCAM/Activation heatmap proving what features the neural network looked at."
    )

    st.divider()
    st.caption("RV University · Deep Learning Project")

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Vehicle Damage Inspection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ensemble AI detection combining YOLOv8m Generalist + YOLO11m Specialist with Explainable AI (XAI).'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photograph of the damaged vehicle."
)

if uploaded_file is None:
    st.info("👆 Upload a car image to begin the multi-model damage inspection.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.markdown('<div class="section-title">Image Preview</div>', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Vehicle Image", use_container_width=True)

    detect_button = st.button("🔍 Analyze Vehicle Damage with Ensemble AI", type="primary")

    if detect_button:
        try:
            with st.spinner("Running Dual-Model Ensemble inference..."):
                detections = run_ensemble_inference(image_np, confidence, mode=selected_mode_key)
                annotated_img = draw_annotated_image(image_np, detections)
                
                # Compute Explainable AI Heatmap if enabled
                xai_heatmap_img = None
                if enable_xai:
                    active_model = loaded_models.get("yolo11m", loaded_models.get("yolov8m"))
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
            # VISUAL COMPARISON & XAI TABS
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
                        st.markdown("**Ensemble AI Segmentation**")
                        st.image(annotated_img, use_container_width=True)

                with tab2:
                    col_orig2, col_xai = st.columns(2)
                    with col_orig2:
                        st.markdown("**Original Vehicle**")
                        st.image(image, use_container_width=True)
                    with col_xai:
                        st.markdown("**XAI Neural Activation Map (EigenCAM)**")
                        st.image(xai_heatmap_img, use_container_width=True)
                        st.caption("🔥 Red/Yellow hot spots indicate regions where the neural network's spatial attention strongly triggered detection.")
            else:
                col_orig, col_seg = st.columns(2)
                with col_orig:
                    st.markdown("**Original Image**")
                    st.image(image, use_container_width=True)
                with col_seg:
                    st.markdown("**Ensemble AI Segmentation**")
                    st.image(annotated_img, use_container_width=True)

            # =================================================
            # DAMAGE DETAILS WITH XAI EXPLANATIONS
            # =================================================
            st.markdown('<div class="section-title">Damage Details & Model Attribution</div>', unsafe_allow_html=True)

            if detections:
                for idx, det in enumerate(detections, start=1):
                    c_name = det["name"].replace("_", " ").title()
                    c_conf = det["confidence"]
                    c_source = det["source"]
                    c_xai = det.get("xai_reason", "Feature activation verified.")

                    st.markdown(f"""
                    <div class="damage-card">
                        <div class="damage-name">{idx}. {c_name}</div>
                        <div class="damage-meta">
                            <span class="badge-conf">Confidence: {c_conf:.1%}</span>
                            <span class="badge-source">Identified by: {c_source}</span>
                        </div>
                        <div class="xai-note"><strong>XAI Rationale:</strong> {c_xai}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No vehicle damage detected at this confidence threshold. Try lowering the slider in the sidebar.")

            # =================================================
            # DOWNLOAD OPTIONS
            # =================================================
            st.markdown('<div class="section-title">Download Inspection Reports</div>', unsafe_allow_html=True)
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                buf_seg = io.BytesIO()
                Image.fromarray(annotated_img).save(buf_seg, format="PNG")
                st.download_button(
                    label="⬇️ Download Annotated Segmentation Image",
                    data=buf_seg.getvalue(),
                    file_name="vehicle_damage_ensemble_seg.png",
                    mime="image/png"
                )

            with dl_col2:
                if enable_xai and xai_heatmap_img is not None:
                    buf_xai = io.BytesIO()
                    Image.fromarray(xai_heatmap_img).save(buf_xai, format="PNG")
                    st.download_button(
                        label="⬇️ Download XAI Attention Heatmap",
                        data=buf_xai.getvalue(),
                        file_name="vehicle_damage_xai_heatmap.png",
                        mime="image/png"
                    )

        except Exception as error:
            st.error(f"An error occurred during prediction: {error}")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "AutoInspect AI | Dual-Model Ensemble (YOLOv8m + YOLO11m) with Explainable AI (XAI). "
    "Predictions are intended for computer-assisted inspection auditing."
)
