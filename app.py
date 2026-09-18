"""
GreenScan - drone-based crop health analysis
============================================
Streamlit app that maps field health from drone images.

* Multispectral mode (Red + NIR uploaded): NDVI, NDRE, LCI, GNDVI
* RGB mode (only a normal photo): VARI and GLI, with fewer checks

No synthetic data is ever generated: every result comes from the uploaded images.
Sample images can be placed in a `sample_data/` folder next to this file.
"""

import hashlib
import html
import io
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

st.set_page_config(
    page_title="GreenScan | Crop health from drone images",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"
IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]
IMAGE_EXTS = tuple(f".{t}" for t in IMAGE_TYPES)

MAX_PROC_SIDE = 2000      # images are processed at most at this size (memory + speed)
MAX_DISPLAY_SIDE = 1400   # maps are drawn at most at this size
DEFAULT_GRID_COLS = 256
DEFAULT_GRID_ROWS = 192
SQM_PER_ACRE = 4046.86

BRAND = "Build With Mayank"
BRAND_CONTACT = "letsbuildwithmayank@gmail.com"

STAGES = ["Reading the images", "Calibrating the bands", "Lining up the images",
          "Measuring the grid", "Checking problem areas", "Report ready"]

NAV_UPLOAD = "Upload & set up"
NAV_RESULTS = "Results"

# Palette
INK = "#1E2A22"
MUTED = "#5E6B62"
LINE = "#D6DDD3"
CROP = "#2F6B3F"
NODATA = "#B9C1BB"
SEVERITY = {
    3: ("Critical", "#B8392B"),
    2: ("Moderate", "#DB7F2A"),
    1: ("Low", "#E4C13A"),
}

BANDS = {
    "red": {
        "label": "Red band",
        "help": "Red band image from a multispectral camera (TIFF preferred). "
                "Needed together with NIR for NDVI.",
    },
    "green": {
        "label": "Green band",
        "help": "Green band image. Used for the GNDVI growth check. "
                "Without it, growth is checked from the RGB photo.",
    },
    "nir": {
        "label": "NIR band",
        "help": "Near-infrared band. Healthy plants reflect a lot of NIR light. "
                "Needed for every multispectral check.",
    },
    "rededge": {
        "label": "RedEdge band",
        "help": "Red-edge band. Needed for the nitrogen (NDRE) and crop stress (LCI) checks.",
    },
}
BAND_ORDER = ["red", "green", "nir", "rededge"]

INDEX_INFO = {
    "NDVI": "Normalized Difference Vegetation Index: overall green cover and plant vigour.",
    "NDRE": "Normalized Difference Red Edge: sensitive to leaf chlorophyll and nitrogen.",
    "LCI": "Leaf Chlorophyll Index: picks up early crop stress.",
    "GNDVI": "Green NDVI: canopy chlorophyll and growth.",
    "VARI": "Visible Atmospherically Resistant Index: greenness measured from a normal RGB photo.",
    "GLI": "Green Leaf Index: separates green plants from bare soil in a normal RGB photo.",
}
INDEX_RANGE = {
    "NDVI": (-0.2, 0.9), "NDRE": (-0.1, 0.6), "LCI": (-0.1, 0.6),
    "GNDVI": (-0.1, 0.8), "VARI": (-0.3, 0.5), "GLI": (-0.2, 0.4),
}
INDEX_ORDER = ["NDVI", "NDRE", "LCI", "GNDVI", "VARI", "GLI"]

ACTIONS = {
    "RESEEDING": {
        "label": "Reseeding",
        "desc": "Bare soil or failed germination. These patches may need replanting.",
        "todo": "Walk the marked patches, check seed and germination, and replant the gaps.",
        "sources": ["NDVI", "GLI"], "needs": "a photo of the field",
        "cost_label": "Reseeding", "rate": 5.0,
    },
    "WATER": {
        "label": "Water stress",
        "desc": "Low plant vigour that may point to moisture stress.",
        "todo": "Check soil moisture in the marked areas before irrigating, "
                "and look for blocked channels or uneven watering.",
        "sources": ["NDVI"], "needs": "Red + NIR bands",
        "cost_label": "Irrigation", "rate": 2.0,
    },
    "UREA": {
        "label": "Nitrogen need",
        "desc": "Low leaf chlorophyll that may indicate nitrogen deficiency.",
        "todo": "Check leaf colour or test the soil, then apply nitrogen "
                "as per local agriculture advice.",
        "sources": ["NDRE"], "needs": "Red + NIR + RedEdge bands",
        "cost_label": "Fertilizer", "rate": 3.0,
    },
    "STRESS": {
        "label": "Crop stress",
        "desc": "Early stress that may come from pests, disease or other causes.",
        "todo": "Scout the marked spots for pests and disease, and share photos with an expert.",
        "sources": ["LCI"], "needs": "Red + NIR + RedEdge bands",
        "cost_label": "Pest / disease treatment", "rate": 4.0,
    },
    "GROWTH": {
        "label": "Weak growth",
        "desc": "Below-average canopy growth. General crop care is needed.",
        "todo": "Check plant spacing, weeds and nutrition in the weak patches.",
        "sources": ["GNDVI", "VARI"], "needs": "a photo of the field",
        "cost_label": "General care", "rate": 2.0,
    },
}

# (critical, moderate, low): a cell is flagged when its index is BELOW the limit
DEFAULT_THRESHOLDS = {
    "RESEEDING:NDVI": (0.12, 0.18, 0.25),
    "WATER:NDVI": (0.30, 0.40, 0.50),
    "UREA:NDRE": (0.12, 0.18, 0.25),   # tuned for ~730 nm red-edge bands (DJI)
    "STRESS:LCI": (0.10, 0.20, 0.30),
    "GROWTH:GNDVI": (0.30, 0.40, 0.50),
    "RESEEDING:GLI": (0.00, 0.03, 0.06),
    "GROWTH:VARI": (0.02, 0.06, 0.10),
}
LEVEL_NAMES = ("critical", "moderate", "low")

HELP = {
    "flight_height": "Height of the drone above the ground, in metres, when the photos were taken. "
                     "Filled in from the photo when available. Drones record height above the "
                     "take-off point, so adjust it if the field is higher or lower than that spot.",
    "sensor_width": "Physical width of the camera sensor in millimetres, from the drone or camera "
                    "spec sheet (for the camera that took the analysed image). Filled in automatically "
                    "for DJI drones. Examples: about 6.17 mm for a 1/2.3-inch sensor, "
                    "about 13.2 mm for a 1-inch sensor.",
    "focal_length": "Real focal length of the lens in millimetres (not the 35 mm equivalent). "
                    "You can find it in the spec sheet or in the photo's EXIF details.",
    "image_width": "Width of the analysed image in pixels (the NIR band when bands are added, "
                   "otherwise the RGB photo). Filled in automatically. Change it only if the image "
                   "was resized after the flight.",
    "image_height": "Height of the analysed image in pixels. Filled in automatically.",
    "gsd": "Ground Sampling Distance: how much ground one pixel covers. Smaller means more detail.",
    "footprint": "Width and height of the ground area visible in one photo.",
    "total_area": "Total ground area covered by the photo.",
    "grid_cols": "The field is split into a grid, and every cell gets its own result. "
                 "More columns give finer detail with smaller cells.",
    "grid_rows": "Number of grid rows. For square cells, keep rows ÷ columns close to "
                 "the photo's height ÷ width.",
    "cell_area": "Ground area of one grid cell. Costs are worked out from this.",
    "notes": "Anything useful about the field, such as recent rain, soil type or pests seen. "
             "Added to the PDF report.",
    "labor": "Labour cost in ₹ per square metre, added to every treatment.",
    "overhead": "Extra percentage added on top for transport, equipment and other costs.",
    "cover_critical": "Share of Critical cells you plan to treat.",
    "cover_moderate": "Share of Moderate cells you plan to treat.",
    "cover_low": "Share of Low cells you plan to treat.",
}


# ─────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,400..800&display=swap');

:root {
  --gs-paper: #F2F5EF;
  --gs-surface: #FFFFFF;
  --gs-ink: #1E2A22;
  --gs-muted: #5E6B62;
  --gs-line: #D6DDD3;
  --gs-crop: #2F6B3F;
  --gs-soil: #8A5A3B;
  --gs-warn: #DB7F2A;
  --gs-bad: #B8392B;
}
.stApp :is(p, li, label, h1, h2, h3, h4, input, textarea, button, th, td) {
  font-family: 'Archivo', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp :is(h1, h2, h3) {
  font-stretch: 85%;
  letter-spacing: -0.01em;
  color: var(--gs-ink);
}
.block-container { padding-top: 2rem; max-width: 1280px; }

/* Sidebar brand */
.gs-brand { display: flex; gap: 12px; align-items: center; padding: 4px 0 8px; }
.gs-brand-mark {
  width: 44px; height: 44px; border-radius: 10px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 800; font-size: 1.1rem; letter-spacing: .02em;
  background: var(--gs-crop);
  background-image: repeating-linear-gradient(90deg, rgba(255,255,255,.14) 0 2px, transparent 2px 8px);
}
.gs-brand-name { font-weight: 800; font-size: 1.35rem; font-stretch: 85%; color: var(--gs-ink); line-height: 1.1; }
.gs-brand-sub { color: var(--gs-muted); font-size: .85rem; }
.gs-side-score { font-weight: 800; font-size: 2.4rem; font-stretch: 80%; line-height: 1; }
.gs-side-score small { font-size: .95rem; color: var(--gs-muted); font-weight: 600; }

/* Page intro */
.gs-hero { padding: 4px 0 18px; border-bottom: 2px solid var(--gs-ink); margin-bottom: 8px; }
.gs-hero h1 { font-size: clamp(1.9rem, 3.6vw, 2.8rem); font-weight: 800; line-height: 1.08; margin: 0; padding: 0; }
.gs-hero p { margin: 8px 0 0; color: var(--gs-muted); max-width: 64ch; font-size: 1.05rem; line-height: 1.5; }

/* Numbered steps (the upload page is a real sequence) */
.gs-step { display: flex; align-items: center; gap: 12px; margin: 30px 0 2px; }
.gs-step-n {
  width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--gs-crop); color: #fff; font-weight: 800; font-size: .95rem;
}
.gs-step h3 { margin: 0; padding: 0; font-size: 1.35rem; font-weight: 750; }
.gs-step-note { color: var(--gs-muted); margin: 2px 0 12px 42px; font-size: .95rem; max-width: 70ch; }

/* Mode box */
.gs-mode {
  background: var(--gs-surface); border: 1px solid var(--gs-line);
  border-left: 5px solid var(--gs-crop); border-radius: 0 10px 10px 0;
  padding: 12px 16px; margin: 14px 0 4px;
}
.gs-mode.rgb { border-left-color: var(--gs-warn); }
.gs-mode strong { display: block; font-size: 1.02rem; margin-bottom: 2px; color: var(--gs-ink); }
.gs-mode span { color: var(--gs-muted); font-size: .95rem; }

/* Report header: the field strip is the signature element */
.gs-report {
  display: grid; grid-template-columns: minmax(170px, 230px) 1fr; gap: 30px; align-items: center;
  background: var(--gs-surface); border: 1px solid var(--gs-line); border-radius: 14px;
  padding: 24px 28px; margin: 4px 0 16px;
}
@media (max-width: 760px) { .gs-report { grid-template-columns: 1fr; gap: 16px; } }
.gs-score { font-size: 4.4rem; font-weight: 800; font-stretch: 78%; line-height: .95; }
.gs-score small { font-size: 1.2rem; color: var(--gs-muted); font-weight: 600; }
.gs-score-label { font-weight: 700; font-size: 1.05rem; margin-top: 6px; }
.gs-score-note { color: var(--gs-muted); font-size: .88rem; }
.gs-report h2 { margin: 0; padding: 0; font-size: 1.6rem; font-weight: 800; }
.gs-report-meta { color: var(--gs-muted); margin: 2px 0 14px; font-size: .95rem; }
.gs-strip { display: flex; height: 38px; border-radius: 6px; overflow: hidden; background: var(--gs-line); }
.gs-strip span {
  display: block; height: 100%;
  background-image: repeating-linear-gradient(90deg, rgba(255,255,255,.16) 0 2px, transparent 2px 9px);
}
.gs-legend { display: flex; flex-wrap: wrap; gap: 6px 22px; margin-top: 10px; font-size: .93rem; color: var(--gs-ink); }
.gs-legend i { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }

/* Panels */
.gs-panel {
  background: var(--gs-surface); border: 1px solid var(--gs-line); border-radius: 10px;
  padding: 14px 16px; margin-bottom: 12px; line-height: 1.5;
}
.gs-panel h4 { margin: 0 0 4px; padding: 0; font-size: 1rem; }
.gs-panel p { margin: 0; color: var(--gs-muted); font-size: .94rem; }
.gs-total {
  background: var(--gs-crop); color: #fff; border-radius: 12px; padding: 16px 20px; margin: 12px 0;
  background-image: repeating-linear-gradient(90deg, rgba(255,255,255,.08) 0 2px, transparent 2px 10px);
}
.gs-total div { font-size: .95rem; opacity: .9; }
.gs-total strong { display: block; font-size: 2.3rem; font-weight: 800; font-stretch: 82%; line-height: 1.1; }
.gs-notes { white-space: pre-wrap; }

/* Progress checklist in the sidebar */
.gs-check { margin: 2px 0 0; line-height: 1.85; font-size: .93rem; }
.gs-check div { color: var(--gs-muted); }
.gs-check div.done { color: var(--gs-ink); }
.gs-check div.now { color: var(--gs-crop); font-weight: 700; }
.gs-check i { display: inline-block; width: 18px; font-style: normal; }
.gs-check div.done i { color: var(--gs-crop); font-weight: 700; }
.gs-check b { font-weight: 700; }

/* Problem spots */
.gs-spot { border-left: 3px solid var(--gs-line); padding: 2px 0 2px 10px; margin: 6px 0; font-size: .93rem; }
.gs-spot b { color: var(--gs-ink); }
.gs-spot span { color: var(--gs-muted); }

/* Watermark / footer */
.gs-footer {
  margin: 44px 0 8px; padding-top: 14px; border-top: 1px solid var(--gs-line);
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px;
  font-size: .85rem; color: var(--gs-muted);
}
.gs-footer .mark { display: flex; align-items: center; gap: 8px; color: var(--gs-ink); font-weight: 700; }
.gs-footer .mark i {
  width: 16px; height: 16px; border-radius: 4px; background: var(--gs-crop);
  background-image: repeating-linear-gradient(90deg, rgba(255,255,255,.2) 0 1px, transparent 1px 5px);
}
.gs-side-footer { margin-top: 10px; font-size: .8rem; color: var(--gs-muted); line-height: 1.5; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STATE HELPERS
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "nav": NAV_UPLOAD,
        "result": None,      # analysis output
        "meta": None,        # camera / grid / notes used for the result
        "kept_files": None,  # {"label", "rgb", "bands"} when using sample or previous files
        "cost": None,
        "pdf": None,         # (key, bytes)
        "_keep": {},         # widget values that survive page switches
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def kept_widget(widget, label, key, default, **kwargs):
    """Render a widget whose value survives switching pages."""
    if key not in st.session_state:
        st.session_state[key] = st.session_state["_keep"].get(key, default)
    value = widget(label, key=key, **kwargs)
    st.session_state["_keep"][key] = value
    return value


def set_widget_value(key, value):
    st.session_state[key] = value
    st.session_state["_keep"][key] = value


def go_to(page):
    st.session_state.nav = page


def reset_thresholds():
    for combo, values in DEFAULT_THRESHOLDS.items():
        action, index = combo.split(":")
        for level, value in zip(LEVEL_NAMES, values):
            set_widget_value(f"thr_{action}_{index}_{level}", value)


def use_sample_files():
    samples = find_sample_files()
    if "rgb" not in samples:
        return
    st.session_state.kept_files = {
        "label": "the sample field",
        "rgb": read_path(samples["rgb"]),
        "bands": {k: read_path(p) for k, p in samples.items() if k != "rgb"},
    }


def clear_kept_files():
    st.session_state.kept_files = None


# ─────────────────────────────────────────────────────────────
# FILE HELPERS
# ─────────────────────────────────────────────────────────────
def find_sample_files():
    """Look for sample_data/rgb.*, red.*, green.*, nir.*, rededge.*"""
    found = {}
    if not SAMPLE_DIR.is_dir():
        return found
    files = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for key in ["rgb"] + BAND_ORDER:
        for path in files:
            if path.stem.lower() == key:
                found[key] = path
                break
    return found


@st.cache_data(show_spinner=False, max_entries=10)
def _read_bytes(path_str, mtime):
    return Path(path_str).read_bytes()


def read_path(path):
    return path.name, _read_bytes(str(path), path.stat().st_mtime)


# ─────────────────────────────────────────────────────────────
# CAMERA METADATA (DJI drones store it as XMP text inside the file)
# ─────────────────────────────────────────────────────────────
def read_xmp(data):
    start, end = data.find(b"<x:xmpmeta"), data.find(b"</x:xmpmeta>")
    if start < 0 or end < 0:
        return {}
    text = data[start:end].decode("utf-8", "ignore")
    meta = {}
    for m in re.finditer(r'(?:drone-dji|dji|Camera):(\w+)="([^"]*)"', text):
        meta.setdefault(m.group(1), m.group(2).strip())
    for m in re.finditer(r"<(?:drone-dji|dji|Camera):(\w+)>([^<]*)<", text):
        meta.setdefault(m.group(1), m.group(2).strip())
    return meta


def _num(meta, key):
    try:
        return float(meta[key])
    except (KeyError, TypeError, ValueError):
        return None


def _floats(text):
    try:
        return [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    except ValueError:
        return None


def _exif_gps(data):
    """GPS from standard EXIF tags, for cameras that don't write DJI metadata."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            gps = im.getexif().get_ifd(0x8825)
        if not gps:
            return None, None

        def dms(values, ref, negative):
            deg = float(values[0]) + float(values[1]) / 60 + float(values[2]) / 3600
            return -deg if str(ref).upper() == negative else deg
        return dms(gps[2], gps.get(1, "N"), "S"), dms(gps[4], gps.get(3, "E"), "W")
    except Exception:
        return None, None


METRES_PER_DEGREE = 111320.0


def cell_latlon(geo, fx, fy):
    """Latitude/longitude of a point given as fractions across and down the analysed area."""
    # metres right and down from the middle of the photo
    u = (fx - 0.5) * geo["width_m"] + geo["offset_u"]
    v = (fy - 0.5) * geo["height_m"] + geo["offset_v"]
    yaw = np.radians(geo["yaw"])
    north = -u * np.sin(yaw) - v * np.cos(yaw)
    east = u * np.cos(yaw) - v * np.sin(yaw)
    lat = geo["lat"] + north / METRES_PER_DEGREE
    lon = geo["lon"] + east / (METRES_PER_DEGREE * max(0.05, np.cos(np.radians(geo["lat"]))))
    return round(float(lat), 7), round(float(lon), 7)


def maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"


@st.cache_data(show_spinner=False, max_entries=20)
def camera_info(name, data):
    """Image size plus any camera details stored in the file."""
    info = {"width": None, "height": None, "model": "", "flight_height": None,
            "sensor_width": None, "focal_length": None, "lat": None, "lon": None, "yaw": None}
    meta = read_xmp(data)
    f_mm = plane_res = plane_unit = None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            info["width"], info["height"] = im.size
            exif = im.getexif()
            info["model"] = str(exif.get(0x0110) or "")
            sub = exif.get_ifd(0x8769)
            f_mm, plane_res, plane_unit = sub.get(0x920A), sub.get(0xA20E), sub.get(0xA210)
    except Exception:
        pass
    if info["width"] is None:
        img = decode_rgb(name, data)
        if img is not None:
            info["height"], info["width"] = img.shape[:2]
    info["model"] = info["model"] or meta.get("DroneModel", "")

    height = _num(meta, "RelativeAltitude")
    if height and height > 0:
        info["flight_height"] = round(height, 1)
    try:
        f_mm = float(f_mm) if f_mm else None
    except (TypeError, ValueError):
        f_mm = None
    lat, lon = _num(meta, "GpsLatitude"), _num(meta, "GpsLongitude")
    if lat is None or lon is None:
        lat, lon = _exif_gps(data)
    if lat is not None and lon is not None and abs(lat) <= 90 and abs(lon) <= 180:
        info["lat"], info["lon"] = lat, lon
    yaw = _num(meta, "GimbalYawDegree")
    roll = _num(meta, "GimbalRollDegree")
    if yaw is None:
        yaw = _num(meta, "FlightYawDegree")
    elif roll is not None and abs(roll) > 90:
        yaw += 180          # DJI mounts the multispectral cameras upside down
    if yaw is not None:
        info["yaw"] = yaw % 360

    f_px = _num(meta, "CalibratedFocalLength")
    width = info["width"]
    if f_mm and width:
        if f_px:
            info["sensor_width"] = round(f_mm / f_px * width, 3)
        elif plane_res and plane_unit:
            unit_mm = {2: 25.4, 3: 10.0, 4: 1.0}.get(int(plane_unit))
            if unit_mm and float(plane_res) > 0:
                info["sensor_width"] = round(width / float(plane_res) * unit_mm, 3)
        if info["sensor_width"]:
            info["focal_length"] = round(f_mm, 2)
    return info


# ─────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────
def _to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return (arr.astype(np.float32) / np.iinfo(arr.dtype).max * 255).astype(np.uint8)
    a = arr.astype(np.float32)
    top = np.nanmax(a) if np.isfinite(a).any() else 1.0
    a = a / (1.0 if top <= 1.5 else top)
    return (np.nan_to_num(np.clip(a, 0, 1)) * 255).astype(np.uint8)


def decode_rgb(name, data, reduce=1):
    """Decode the RGB photo as an 8-bit BGR image (optionally at 1/2 or 1/4 size)."""
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}[reduce]
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if img is None and name.lower().endswith((".tif", ".tiff")):
        try:
            import tifffile
            arr = np.squeeze(tifffile.imread(io.BytesIO(data)))
            if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
                arr = np.moveaxis(arr, 0, -1)          # channels-first -> channels-last
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            img = cv2.cvtColor(_to_uint8(arr[..., :3]), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    return img


def _scale_band(arr):
    """Scale a band to 0-1 by its data type, so every band keeps the same scale."""
    if np.issubdtype(arr.dtype, np.integer):
        bits = np.iinfo(arr.dtype).bits
        return arr.astype(np.float32) / np.iinfo(arr.dtype).max, f"{bits}-bit, not calibrated"
    a = arr.astype(np.float32)
    finite = a[np.isfinite(a)]
    top = float(finite.max()) if finite.size else 0.0
    if top <= 1.5:
        return a, "reflectance (0-1)"
    if top <= 10000:
        return a / 10000.0, "reflectance x10000"
    if top <= 65535:
        return a / 65535.0, "16-bit float"
    return a / top, "float (own max)"


def _black_level(meta, tags, model):
    value = _num(meta, "BlackLevel")
    if value is not None:
        return value
    tag = tags.get(50714)                      # DNG BlackLevel tag
    if tag is not None:
        try:
            v = tag[0] if isinstance(tag, (tuple, list)) else tag
            return float(v)
        except (TypeError, ValueError):
            pass
    return 3200.0 if "M3M" in model else 0.0   # DJI Mavic 3 Multispectral


def calibrate_dji_band(arr, meta, tags, model):
    """Raw DJI multispectral numbers -> reflectance-like values.

    Steps from DJI's image processing guide: remove the black level,
    correct lens vignetting, divide by gain and exposure time, and
    correct for sunlight with the sunlight sensor (irradiance).
    Without this, indices such as NDVI come out far too low.
    """
    keys = ("SensorGain", "ExposureTime", "SensorGainAdjustment", "Irradiance")
    vals = {k: _num(meta, k) for k in keys}
    if arr.dtype != np.uint16 or any(v is None or v <= 0 for v in vals.values()):
        return None
    x = (arr.astype(np.float32) - _black_level(meta, tags, model)) / 65535.0
    np.maximum(x, 0, out=x)

    vig = _floats(meta.get("VignettingData", ""))
    if vig and len(vig) >= 6 and meta.get("VignettingFlag") != "1":
        h, w = x.shape
        cx = _num(meta, "CalibratedOpticalCenterX") or w / 2
        cy = _num(meta, "CalibratedOpticalCenterY") or h / 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        poly, rp = np.ones_like(r), np.ones_like(r)
        for k in vig[:6]:
            rp *= r
            poly += np.float32(k) * rp
        x *= np.clip(poly, 0.5, 6.0)

    x *= vals["SensorGainAdjustment"] / (vals["SensorGain"] * vals["ExposureTime"] / 1e6 * vals["Irradiance"])
    return x


def _undistort_params(meta, w, h):
    """Lens distortion data from DJI metadata, or None."""
    data = meta.get("DewarpData", "")
    if meta.get("DewarpFlag") == "1" or ";" not in data:
        return None
    vals = _floats(data.split(";", 1)[1])
    if not vals or len(vals) < 9 or not any(abs(v) > 1e-9 for v in vals[4:9]):
        return None
    fx, fy, cxo, cyo, k1, k2, p1, p2, k3 = vals[:9]
    cx = (_num(meta, "CalibratedOpticalCenterX") or w / 2) + cxo
    cy = (_num(meta, "CalibratedOpticalCenterY") or h / 2) + cyo
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, np.array([k1, k2, p1, p2, k3], dtype=np.float64)


def load_band(name, data):
    """Returns (values 0-1 float32, format label, metadata)."""
    arr, tags, model = None, {}, ""
    if name.lower().endswith((".tif", ".tiff")):
        try:
            import tifffile
            with tifffile.TiffFile(io.BytesIO(data)) as tf:
                page = tf.pages[0]
                arr = page.asarray()
                for tag in page.tags.values():
                    if tag.code in (272, 50714):
                        tags[tag.code] = tag.value
            model = str(tags.get(272, ""))
        except Exception:
            arr = None
    if arr is None:
        # IMREAD_ANYDEPTH keeps 16-bit data (IMREAD_GRAYSCALE would cut it to 8-bit)
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise ValueError(f"Could not read '{name}'. Please use a JPG, PNG or TIFF file.")
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"'{name}' should be a single-band image.")

    meta = read_xmp(data)
    model = model or meta.get("DroneModel", "")
    und = _undistort_params(meta, arr.shape[1], arr.shape[0])
    calibrated = calibrate_dji_band(arr, meta, tags, model)
    if calibrated is not None:
        values, fmt = calibrated, "DJI calibrated reflectance"
    else:
        values, fmt = _scale_band(arr)
    if und is not None:
        values = cv2.undistort(values, und[0], und[1])
    return values.astype(np.float32), fmt, meta


# ─────────────────────────────────────────────────────────────
# ALIGNMENT (each band comes from a different lens, so they never line up exactly)
# ─────────────────────────────────────────────────────────────
IDENTITY = np.float32([[1, 0, 0], [0, 1, 0]])


def _grad(img):
    g = cv2.GaussianBlur(np.nan_to_num(img).astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _u8(img):
    a = np.nan_to_num(img).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def _inner(shape, margin=0.06):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[int(h * margin): h - int(h * margin), int(w * margin): w - int(w * margin)] = True
    return mask


def _ncc(a, b, mask):
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else -1.0


def _warp(img, W, size):
    """W maps output pixels to input pixels."""
    return cv2.warpAffine(img, W, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))


def _plausible(W, size, max_shift=0.15):
    w, h = size
    lin = W[:, :2]
    return (0.85 < abs(np.linalg.det(lin)) < 1.15
            and abs(W[0, 2]) < w * max_shift and abs(W[1, 2]) < h * max_shift)


def _ecc(ref_grad, mov_grad, init):
    """Coarse-to-fine ECC on gradient images (works even when brightness is inverted)."""
    W, ok = init.astype(np.float32).copy(), False
    crit = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    for scale, iters in ((0.125, 60), (0.25, 40), (0.5, 25)):
        r = cv2.resize(ref_grad, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mov_grad, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        Ws = W.copy()
        Ws[:, 2] *= scale
        try:
            _, Ws = cv2.findTransformECC(r, m, Ws, cv2.MOTION_AFFINE, (crit, iters, 1e-4), None, 3)
        except cv2.error:
            continue
        Ws[:, 2] /= scale
        W, ok = Ws, True
    return W if ok else None


def _sift_matches(ref_u8, mov_u8, scale):
    sift = cv2.SIFT_create(4000)
    a = cv2.resize(ref_u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(mov_u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None, None
    pairs = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 30:
        return None, None
    src = np.float32([kb[g.queryIdx].pt for g in good]) / scale
    dst = np.float32([ka[g.trainIdx].pt for g in good]) / scale
    return src, dst


def _sift_affine(ref, mov, scale=0.5):
    src, dst = _sift_matches(_u8(ref), _u8(mov), scale)
    if src is None:
        return None
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None or inl.sum() < 30:
        return None
    return cv2.invertAffineTransform(A).astype(np.float32)


def align_bands(bands, offsets, info):
    """Line up Red, Green and RedEdge with the NIR band."""
    nir = bands["nir"]
    h, w = nir.shape
    size, mask = (w, h), _inner(nir.shape)
    g_nir = _grad(nir)
    out = {"nir": nir}
    for key in ("rededge", "green", "red"):
        if key not in bands:
            continue
        mov = bands[key]
        # red looks most like green, the others are checked against NIR
        use_green = key == "red" and "green" in out
        ref_grad = _grad(out["green"]) if use_green else g_nir
        init = offsets.get(key, IDENTITY)
        candidates = [("camera data" if key in offsets else "no shift", init)]
        W = _sift_affine(out["green"], mov) if use_green else None
        if W is not None:
            candidates.append(("feature matching", W))
        else:
            W = _ecc(ref_grad, _grad(mov), init)
            if W is not None:
                candidates.append(("image matching", W))

        base_name, base_W = candidates[0]
        best_name, best_img = base_name, _warp(mov, base_W, size)
        best_score = _ncc(ref_grad, _grad(best_img), mask)
        for name, W in candidates[1:]:
            if not _plausible(W, size):
                continue
            img = _warp(mov, W, size)
            score = _ncc(ref_grad, _grad(img), mask)
            if score > best_score + 0.02:        # a refinement must clearly help
                best_name, best_img, best_score = name, img, score
        out[key] = best_img
        info.append(f"{BANDS[key]['label']} lined up using {best_name}.")
    return out


def align_rgb(rgb_data, rgb_meta, rgb_size, band_meta, frame_full, p, ref_img, ref_key, info, notes):
    """Warp the RGB photo into the band frame. Returns an RGB (not BGR) image."""
    fw, fh = frame_full
    w, h = int(round(fw * p)), int(round(fh * p))
    rw, rh = rgb_size
    hm = _floats(band_meta.get("DewarpHMatrix", ""))
    H = np.array(hm, dtype=np.float64).reshape(3, 3) if hm and len(hm) == 9 else None

    target_q = p / abs(H[0, 0]) if H is not None else min(1.0, 1.3 * max(w / rw, h / rh))
    reduce = 4 if target_q <= 0.25 else (2 if target_q <= 0.5 else 1)
    img = decode_rgb(rgb_data[0], rgb_data[1], reduce)
    if img is None:
        raise ValueError(f"Could not read the RGB photo '{rgb_data[0]}'.")
    q0 = img.shape[1] / rw
    if target_q < q0 * 0.95:
        img = cv2.resize(img, (int(round(rw * target_q)), int(round(rh * target_q))),
                         interpolation=cv2.INTER_AREA)
    q = img.shape[1] / rw

    und = _undistort_params(rgb_meta, rw, rh)
    if und is not None:
        K = und[0].copy()
        K[:2] *= q
        img = cv2.undistort(img, K, und[1])

    channel = 2 if ref_key == "red" else 1          # BGR: red=2, green=1
    mask = _inner((h, w))
    ref_grad = _grad(ref_img)
    base, base_score = None, -2.0
    if H is not None:
        Hp = np.diag([q, q, 1.0]) @ H @ np.diag([1 / p, 1 / p, 1.0])
        base = cv2.warpPerspective(img, Hp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        base_score = _ncc(ref_grad, _grad(base[..., channel].astype(np.float32)), mask)
    source = base if base is not None else img

    result, how = base, "camera data"
    src, dst = _sift_matches(_u8(ref_img), _u8(source[..., channel]), 0.6)
    if src is not None:
        Hr, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hr is not None and inl.sum() >= 30 and 0.2 < abs(np.linalg.det(Hr[:2, :2])) < 5:
            cand = cv2.warpPerspective(source, Hr, (w, h), flags=cv2.INTER_LINEAR)
            score = _ncc(ref_grad, _grad(cand[..., channel].astype(np.float32)), mask)
            if score > base_score + 0.02:
                result, how = cand, ("camera data + feature matching" if base is not None
                                     else "feature matching")
    if result is None:
        result = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        notes.append("The RGB photo could not be lined up with the bands automatically, "
                     "so the maps may look shifted. The analysis itself uses only the bands.")
        how = "simple resizing"
    info.append(f"RGB photo lined up with the bands using {how}.")
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────
# MATHS
# ─────────────────────────────────────────────────────────────
def ground_footprint(flight_h, sensor_w, focal_len, img_w, img_h):
    """Returns (GSD m/px, ground width m, ground height m). Assumes square pixels."""
    gsd = (sensor_w / 1000.0 * flight_h) / (focal_len / 1000.0 * img_w)
    return gsd, gsd * img_w, gsd * img_h


def safe_ratio(num, den):
    """num / den, with NaN where the denominator is ~0 (e.g. black no-data pixels)."""
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-6)
    return out


def block_mean(arr, rows, cols):
    """Average an image into a rows x cols grid covering the WHOLE image.

    NaN pixels are ignored. A cell counts as "no data" when less than half
    of its pixels are usable.
    """
    mask = np.isfinite(arr)
    values = np.where(mask, arr, 0).astype(np.float32)
    weight = cv2.resize(mask.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    total = cv2.resize(values, (cols, rows), interpolation=cv2.INTER_AREA)
    out = np.full((rows, cols), np.nan, dtype=np.float32)
    ok = weight >= 0.5
    out[ok] = total[ok] / weight[ok]
    return out


def level_map(values, limits):
    critical, moderate, low = limits
    lv = np.zeros(values.shape, dtype=np.int8)
    lv[values < low] = 1
    lv[values < moderate] = 2
    lv[values < critical] = 3
    return lv  # NaN compares as False, so no-data cells stay 0


RELATIVE_MIN_SPREAD = 0.04      # a field this uniform has nothing to compare


def relative_limits(values, valid, percents):
    """Cutoffs taken from this field's own numbers, or None if the field is uniform."""
    vals = values[valid & np.isfinite(values)]
    if vals.size < 50:
        return None
    if float(np.percentile(vals, 95) - np.percentile(vals, 5)) < RELATIVE_MIN_SPREAD:
        return None
    pc, pm, pl = percents
    cuts = np.percentile(vals, [pc, pc + pm, pc + pm + pl])
    return tuple(round(float(c), 4) for c in cuts)


def plot_labels(prow, pcol):
    return [f"{chr(65 + r)}{c + 1}" for r in range(prow) for c in range(pcol)]


def plot_ids(rows, cols, prow, pcol):
    pr = (np.arange(rows) * prow) // rows
    pc = (np.arange(cols) * pcol) // cols
    return (pr[:, None] * pcol + pc[None, :]).astype(np.int32)


def plot_stats(cells, flags, valid, healthy, worst, rows, cols, prow, pcol):
    """Per-plot summary: score, healthy share and how much each check flagged."""
    n = prow * pcol
    idx = plot_ids(rows, cols, prow, pcol).ravel()
    vmask = valid.ravel()
    valid_n = np.bincount(idx[vmask], minlength=n)
    healthy_n = np.bincount(idx[healthy.ravel()], minlength=n)
    worst_sum = np.bincount(idx[vmask], weights=worst.ravel()[vmask].astype(float), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(valid_n > 0, 100 - worst_sum / np.maximum(valid_n, 1) / 3 * 100, np.nan)
    action_total, action_critical = {}, {}
    for action, lv in flags.items():
        flat = lv.ravel()
        action_total[action] = np.bincount(idx[flat > 0], minlength=n)
        action_critical[action] = np.bincount(idx[flat == 3], minlength=n)
    index_mean = {}
    for name, arr in cells.items():
        flat = arr.ravel()
        ok = np.isfinite(flat) & vmask
        total = np.bincount(idx[ok], weights=flat[ok].astype(float), minlength=n)
        counted = np.bincount(idx[ok], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            index_mean[name] = np.where(counted > 0, total / np.maximum(counted, 1), np.nan)
    return {
        "rows": prow, "cols": pcol, "labels": plot_labels(prow, pcol),
        "cells": np.bincount(idx, minlength=n), "valid": valid_n, "healthy": healthy_n,
        "score": score, "action_total": action_total, "action_critical": action_critical,
        "index_mean": index_mean,
    }


def available_indices(band_keys):
    keys = set(band_keys)
    if {"red", "nir"} <= keys:
        indices = {"NDVI"}
        if "rededge" in keys:
            indices |= {"NDRE", "LCI"}
        indices.add("GNDVI" if "green" in keys else "VARI")
        return "multispectral", indices
    return "rgb", {"GLI", "VARI"}


def action_sources(indices):
    return {a: next((s for s in m["sources"] if s in indices), None) for a, m in ACTIONS.items()}


def score_label(score):
    if score >= 70:
        return "Good condition", CROP
    if score >= 40:
        return "Needs attention", SEVERITY[2][1]
    return "Serious problems", SEVERITY[3][1]


def problem_spots(lv, meta, cols, rows, limit=5, merge_m=2.0, split_m=12.0):
    """Areas that need attention, as a short list of places to visit.

    Patches whose edges are within `merge_m` metres of each other are one
    problem, so they get one pin. A long strip (a whole bund or road edge)
    is split into pieces of about `split_m` metres so the pins stay useful.
    """
    mask = (lv == 3).astype(np.uint8)
    if not mask.any():
        return []
    ground_w, ground_h = meta.get("ground_size_m", (0, 0))
    cell_area = meta["cell_area"]
    cell_w = (ground_w / cols) if ground_w else 1.0
    cell_h = (ground_h / rows) if ground_h else 1.0

    # dilating by half the merge distance makes patches that close touch
    rx = int(np.clip(round(merge_m / 2 / max(cell_w, 1e-6)), 1, 30))
    ry = int(np.clip(round(merge_m / 2 / max(cell_h, 1e-6)), 1, 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(mask, kernel), connectivity=8)

    min_area = max(0.3, 0.0005 * cell_area * rows * cols)
    spots = []
    for i in range(1, count):
        ys, xs = np.nonzero((labels == i) & (mask > 0))
        if ys.size == 0:
            continue
        gx, gy = xs * cell_w, ys * cell_h            # positions in metres
        spread = max(gx.max() - gx.min() + cell_w, gy.max() - gy.min() + cell_h)
        pieces = [(xs, ys)]
        if spread > split_m:
            pieces = _split_along_length(xs, ys, gx, gy, spread, split_m)
        for px, py in pieces:
            area = px.size * cell_area
            if area < min_area:
                continue
            fx, fy = float(px.mean()) / cols, float(py.mean()) / rows
            spot = {"area": area, "where": where_in_field(fx, fy),
                    "spread_m": round(max((px.max() - px.min() + 1) * cell_w,
                                          (py.max() - py.min() + 1) * cell_h), 1),
                    "x_m": round(fx * ground_w, 1), "y_m": round(fy * ground_h, 1)}
            if len(pieces) > 1:
                spot["strip_m"] = round(spread, 1)
            geo = meta.get("geo")
            if geo:
                spot["lat"], spot["lon"] = cell_latlon(geo, fx, fy)
            spots.append(spot)
    spots.sort(key=lambda sp: sp["area"], reverse=True)
    return spots[:limit]


def _split_along_length(xs, ys, gx, gy, spread, split_m):
    """Cut a long strip into pieces of about split_m metres along its length."""
    points = np.stack([gx - gx.mean(), gy - gy.mean()], axis=1)
    # main direction of the strip
    _, _, vectors = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    along = points @ vectors[0]
    parts = int(np.clip(round(spread / split_m), 2, 4))
    edges = np.linspace(along.min(), along.max() + 1e-6, parts + 1)
    pieces = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (along >= a) & (along < b)
        if sel.any():
            pieces.append((xs[sel], ys[sel]))
    return pieces or [(xs, ys)]


def where_in_field(fx, fy):
    across = "left" if fx < 0.34 else ("middle" if fx < 0.67 else "right")
    down = "top" if fy < 0.34 else ("middle" if fy < 0.67 else "bottom")
    return "centre of the field" if across == down == "middle" else f"{down}-{across}"


def spread_text(spot):
    if spot.get("strip_m"):
        return f"part of a {spot['strip_m']:,.0f} m long strip"
    spread = spot.get("spread_m", 0)
    if spread >= 8:
        return f"spread over about {spread:,.0f} m"
    return "in one patch"


def plot_rows(res, meta):
    """Per-plot rows, worst plot first."""
    plots = res["plots"]
    if not plots:
        return []
    primary = "NDVI" if res["mode"] == "multispectral" else "GLI"
    cell_area = meta["cell_area"]
    rows_out = []
    for i, label in enumerate(plots["labels"]):
        valid_n = int(plots["valid"][i])
        if valid_n == 0:
            continue
        needs = {a: int(plots["action_total"][a][i]) for a in plots["action_total"]}
        crit = {a: int(plots["action_critical"][a][i]) for a in plots["action_critical"]}
        # rank by how serious it is, not just by how much area it covers
        weighted = {a: 3 * crit[a] + (needs[a] - crit[a]) for a in needs}
        worst_action = max(weighted, key=weighted.get) if weighted else None
        if worst_action and weighted[worst_action] == 0:
            worst_action = None
        main = "-"
        if worst_action:
            main = ACTIONS[worst_action]["label"] + ("" if crit[worst_action] else " (mild)")
        row = {
            "Plot": label,
            "Health score": None if not np.isfinite(plots["score"][i]) else round(float(plots["score"][i]), 1),
            "Area (m²)": round(valid_n * cell_area, 1),
            "Healthy (%)": round(int(plots["healthy"][i]) / valid_n * 100, 1),
            "Needs action (m²)": round((valid_n - int(plots["healthy"][i])) * cell_area, 1),
            "Critical (m²)": round(max(crit.values(), default=0) * cell_area, 1),
            f"Average {primary}": None if primary not in plots["index_mean"] or not np.isfinite(
                plots["index_mean"][primary][i]) else round(float(plots["index_mean"][primary][i]), 3),
            "Main problem": main,
        }
        rows_out.append(row)
    rows_out.sort(key=lambda r: (r["Health score"] if r["Health score"] is not None else 101))
    return rows_out


def priority_of(stat, valid):
    if valid and stat["critical"] / valid > 0.10:
        return "High"
    if valid and stat["total"] / valid > 0.05:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=3)
def run_analysis(rgb_file, band_files, rows, cols, thresholds, crop=(0.0, 0.0, 0.0, 0.0),
                 limit_mode="fixed", percents=(5, 10, 15), plots=None, bare_limit=None,
                 _progress=None):
    """
    rgb_file:   (name, bytes)
    band_files: tuple of (band_key, name, bytes)
    thresholds: tuple of ("ACTION:INDEX", (critical, moderate, low))
    crop:       (left, right, top, bottom) as fractions of the image to leave out
    """
    def tick(stage):
        if _progress is not None:
            _progress(stage)

    tick(0)
    notes = []
    digest = hashlib.sha1()
    digest.update(rgb_file[1])
    for key, _, data in band_files:
        digest.update(key.encode())
        digest.update(data)
    digest.update(repr((rows, cols, thresholds, crop, limit_mode, percents, plots,
                        bare_limit)).encode())

    info = []
    rgb_cam = camera_info(*rgb_file)
    if not rgb_cam["width"]:
        raise ValueError(f"Could not read the RGB photo '{rgb_file[0]}'. Please use JPG, PNG or TIFF.")
    mode, indices = available_indices(k for k, _, _ in band_files)
    if mode == "rgb" and band_files:
        notes.append("Red and NIR bands are both needed for multispectral analysis, "
                     "so the uploaded bands were not used.")

    if mode == "multispectral":
        # 1. Bands: calibrate, bring to one size, line up with NIR
        loaded, formats = {}, {}
        for key, name, data in band_files:
            values, fmt, meta = load_band(name, data)
            loaded[key] = (values, meta)
            formats[BANDS[key]["label"]] = fmt
        if len(set(formats.values())) > 1:
            detail = ", ".join(f"{k}: {v}" for k, v in formats.items())
            notes.append(f"The bands are in different formats ({detail}). "
                         "For reliable results, export all bands in the same format.")
        tick(1)
        fmt = next(iter(formats.values()))
        if "not calibrated" in fmt:
            notes.append("The bands have no calibration data, so index values may be lower or "
                         "higher than real ones. Calibrated reflectance maps give the most reliable results.")
        else:
            info.append(f"Band values: {fmt}.")

        nir_vals, nir_meta = loaded["nir"]
        fh_full, fw_full = nir_vals.shape
        p = min(1.0, MAX_PROC_SIDE / max(fh_full, fw_full))
        pw = max(int(round(fw_full * p)), cols)
        ph = max(int(round(fh_full * p)), rows)
        p = pw / fw_full
        bands, offsets = {}, {}
        for key, (values, meta) in loaded.items():
            if values.shape != (fh_full, fw_full):
                values = cv2.resize(values, (fw_full, fh_full), interpolation=cv2.INTER_LINEAR)
            bands[key] = cv2.resize(values, (pw, ph), interpolation=cv2.INTER_AREA if p < 1 else cv2.INTER_LINEAR)
            dx, dy = _num(meta, "RelativeOpticalCenterX"), _num(meta, "RelativeOpticalCenterY")
            if key != "nir" and dx is not None and dy is not None:
                offsets[key] = np.float32([[1, 0, dx * p], [0, 1, dy * p]])
        del loaded
        tick(2)
        bands = align_bands(bands, offsets, info)

        # 2. RGB photo, lined up with the bands for the maps
        ref_key = "green" if "green" in bands else "red"
        rgb = align_rgb(rgb_file, read_xmp(rgb_file[1]), (rgb_cam["width"], rgb_cam["height"]),
                        nir_meta, (fw_full, fh_full), p, bands[ref_key], ref_key, info, notes)
        orig_w, orig_h = fw_full, fh_full
    else:
        bands = {}
        orig_w, orig_h = rgb_cam["width"], rgb_cam["height"]
        scale = min(1.0, MAX_PROC_SIDE / max(orig_h, orig_w))
        reduce = 4 if scale <= 0.25 else (2 if scale <= 0.5 else 1)
        rgb_bgr = decode_rgb(rgb_file[0], rgb_file[1], reduce)
        if rgb_bgr is None:
            raise ValueError(f"Could not read the RGB photo '{rgb_file[0]}'. Please use JPG, PNG or TIFF.")
        pw = max(int(round(orig_w * scale)), cols)
        ph = max(int(round(orig_h * scale)), rows)
        if rgb_bgr.shape[:2] != (ph, pw):
            interp = cv2.INTER_AREA if pw < rgb_bgr.shape[1] else cv2.INTER_LINEAR
            rgb_bgr = cv2.resize(rgb_bgr, (pw, ph), interpolation=interp)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        del rgb_bgr

    # 3. Keep only the chosen part of the image
    if any(c > 0 for c in crop):
        x0, x1 = int(round(pw * crop[0])), pw - int(round(pw * crop[1]))
        y0, y1 = int(round(ph * crop[2])), ph - int(round(ph * crop[3]))
        if x1 - x0 < cols or y1 - y0 < rows:
            raise ValueError("The chosen analysis area is too small for this grid. "
                             "Trim less of the image or use fewer grid cells.")
        rgb = rgb[y0:y1, x0:x1]
        bands = {k: v[y0:y1, x0:x1] for k, v in bands.items()}
        ph, pw = rgb.shape[:2]
        info.append(f"Analysis area: middle {100 * (1 - crop[0] - crop[1]):.0f}% across "
                    f"and {100 * (1 - crop[2] - crop[3]):.0f}% down the image.")

    # 4. Pixel-level indices
    tick(3)
    pix = {}
    if mode == "multispectral":
        r, n = bands["red"], bands["nir"]
        pix["NDVI"] = safe_ratio(n - r, n + r)
        if "NDRE" in indices:
            e = bands["rededge"]
            pix["NDRE"] = safe_ratio(n - e, n + e)
            pix["LCI"] = safe_ratio(n - e, n + r)
        if "GNDVI" in indices:
            g = bands["green"]
            pix["GNDVI"] = safe_ratio(n - g, n + g)
    del bands
    if "VARI" in indices or "GLI" in indices:
        f = rgb.astype(np.float32) / 255.0
        R, G, B = f[..., 0], f[..., 1], f[..., 2]
        if "VARI" in indices:
            pix["VARI"] = np.clip(safe_ratio(G - R, G + R - B), -1.0, 1.0)
        if "GLI" in indices:
            pix["GLI"] = safe_ratio(2 * G - R - B, 2 * G + R + B)
        del f, R, G, B

    # 4. Grid averages
    cells = {}
    for name in list(pix):
        cells[name] = block_mean(pix.pop(name), rows, cols)

    # 5. Flags
    tick(4)
    sources = action_sources(indices)
    primary = "NDVI" if mode == "multispectral" else "GLI"
    valid = np.isfinite(cells[primary])
    # roads, paths and open soil: no crop there, so nutrient and stress checks don't apply
    if bare_limit is None:
        bare = np.zeros_like(valid)
    else:
        bare = valid & (cells[primary] < bare_limit)
    limits, uniform = dict(thresholds), []
    if limit_mode == "relative":
        for action, src in sources.items():
            if src is None or action == "RESEEDING":
                continue          # bare soil is an absolute thing, not a relative one
            cuts = relative_limits(cells[src], valid & ~bare, percents)
            limits[f"{action}:{src}"] = cuts
            if cuts is None:
                uniform.append(ACTIONS[action]["label"])
        if uniform:
            notes.append("Nothing marked for " + ", ".join(uniform) +
                         ": these readings are almost the same across the whole field, "
                         "so there is no weaker part to point at.")
        info.append(f"Limits taken from this field: weakest {percents[0]}% Critical, "
                    f"next {percents[1]}% Moderate, next {percents[2]}% Low. "
                    "Reseeding keeps fixed limits, and places marked as bare soil are left "
                    "out of the other checks, so the marked shares come out a little smaller.")

    flags = {}
    for action, src in sources.items():
        if src is None:
            continue
        cuts = limits.get(f"{action}:{src}")
        if cuts is None:
            flags[action] = np.zeros((rows, cols), dtype=np.int8)
            continue
        lv = level_map(cells[src], cuts)
        lv[~valid] = 0
        if action != "RESEEDING":
            lv[bare] = 0          # bare ground is not a crop problem
        flags[action] = lv
    if "RESEEDING" in flags:
        reseed_mask = flags["RESEEDING"] > 0     # no crop there, so other checks don't apply
        bare = bare | reseed_mask
        for action, lv in flags.items():
            if action != "RESEEDING":
                lv[reseed_mask] = 0

    worst = np.zeros((rows, cols), dtype=np.int8)
    for lv in flags.values():
        worst = np.maximum(worst, lv)
    crop_mask = valid & ~bare
    healthy = crop_mask & (worst == 0)

    # 6. Summary  
    valid_n = int(valid.sum())
    if valid_n == 0:
        raise ValueError("No usable pixels were found. Check that the images are not blank "
                         "and that the bands match the photo.")
    stats = {}
    for action, lv in flags.items():
        c, m, lo = int((lv == 3).sum()), int((lv == 2).sum()), int((lv == 1).sum())
        stats[action] = {"critical": c, "moderate": m, "low": lo, "total": c + m + lo}
    crop_n = int(crop_mask.sum())
    scored = worst[crop_mask] if crop_n else worst[valid]
    score = round(100.0 - float(scored.mean()) / 3.0 * 100.0, 1)
    summary = {
        "score": score,
        "cells": rows * cols,
        "valid": valid_n,
        "crop": crop_n,
        "bare": int(bare.sum()),
        "healthy": int(healthy.sum()),
        "attention": max(crop_n - int(healthy.sum()), 0),
        "nodata": rows * cols - valid_n,
        "actions": stats,
    }
    if bare_limit is not None and int(bare.sum()):
        info.append(f"Bare ground (below {primary} {bare_limit:.2f}) covers "
                    f"{int(bare.sum()) / (rows * cols) * 100:.1f}% of the image. "
                    "It is left out of the health score and of the nutrient, stress and "
                    "growth checks.")

    # 7. Display image
    ds = min(1.0, MAX_DISPLAY_SIDE / max(pw, ph))
    if ds < 1.0:
        rgb = cv2.resize(rgb, (int(pw * ds), int(ph * ds)), interpolation=cv2.INTER_AREA)

    plot_summary = None
    if plots:
        plot_summary = plot_stats(cells, flags, crop_mask, healthy, worst,
                                  rows, cols, plots[0], plots[1])

    tick(5)
    return {
        "id": digest.hexdigest()[:16],
        "limit_mode": limit_mode,
        "percents": tuple(percents),
        "plots": plot_summary,
        "crop": tuple(crop),
        "mode": mode,
        "rows": rows,
        "cols": cols,
        "rgb": rgb,
        "orig_size": (orig_w, orig_h),
        "cells": cells,
        "sources": sources,
        "flags": flags,
        "valid": valid,
        "healthy": healthy,
        "summary": summary,
        "notes": notes,
        "info": info,
        "thresholds": {k: (list(v) if v else None) for k, v in limits.items()},
        "files": [rgb_file[0]] + [name for _, name, _ in band_files],
    }


# ─────────────────────────────────────────────────────────────
# CHARTS (matplotlib: text only, no emoji, so nothing renders as boxes)
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": LINE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def overlay_image(rgb, lv):
    h, w = rgb.shape[:2]
    big = cv2.resize(lv.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    out = rgb.astype(np.float32)
    for level, (_, color) in SEVERITY.items():
        mask = big == level
        out[mask] = out[mask] * 0.42 + _hex_to_rgb(color) * 0.58
    return out.astype(np.uint8)


def _fig_size(rgb, width=11.0):
    h, w = rgb.shape[:2]
    return width, max(3.0, width * h / w + 0.7)


def stamp(fig):
    fig.text(0.995, 0.005, BRAND, ha="right", va="bottom", fontsize=7.5,
             color=MUTED, alpha=0.85)
    return fig


def fig_to_png(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def make_action_map(rgb, lv, title, plots=None):
    fig, ax = plt.subplots(figsize=_fig_size(rgb), facecolor="white")
    ax.imshow(overlay_image(rgb, lv))
    if plots:
        _draw_plot_grid(ax, plots, rgb.shape[1], rgb.shape[0])
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=10)
    handles = [mpatches.Patch(color=SEVERITY[k][1], label=SEVERITY[k][0]) for k in (3, 2, 1)]
    ax.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.92, edgecolor=LINE)
    fig.tight_layout()
    return stamp(fig)


def _draw_plot_grid(ax, plots, width, height, labels_above=False):
    """Plot boundaries and names drawn over a map."""
    prow, pcol = plots["rows"], plots["cols"]
    names = np.array(plots["labels"]).reshape(prow, pcol)
    for r in range(prow):
        for c in range(pcol):
            x0, x1 = c * width / pcol, (c + 1) * width / pcol
            y0, y1 = r * height / prow, (r + 1) * height / prow
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                            edgecolor="white", linewidth=1.4, alpha=0.9))
            ax.text(x0 + (x1 - x0) * 0.04, y0 + (y1 - y0) * (0.12 if labels_above else 0.16),
                    names[r, c], fontsize=9, fontweight="bold", color="white", va="center",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="#00000099")])


def make_index_map(values, index_name, aspect_src, plots=None):
    vmin, vmax = INDEX_RANGE[index_name]
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(NODATA)
    fig, ax = plt.subplots(figsize=_fig_size(aspect_src), facecolor="white")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"{index_name} value (green = healthier)", color=MUTED)
    cbar.outline.set_edgecolor(LINE)
    if plots:
        _draw_plot_grid(ax, plots, values.shape[1], values.shape[0], labels_above=True)
    ax.set_title(f"{index_name} map", fontsize=14, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    fig.tight_layout()
    return stamp(fig)


def make_plot_map(rgb, plots, title="Plot map"):
    """The photo with plot boxes, labels and a health colour per plot."""
    prow, pcol = plots["rows"], plots["cols"]
    score = np.array(plots["score"], dtype=float).reshape(prow, pcol)
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=_fig_size(rgb), facecolor="white")
    ax.imshow(rgb)
    cmap = plt.get_cmap("RdYlGn")
    labels = np.array(plots["labels"]).reshape(prow, pcol)
    for r in range(prow):
        for c in range(pcol):
            x0, x1 = c * w / pcol, (c + 1) * w / pcol
            y0, y1 = r * h / prow, (r + 1) * h / prow
            value = score[r, c]
            colour = "#9AA39C" if not np.isfinite(value) else cmap(value / 100)
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=colour,
                                            alpha=0.38, edgecolor="white", linewidth=1.6))
            text = labels[r, c] if not np.isfinite(value) else f"{labels[r, c]}\n{value:.0f}"
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="#00000099")])
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.tight_layout()
    return stamp(fig)


def make_overview_chart(summary, actions):
    valid = summary["valid"]
    labels = [ACTIONS[a]["label"] for a in actions]
    y = np.arange(len(actions))
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(actions) + 1.3), facecolor="white")
    left = np.zeros(len(actions))
    for level in (3, 2, 1):
        key = LEVEL_NAMES[3 - level]
        vals = np.array([summary["actions"][a][key] / valid * 100 for a in actions])
        ax.barh(y, vals, left=left, color=SEVERITY[level][1], label=SEVERITY[level][0], height=0.58)
        left += vals
    for i, total in enumerate(left):
        ax.text(total + 0.8, i, f"{total:.1f}%", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(10.0, float(left.max()) * 1.18))
    ax.set_xlabel("% of analysed field")
    ax.set_title("Area needing action, by check", fontsize=13, fontweight="bold", loc="left", pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    return stamp(fig)


@st.cache_data(show_spinner=False, max_entries=40)
def action_map_png(result_id, action, title, _rgb, _lv, _plots=None):
    return fig_to_png(make_action_map(_rgb, _lv, title, _plots))


@st.cache_data(show_spinner=False, max_entries=40)
def index_map_png(result_id, index_name, _values, _aspect_src, _plots=None):
    return fig_to_png(make_index_map(_values, index_name, _aspect_src, _plots))


@st.cache_data(show_spinner=False, max_entries=10)
def plot_map_png(result_id, _rgb, _plots):
    return fig_to_png(make_plot_map(_rgb, _plots))


@st.cache_data(show_spinner=False, max_entries=10)
def overview_png(result_id, _summary, _actions):
    return fig_to_png(make_overview_chart(_summary, _actions))


# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=5)
def build_csv(result_id, _res):
    rows, cols = _res["rows"], _res["cols"]
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    data = {"row": rr.ravel(), "col": cc.ravel()}
    for name in INDEX_ORDER:
        if name in _res["cells"]:
            data[name] = np.round(_res["cells"][name].ravel(), 4)
    for action, lv in _res["flags"].items():
        data[f"{action.lower()}_level"] = lv.ravel()
    if _res.get("plots"):
        ids = plot_ids(rows, cols, _res["plots"]["rows"], _res["plots"]["cols"]).ravel()
        names = np.array(_res["plots"]["labels"])
        data["plot"] = names[ids]
    data["has_data"] = _res["valid"].ravel().astype(int)
    data["healthy"] = _res["healthy"].ravel().astype(int)
    return pd.DataFrame(data).to_csv(index=False).encode()


def build_json(res, meta, cost):
    return json.dumps({
        "app": "GreenScan",
        "made_by": BRAND,
        "contact": BRAND_CONTACT,
        "field_name": meta.get("field_name", ""),
        "crop": meta.get("crop_type", ""),
        "generated": meta["created"],
        "mode": res["mode"],
        "files": res["files"],
        "health_score": res["summary"]["score"],
        "grid": {"rows": res["rows"], "cols": res["cols"], "cell_area_sqm": round(meta["cell_area"], 4)},
        "camera": meta["camera"],
        "location": meta.get("geo") and {
            "photo_centre_lat": meta["geo"]["lat"], "photo_centre_lon": meta["geo"]["lon"],
            "heading_deg": round(meta["geo"]["yaw"], 1),
        },
        "summary": res["summary"],
        "indices_used": {a: s for a, s in res["sources"].items() if s},
        "limit_mode": res.get("limit_mode", "fixed"),
        "limits_used": res["thresholds"],
        "plots": plot_rows(res, meta) or None,
        "cost_estimate_inr": cost,
        "field_notes": meta["notes"],
    }, indent=2).encode()


KML_COLOURS = {3: "ff2b39b8", 2: "ff2a7fdb", 1: "ff3ac1e4"}   # KML uses aabbggrr


def build_kml(res, meta, limit=10):
    """Placemarks for the worst patches, plus the boundary of the analysed area."""
    geo = meta.get("geo")
    if not geo:
        return None
    name = meta.get("field_name") or "Field"
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f"<name>{html.escape(name)} - GreenScan</name>",
             f"<description>Made with GreenScan by {BRAND} on {html.escape(meta['created'])}."
             " Positions are worked out from the photo's GPS, so allow a few metres of error."
             "</description>"]
    for level, colour in KML_COLOURS.items():
        parts.append(f'<Style id="sev{level}"><IconStyle><color>{colour}</color><scale>1.1</scale>'
                     '<Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon>'
                     f'</IconStyle><LineStyle><color>{colour}</color><width>2</width></LineStyle>'
                     '</Style>')

    corners = [cell_latlon(geo, x, y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))]
    ring = " ".join(f"{lon},{lat},0" for lat, lon in corners)
    parts.append("<Placemark><name>Analysed area</name><styleUrl>#sev1</styleUrl>"
                 f"<LineString><tessellate>1</tessellate><coordinates>{ring}</coordinates>"
                 "</LineString></Placemark>")

    for action, lv in res["flags"].items():
        spots = problem_spots(lv, meta, res["cols"], res["rows"], limit=limit)
        spots = [sp for sp in spots if "lat" in sp]
        if not spots:
            continue
        label = ACTIONS[action]["label"]
        parts.append(f"<Folder><name>{label}</name>")
        for i, sp in enumerate(spots, 1):
            parts.append(
                f"<Placemark><name>{label} {i} - {sp['area']:,.1f} sq m</name>"
                f"<description>About {sp['area']:,.1f} sq m needs attention, in the "
                f"{sp['where']} of the analysed area ({spread_text(sp)}). "
                f"{html.escape(ACTIONS[action]['todo'])}"
                "</description><styleUrl>#sev3</styleUrl>"
                f"<Point><coordinates>{sp['lon']},{sp['lat']},0</coordinates></Point></Placemark>")
        parts.append("</Folder>")
    parts.append("</Document></kml>")
    return "\n".join(parts).encode()


def _pdf_table(ax, header, rows, col_widths):
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, colWidths=col_widths,
                     loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if r == 0:
            cell.set_facecolor("#E4EBE1")
            cell.set_text_props(fontweight="bold")


def build_pdf(res, meta, cost):
    s = res["summary"]
    actions = list(res["flags"])
    small = res["rgb"]
    if max(small.shape[:2]) > 1000:
        k = 1000 / max(small.shape[:2])
        small = cv2.resize(small, (int(small.shape[1] * k), int(small.shape[0] * k)),
                           interpolation=cv2.INTER_AREA)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: summary
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        label, color = score_label(s["score"])
        mode_txt = "Multispectral images" if res["mode"] == "multispectral" else "RGB photo only (approximate)"
        title = meta.get("field_name") or "Field health report"
        fig.text(0.05, 0.93, title, fontsize=22, fontweight="bold")
        sub = f"Generated {meta['created']}   |   Analysis: {mode_txt}"
        if meta.get("crop_type"):
            sub = f"Crop: {meta['crop_type']}   |   " + sub
        fig.text(0.05, 0.895, sub, fontsize=10, color=MUTED)
        fig.text(0.95, 0.93, "GreenScan", fontsize=13, fontweight="bold", color=CROP, ha="right")
        fig.text(0.95, 0.905, f"by {BRAND}", fontsize=9, color=MUTED, ha="right")
        fig.text(0.05, 0.80, f"{s['score']:.0f}", fontsize=54, fontweight="bold", color=color)
        fig.text(0.155, 0.815, "/100", fontsize=16, color=MUTED)
        fig.text(0.05, 0.765, f"Field health score: {label}", fontsize=12, fontweight="bold")
        crop_base = max(s.get("crop") or s["valid"], 1)
        facts = [
            f"Healthy: {s['healthy'] / crop_base * 100:.1f}% of the crop area",
            f"Needs attention: {s['attention'] / crop_base * 100:.1f}%",
            f"Area analysed: {meta['area_sqm']:,.0f} sq m ({meta['area_sqm'] / SQM_PER_ACRE:.2f} acres)",
        ]
        if res.get("limit_mode") == "relative":
            pc, pm, pl = res["percents"]
            facts.append(f"Limits from this field: weakest {pc}% Critical, next {pm}% Moderate, "
                         f"next {pl}% Low")
        trims = meta.get("crop_percent") or {}
        if any(trims.values()):
            facts.append("Trimmed from the image: "
                         + ", ".join(f"{k} {v}%" for k, v in trims.items() if v))
        if s.get("bare"):
            facts.append(f"Bare ground left out of the score: "
                         f"{s['bare'] * meta['cell_area']:,.1f} sq m")
        if s["nodata"]:
            facts.append(f"No data: {s['nodata'] * meta['cell_area']:,.1f} sq m "
                         "(blank or unreadable parts of the image)")
        for i, line in enumerate(facts):
            fig.text(0.40, 0.855 - i * 0.027, line, fontsize=10.5)

        ax1 = fig.add_axes([0.05, 0.38, 0.90, 0.30])
        header = ["Check", "Based on", "Critical (sq m)", "Moderate (sq m)", "Low (sq m)",
                  "Total (sq m)", "% of field", "Priority"]
        area_of = lambda n: f"{n * meta['cell_area']:,.1f}"
        rows = []
        for a in actions:
            st_ = s["actions"][a]
            rows.append([ACTIONS[a]["label"], res["sources"][a], area_of(st_["critical"]),
                         area_of(st_["moderate"]), area_of(st_["low"]), area_of(st_["total"]),
                         f"{st_['total'] / s['valid'] * 100:.1f}%", priority_of(st_, s["valid"])])
        for a, src in res["sources"].items():
            if src is None:
                rows.append([ACTIONS[a]["label"], "-", "-", "-", "-", "-", "-",
                             f"Not checked (needs {ACTIONS[a]['needs']})"])
        _pdf_table(ax1, header, rows, [0.14, 0.09, 0.12, 0.12, 0.10, 0.10, 0.09, 0.24])

        ax2 = fig.add_axes([0.05, 0.05, 0.42, 0.30])
        if cost:
            crow = [[r["treatment"], f"{r['area_sqm']:,.1f}", f"Rs {r['cost']:,.0f}"] for r in cost["rows"]]
            crow.append([f"Overhead ({cost['overhead_pct']:.0f}%)", "-", f"Rs {cost['overhead']:,.0f}"])
            crow.append(["Total estimate", "-", f"Rs {cost['total']:,.0f}"])
            _pdf_table(ax2, ["Treatment", "Area treated (sq m)", "Cost"], crow, [0.45, 0.30, 0.25])
        else:
            ax2.axis("off")
            ax2.text(0, 0.9, "Open the Cost estimate tab to include costs.", color=MUTED)

        ax3 = fig.add_axes([0.53, 0.05, 0.42, 0.30])
        ax3.axis("off")
        ax3.text(0, 0.97, "Field notes", fontsize=11, fontweight="bold", va="top")
        note = meta["notes"].strip() or "No field notes added."
        wrapped = "\n".join(textwrap.fill(p, 62) for p in note.splitlines())
        wrapped = "\n".join(wrapped.splitlines()[:14])
        ax3.text(0, 0.86, wrapped, fontsize=9, va="top", color=INK)
        fig.text(0.05, 0.015, "Results are estimates from image analysis. Check the field before any treatment.",
                 fontsize=8, color=MUTED)
        stamp(fig)
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

        # Page 2: overview chart
        if actions:
            f2 = make_overview_chart(s, actions)
            pdf.savefig(f2, facecolor="white")
            plt.close(f2)

        # Action maps
        for a in actions:
            f3 = make_action_map(small, res["flags"][a],
                                 f"{ACTIONS[a]['label']} map (based on {res['sources'][a]})",
                                 res.get("plots"))
            spots = problem_spots(res["flags"][a], meta, res["cols"], res["rows"], limit=3)
            if spots:
                lines = "   |   ".join(
                    f"{i}. {s_['area']:,.1f} sq m, {s_['where']}"
                    + (f" ({s_['lat']:.5f}, {s_['lon']:.5f})" if "lat" in s_ else "")
                    for i, s_ in enumerate(spots, 1))
                f3.text(0.01, 0.005, f"Spots needing attention first:  {lines}",
                        fontsize=8.5, color=MUTED)
            pdf.savefig(f3, facecolor="white")
            plt.close(f3)

        # Plots
        if res.get("plots"):
            fp = make_plot_map(small, res["plots"], "Plot map (score out of 100)")
            pdf.savefig(fp, facecolor="white")
            plt.close(fp)
            rows_p = plot_rows(res, meta)
            if rows_p:
                header_p = list(rows_p[0].keys())
                chunk = 18
                for start in range(0, len(rows_p), chunk):
                    part = rows_p[start:start + chunk]
                    figp = plt.figure(figsize=(11.69, 8.27), facecolor="white")
                    figp.text(0.05, 0.94, "Plots, weakest first", fontsize=16, fontweight="bold")
                    axp = figp.add_axes([0.05, 0.08, 0.90, 0.80])
                    def fmt(value):
                        if value is None:
                            return "-"
                        if isinstance(value, float):
                            return f"{value:,.2f}" if abs(value) < 3 else f"{value:,.1f}"
                        return str(value)
                    body = [[fmt(r[k]) for k in header_p] for r in part]
                    _pdf_table(axp, [h.replace("m²", "sq m") for h in header_p], body,
                               [0.08, 0.11, 0.10, 0.11, 0.15, 0.12, 0.14, 0.17])
                    stamp(figp)
                    pdf.savefig(figp, facecolor="white")
                    plt.close(figp)

        # Primary index map
        primary = "NDVI" if res["mode"] == "multispectral" else "GLI"
        f4 = make_index_map(res["cells"][primary], primary, small, res.get("plots"))
        pdf.savefig(f4, facecolor="white")
        plt.close(f4)

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# UI PIECES
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=8)
def crop_preview_png(name, data, crop):
    """Small preview of the photo with the chosen analysis area marked."""
    img = decode_rgb(name, data, 4)
    if img is None:
        return None
    scale = min(1.0, 900 / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    x0, x1 = int(w * crop[0]), w - int(w * crop[1])
    y0, y1 = int(h * crop[2]), h - int(h * crop[3])
    shade = (img * 0.45).astype(np.uint8)
    shade[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    cv2.rectangle(shade, (x0, y0), (x1 - 1, y1 - 1), (80, 180, 90), 3)
    return cv2.cvtColor(shade, cv2.COLOR_BGR2RGB)


def step_header(number, title, note=None):
    note_html = f"<p class='gs-step-note'>{note}</p>" if note else ""
    st.markdown(
        f"<div class='gs-step'><span class='gs-step-n'>{number}</span><h3>{title}</h3></div>{note_html}",
        unsafe_allow_html=True,
    )


class Progress:
    """Ticks off the analysis stages in the sidebar."""

    def __init__(self, slot):
        self.slot = slot
        self.done = 0

    def render(self, running=False, title="Progress"):
        rows = []
        for i, stage in enumerate(STAGES):
            if i < self.done:
                rows.append(f"<div class='done'><i>&#10003;</i>{stage}</div>")
            elif i == self.done and running:
                rows.append(f"<div class='now'><i>&rarr;</i>{stage}</div>")
            else:
                rows.append(f"<div><i>&middot;</i>{stage}</div>")
        self.slot.markdown(f"<b>{title}</b><div class='gs-check'>{''.join(rows)}</div>",
                           unsafe_allow_html=True)

    def step(self, index):
        self.done = max(self.done, index)
        self.render(running=True)

    def finish(self):
        self.done = len(STAGES)
        self.render(title="Analysis complete")


def make_stage_reporter(progress, status):
    """Ticks the sidebar checklist and writes the same steps into the status box."""
    seen = set()

    def report(stage):
        progress.step(stage)
        if stage not in seen and stage < len(STAGES):
            seen.add(stage)
            status.write(STAGES[stage])
            status.update(label=STAGES[stage])
    return report


def render_footer():
    st.markdown(
        f"<div class='gs-footer'><span class='mark'><i></i>{BRAND}</span>"
        f"<span>GreenScan &middot; {BRAND_CONTACT}</span></div>",
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div class='gs-brand'><div class='gs-brand-mark'>GS</div><div>"
            "<div class='gs-brand-name'>GreenScan</div>"
            "<div class='gs-brand-sub'>Crop health from drone images</div></div></div>",
            unsafe_allow_html=True,
        )
        st.radio("Go to", [NAV_UPLOAD, NAV_RESULTS], key="nav", label_visibility="collapsed")
        st.divider()
        slot = st.empty()
        progress = Progress(slot)

        res = st.session_state.result
        if res is not None:
            progress.finish()
            s = res["summary"]
            label, color = score_label(s["score"])
            st.divider()
            st.markdown(
                f"<div class='gs-side-score' style='color:{color}'>{s['score']:.0f}<small> / 100</small></div>"
                f"<div style='font-weight:700'>{label}</div>",
                unsafe_allow_html=True,
            )
            base = max(s.get("crop") or s["valid"], 1)
            st.caption(
                f"Healthy: {s['healthy'] / base * 100:.0f}% of the crop area\n\n"
                f"Needs attention: {s['attention'] / base * 100:.0f}%"
            )
        else:
            progress.render(title="Progress")
            st.caption("Add your images and run the analysis.")

        st.divider()
        st.caption(
            "Multispectral images are analysed with NDVI, NDRE, LCI and GNDVI. "
            "Normal photos use VARI and GLI. Results are estimates, so check the field before treating."
        )
        st.markdown(f"<div class='gs-side-footer'><b>{BRAND}</b><br>{BRAND_CONTACT}</div>",
                    unsafe_allow_html=True)
    return progress


def render_mode_box(mode, sources, band_keys):
    checked = [ACTIONS[a]["label"] for a, s in sources.items() if s]
    skipped = [ACTIONS[a]["label"] for a, s in sources.items() if not s]
    if mode == "multispectral":
        used = ", ".join(sorted({s for s in sources.values() if s}, key=INDEX_ORDER.index))
        title = "Multispectral analysis"
        body = f"Checks: {', '.join(checked)}. Indices used: {used}."
        if skipped:
            body += f" Skipped: {', '.join(skipped)} (add the RedEdge band)."
        css = ""
    else:
        title = "RGB photo only: approximate results"
        body = (f"Checks: {', '.join(checked)}. Water stress, nitrogen and crop stress "
                "need Red, NIR and RedEdge bands from a multispectral camera.")
        missing = [BANDS[k]["label"] for k in ("red", "nir") if k not in band_keys]
        if band_keys and missing:
            body += f" To use your bands, also add: {', '.join(missing)}."
        css = " rgb"
    st.markdown(f"<div class='gs-mode{css}'><strong>{title}</strong><span>{body}</span></div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 1: UPLOAD & SET UP
# ─────────────────────────────────────────────────────────────
def page_upload(progress):
    st.markdown(
        "<div class='gs-hero'><h1>Check your field's health from drone images</h1>"
        "<p>Upload a drone photo of the field, plus multispectral bands if you have them. "
        "GreenScan marks the problem areas, suggests what to do and estimates the treatment cost.</p></div>",
        unsafe_allow_html=True,
    )

    samples = find_sample_files()
    kept = st.session_state.kept_files

    # Step 1: images
    step_header(1, "Add field images",
                "The RGB photo is required. Add Red and NIR bands (plus Green and RedEdge) "
                "for the full multispectral analysis.")
    if kept is not None:
        names = ", ".join([kept["rgb"][0]] + [n for n, _ in kept["bands"].values()])
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.info(f"Using {kept['label']}: {names}")
        c2.button("Use different images", on_click=clear_kept_files, width="stretch")
        rgb_file = kept["rgb"]
        band_files = dict(kept["bands"])
    else:
        if "rgb" in samples:
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.caption("No drone images at hand? Try GreenScan with a sample field.")
            c2.button("Load sample field", on_click=use_sample_files, width="stretch")
        up = st.file_uploader(
            "RGB photo of the field (required)", type=IMAGE_TYPES, key="up_rgb",
            help="A normal colour drone photo of the field (JPG, PNG or TIFF). "
                 "Used for the maps, and for the analysis when no bands are added.",
        )
        st.markdown("**Multispectral bands** (optional)")
        band_files = {}
        for col, key in zip(st.columns(4), BAND_ORDER):
            with col:
                f = st.file_uploader(BANDS[key]["label"], type=IMAGE_TYPES,
                                     key=f"up_{key}", help=BANDS[key]["help"])
                if f is not None:
                    band_files[key] = (f.name, f.getvalue())
        rgb_file = (up.name, up.getvalue()) if up is not None else None

    mode, indices = available_indices(band_files.keys())
    sources = action_sources(indices)
    render_mode_box(mode, sources, list(band_files))

    # Step 2: camera
    step_header(2, "Camera and flight details",
                "Used to work out how much ground each pixel and each grid cell covers. "
                "Hover over the ? next to a field to see what it means.")
    # The analysis runs on the NIR band in multispectral mode, otherwise on the RGB photo
    frame_file = band_files["nir"] if mode == "multispectral" else rgb_file
    cam = camera_info(*frame_file) if frame_file else None
    if cam and cam["width"]:
        signature = (frame_file[0], len(frame_file[1]))
        if st.session_state["_keep"].get("_cam_sig") != signature:
            set_widget_value("cam_iw", int(min(max(cam["width"], 100), 20000)))
            set_widget_value("cam_ih", int(min(max(cam["height"], 100), 20000)))
            if cam["flight_height"]:
                set_widget_value("cam_fh", float(min(max(cam["flight_height"], 1.0), 1000.0)))
            if cam["sensor_width"] and cam["focal_length"]:
                set_widget_value("cam_sw", float(min(max(cam["sensor_width"], 1.0), 60.0)))
                set_widget_value("cam_fl", float(min(max(cam["focal_length"], 0.5), 300.0)))
            st.session_state["_keep"]["_cam_sig"] = signature
        filled = ["image size"]
        if cam["flight_height"]:
            filled.append("flight height")
        if cam["sensor_width"] and cam["focal_length"]:
            filled.append("sensor width and focal length")
        model = f" ({html.escape(cam['model'])})" if cam["model"] else ""
        st.caption(f"Filled in from the camera data in **{html.escape(frame_file[0])}**{model}: "
                   f"{', '.join(filled)}. Please check the values.")

    cam_box = st.container(border=True)
    left, right = cam_box.columns([3, 2], gap="large")
    with left:
        c1, c2 = st.columns(2)
        with c1:
            fh = kept_widget(st.number_input, "Flight height (m)", "cam_fh", 50.0,
                             min_value=1.0, max_value=1000.0, step=1.0, help=HELP["flight_height"])
            sw = kept_widget(st.number_input, "Sensor width (mm)", "cam_sw", 6.17,
                             min_value=1.0, max_value=60.0, step=0.01, help=HELP["sensor_width"])
            fl = kept_widget(st.number_input, "Focal length (mm)", "cam_fl", 4.5,
                             min_value=0.5, max_value=300.0, step=0.1, help=HELP["focal_length"])
        with c2:
            iw = kept_widget(st.number_input, "Image width (px)", "cam_iw", 4000,
                             min_value=100, max_value=20000, step=10, help=HELP["image_width"])
            ih = kept_widget(st.number_input, "Image height (px)", "cam_ih", 3000,
                             min_value=100, max_value=20000, step=10, help=HELP["image_height"])
    gsd, ground_w, ground_h = ground_footprint(fh, sw, fl, iw, ih)
    full_area = ground_w * ground_h
    with right:
        st.metric("Ground detail (GSD)", f"{gsd * 100:.2f} cm per pixel", help=HELP["gsd"], border=True)
        st.metric("Ground area in the image", f"{ground_w:,.0f} m × {ground_h:,.0f} m",
                  help=HELP["footprint"], border=True)
        st.metric("Area in the image", f"{full_area:,.0f} m²  ≈ {full_area / SQM_PER_ACRE:.2f} acres",
                  help=HELP["total_area"], border=True)

    # Step 3: analysis area and grid
    step_header(3, "Analysis area and grid",
                "Leave out roads, buildings and neighbouring fields, then choose how finely "
                "the field is split into cells.")
    crop_keys = [("crop_l", "Trim from left (%)"), ("crop_r", "Trim from right (%)"),
                 ("crop_t", "Trim from top (%)"), ("crop_b", "Trim from bottom (%)")]
    with st.container(border=True):
        area_col, preview_col = st.columns([3, 2], gap="large")
        with area_col:
            st.markdown("**Analysis area**")
            trims = []
            cc = st.columns(2)
            for i, (key, label) in enumerate(crop_keys):
                with cc[i % 2]:
                    trims.append(kept_widget(
                        st.number_input, label, key, 0,
                        min_value=0, max_value=45, step=5,
                        help="Part of the image to leave out of the analysis. "
                             "Use it to cut out roads, sheds or other fields.",
                    ))
            crop = tuple(t / 100 for t in trims)
            st.caption("0 means the whole image is analysed.")
        with preview_col:
            if rgb_file is not None:
                preview = crop_preview_png(rgb_file[0], rgb_file[1], crop)
                if preview is not None:
                    st.image(preview, caption="Bright part = analysed area", width="stretch")
                    if mode == "multispectral":
                        st.caption("The bands cover a slightly smaller area than this photo, "
                                   "so treat the preview as a guide.")
            else:
                st.caption("Add the RGB photo to see a preview.")

    keep_w, keep_h = 1 - crop[0] - crop[1], 1 - crop[2] - crop[3]
    area_sqm = ground_w * keep_w * ground_h * keep_h
    geo = None
    if cam and cam["lat"] is not None:
        geo = {
            "lat": cam["lat"], "lon": cam["lon"],
            "yaw": cam["yaw"] if cam["yaw"] is not None else 0.0,
            "north_up": cam["yaw"] is None,
            "width_m": ground_w * keep_w, "height_m": ground_h * keep_h,
            # middle of the analysed area, measured from the middle of the photo
            "offset_u": ((crop[0] + 1 - crop[1]) / 2 - 0.5) * ground_w,
            "offset_v": ((crop[2] + 1 - crop[3]) / 2 - 0.5) * ground_h,
        }
    with st.container(border=True):
        st.markdown("**Grid**")
        c1, c2, c3 = st.columns(3)
        with c1:
            grid_c = kept_widget(st.number_input, "Grid columns", "grid_c", DEFAULT_GRID_COLS,
                                 min_value=16, max_value=512, step=16, help=HELP["grid_cols"])
        with c2:
            grid_r = kept_widget(st.number_input, "Grid rows", "grid_r", DEFAULT_GRID_ROWS,
                                 min_value=12, max_value=384, step=12, help=HELP["grid_rows"])
        cell_area = area_sqm / (grid_r * grid_c)
        with c3:
            st.metric("Area per cell", f"{cell_area:,.2f} m²",
                      help=f"{HELP['cell_area']} Total cells: {grid_r * grid_c:,}.", border=True)
        st.caption(f"Analysed area: {area_sqm:,.0f} m² ≈ {area_sqm / SQM_PER_ACRE:.2f} acres.")
        if geo:
            note = ("The photo has no direction recorded, so north is taken as up. "
                    if geo["north_up"] else "")
            st.caption(f"Location found in the photo: {geo['lat']:.5f}, {geo['lon']:.5f}. "
                       f"{note}Problem spots will get map links and a KML file.")
        else:
            st.caption("No GPS found in the image, so map links and the KML file will not be available. "
                       "Upload the original photo straight from the drone to keep its location data.")
        photo_ratio = (ih * keep_h) / (iw * keep_w)
        grid_ratio = grid_r / grid_c
        if abs(photo_ratio - grid_ratio) / photo_ratio > 0.15:
            suggested = max(12, int(round(grid_c * photo_ratio)))
            st.caption(f"Tip: for square cells, use about {suggested} rows with {grid_c} columns.")

    with st.container(border=True):
        use_plots = kept_widget(
            st.checkbox, "Split the field into plots", "use_plots", False,
            help="Divide the analysed area into equal plots (A1, A2, B1...) and get a score "
                 "for each one. Useful for trial plots or for splitting a big field into parts.",
        )
        if use_plots:
            c1, c2, c3 = st.columns(3)
            with c1:
                plot_r = kept_widget(st.number_input, "Plot rows", "plot_r", 3,
                                     min_value=1, max_value=12, step=1,
                                     help="How many plot rows, top to bottom.")
            with c2:
                plot_c = kept_widget(st.number_input, "Plot columns", "plot_c", 4,
                                     min_value=1, max_value=12, step=1,
                                     help="How many plot columns, left to right.")
            plot_area = area_sqm / (plot_r * plot_c)
            with c3:
                st.metric("Area per plot", f"{plot_area:,.1f} m²",
                          help=f"{plot_r * plot_c} plots in total.", border=True)
            st.caption("Plots are named A1, A2... from the top-left of the analysed area.")
        else:
            plot_r = plot_c = None

    # Detection limits
    active = [(a, s) for a, s in sources.items() if s]
    thresholds, problems = {}, []
    with st.expander("Detection limits"):
        limit_choice = kept_widget(
            st.radio, "How should problem areas be decided?", "limit_mode",
            "Fixed limits (standard values)",
            options=["Fixed limits (standard values)", "Compare within this field"],
            help="Fixed limits use standard index values that suit most field crops. "
                 "Comparing within the field marks the weakest parts of this field itself, "
                 "which works better when your crop or sensor reads differently.",
        )
        limit_mode = "relative" if limit_choice.startswith("Compare") else "fixed"
        if limit_mode == "relative":
            st.caption("The weakest parts of this field are marked. Reseeding still uses fixed "
                       "limits, because bare soil is an absolute thing. If a reading is almost "
                       "the same everywhere, nothing is marked for it.")
            c1, c2, c3 = st.columns(3)
            with c1:
                pc = kept_widget(st.number_input, "Weakest % as Critical", "pct_c", 5,
                                 min_value=1, max_value=30, step=1,
                                 help="This share of the field, the weakest part, is marked Critical.")
            with c2:
                pm = kept_widget(st.number_input, "Next % as Moderate", "pct_m", 10,
                                 min_value=1, max_value=40, step=1,
                                 help="The next slice after the Critical one.")
            with c3:
                pl = kept_widget(st.number_input, "Next % as Low", "pct_l", 15,
                                 min_value=1, max_value=50, step=1,
                                 help="The next slice after the Moderate one.")
            percents = (int(pc), int(pm), int(pl))
            if sum(percents) > 80:
                msg = "The three shares add up to more than 80% of the field. Use smaller shares."
                st.error(msg)
                problems.append(msg)
            st.caption("The limits worked out for your field are shown in the report.")
        else:
            percents = (5, 10, 15)
        st.divider()
        default_bare = 0.30 if mode == "multispectral" else 0.05
        primary_name = "NDVI" if mode == "multispectral" else "GLI"
        skip_bare = kept_widget(
            st.checkbox, "Leave out bare ground (roads, paths, open soil)", "skip_bare", True,
            help="Bare ground has no crop, so nutrient, stress and growth checks do not apply "
                 "there. It is also kept out of the health score. Reseeding still marks it.",
        )
        if skip_bare:
            bare_limit = kept_widget(
                st.number_input, f"Treat as bare ground below {primary_name}",
                f"bare_limit_{primary_name.lower()}", default_bare,
                min_value=-0.2, max_value=0.6, step=0.01,
                help=f"Places with {primary_name} below this value are treated as bare ground. "
                     "0.30 works for most crops; lower it if your crop is very young.",
            )
        else:
            bare_limit = None
        st.divider()
        st.caption("Fixed limits: a place is marked Critical, Moderate or Low when its index value "
                   "is below these numbers. The defaults suit most field crops."
                   + (" In compare mode only Reseeding uses these." if limit_mode == "relative" else ""))
        st.button("Reset to defaults", on_click=reset_thresholds)
        tabs = st.tabs([ACTIONS[a]["label"] for a, _ in active])
        for tab, (action, src) in zip(tabs, active):
            with tab:
                st.caption(f"Based on **{src}**. {INDEX_INFO[src]}")
                defaults = DEFAULT_THRESHOLDS[f"{action}:{src}"]
                values = []
                for col, level, default in zip(st.columns(3), LEVEL_NAMES, defaults):
                    with col:
                        values.append(kept_widget(
                            st.slider, f"{level.capitalize()} below", f"thr_{action}_{src}_{level}",
                            float(default), min_value=-0.5, max_value=1.0, step=0.01,
                            help=f"Cells with {src} below this value are marked {level.capitalize()}.",
                        ))
                if not values[0] < values[1] < values[2]:
                    msg = (f"{ACTIONS[action]['label']}: the limits must go up in order "
                           "(Critical < Moderate < Low).")
                    st.error(msg)
                    problems.append(msg)
                thresholds[f"{action}:{src}"] = tuple(round(v, 4) for v in values)

    # Step 4: report details and run
    step_header(4, "Report details and analysis")
    with st.container(border=True):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            field_name = kept_widget(
                st.text_input, "Field or client name", "field_name", "",
                placeholder="For example: Sharma farm, plot 4",
                help="Shown at the top of the report and used in the file names.",
            )
        with c2:
            crop_type = kept_widget(
                st.text_input, "Crop (optional)", "crop_type", "",
                placeholder="For example: paddy, wheat, sugarcane",
                help="Written into the report so you know which crop the numbers belong to.",
            )
        notes = kept_widget(
            st.text_area, "Field notes (optional)", "field_notes", "",
            placeholder="For example: heavy rain last week, clay soil in the north-east corner, "
                        "pests seen near the boundary, last irrigation 5 days ago",
            height=90, help=HELP["notes"],
        )
    if rgb_file is None:
        problems.insert(0, "Upload the RGB photo of the field to start.")

    run = st.button("Analyse field", type="primary", disabled=bool(problems))
    if problems:
        st.caption(problems[0])

    if run:
        band_tuple = tuple((k, band_files[k][0], band_files[k][1]) for k in BAND_ORDER if k in band_files)
        status = st.status("Working on your field...", expanded=True)
        try:
            res = run_analysis(rgb_file, band_tuple, int(grid_r), int(grid_c),
                               tuple(sorted(thresholds.items())), crop,
                               limit_mode, percents,
                               (int(plot_r), int(plot_c)) if use_plots else None,
                               float(bare_limit) if bare_limit is not None else None,
                               _progress=make_stage_reporter(progress, status))
        except ValueError as exc:
            status.update(label="Could not finish the analysis", state="error")
            st.error(str(exc))
            return
        except Exception as exc:  # show details instead of crashing
            status.update(label="Could not finish the analysis", state="error")
            st.error("Something went wrong while analysing the images. "
                     "Check that all files are valid images of the same field.")
            with st.expander("Error details"):
                st.exception(exc)
            return
        status.update(label="Analysis complete", state="complete", expanded=False)

        if kept is None:
            st.session_state.kept_files = {
                "label": "the images from your last analysis",
                "rgb": rgb_file,
                "bands": band_files,
            }
        st.session_state.result = res
        st.session_state.meta = {
            "created": datetime.now().strftime("%d %b %Y, %H:%M"),
            "field_name": field_name.strip(),
            "crop_type": crop_type.strip(),
            "cell_area": cell_area,
            "area_sqm": area_sqm,
            "gsd_m": gsd,
            "camera": {"flight_height_m": fh, "sensor_width_mm": sw, "focal_length_mm": fl,
                       "image_width_px": iw, "image_height_px": ih},
            "crop_percent": {"left": trims[0], "right": trims[1], "top": trims[2], "bottom": trims[3]},
            "geo": geo,
            "ground_size_m": (round(ground_w * keep_w, 2), round(ground_h * keep_h, 2)),
            "notes": notes,
        }
        st.session_state.pdf = None
        st.session_state.cost = None
        s = res["summary"]
        st.success(f"Analysis complete. Field health score: {s['score']:.0f}/100.")
        st.button("Open results", type="primary", on_click=go_to, args=(NAV_RESULTS,))


# ─────────────────────────────────────────────────────────────
# PAGE 2: RESULTS
# ─────────────────────────────────────────────────────────────
def render_report_header(res, meta):
    s = res["summary"]
    label, color = score_label(s["score"])
    total = s["cells"]
    parts = [
        ("Healthy crop", s["healthy"], CROP),
        ("Needs attention", s["attention"], SEVERITY[2][1]),
        ("Bare ground", s.get("bare", 0), "#8A5A3B"),
        ("No data", s["nodata"], NODATA),
    ]
    strip = "".join(
        f"<span style='width:{n / total * 100:.3f}%;background-color:{c}'></span>" for _, n, c in parts if n
    )
    legend = "".join(
        f"<span><i style='background:{c}'></i>{name} {n / total * 100:.1f}%</span>" for name, n, c in parts if n
    )
    mode_txt = "multispectral images" if res["mode"] == "multispectral" else "RGB photo only"
    extra = f"{html.escape(meta['crop_type'])} &middot; " if meta.get("crop_type") else ""
    st.markdown(
        f"""<div class='gs-report'>
<div>
  <div class='gs-score' style='color:{color}'>{s['score']:.0f}<small>/100</small></div>
  <div class='gs-score-label' style='color:{color}'>{label}</div>
  <div class='gs-score-note'>Field health score</div>
</div>
<div>
  <h2>{html.escape(meta.get('field_name') or 'Field health report')}</h2>
  <p class='gs-report-meta'>{extra}Analysed {html.escape(meta['created'])} from {mode_txt}.
  Area checked: {meta['area_sqm']:,.0f} m² ({meta['area_sqm'] / SQM_PER_ACRE:.2f} acres).</p>
  <div class='gs-strip'>{strip}</div>
  <div class='gs-legend'>{legend}</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_action_tab(res, meta, action):
    m = ACTIONS[action]
    src = res["sources"][action]
    s = res["summary"]
    stat = s["actions"][action]
    valid = s["valid"]
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        title = f"{m['label']} map (based on {src})"
        st.image(action_map_png(res["id"], action, title, res["rgb"], res["flags"][action],
                                res.get("plots")), width="stretch")
    with right:
        st.markdown(f"<div class='gs-panel'><h4>{m['label']}</h4><p>{m['desc']}</p></div>",
                    unsafe_allow_html=True)
        for level in (3, 2, 1):
            name = SEVERITY[level][0]
            n = stat[name.lower()]
            st.metric(name, f"{n * meta['cell_area']:,.1f} m²", f"{n / valid * 100:.1f}% of field",
                      delta_color="off", delta_arrow="off")
        st.metric("Total area needing action", f"{stat['total'] * meta['cell_area']:,.1f} m²",
                  help="Critical, Moderate and Low areas added together for this check.")
        st.metric("Priority", priority_of(stat, valid),
                  help="High: over 10% of the field is Critical. "
                       "Medium: over 5% of the field needs action. Otherwise Low.")
        st.markdown(f"<div class='gs-panel'><h4>What to do</h4><p>{m['todo']}</p></div>",
                    unsafe_allow_html=True)
        spots = problem_spots(res["flags"][action], meta, res["cols"], res["rows"])
        if spots:
            st.markdown("**Where to go first**")
            for i, spot in enumerate(spots, 1):
                place = (f"<br><span>{spot['lat']:.6f}, {spot['lon']:.6f} &nbsp;&middot;&nbsp; "
                         f"<a href='{maps_link(spot['lat'], spot['lon'])}' target='_blank'>"
                         "open in Google Maps</a></span>" if "lat" in spot else "")
                st.markdown(
                    f"<div class='gs-spot'><b>{i}. About {spot['area']:,.1f} m² in the "
                    f"{spot['where']}</b><br><span>{spread_text(spot)}, about "
                    f"{spot['x_m']:,.0f} m right and {spot['y_m']:,.0f} m down from the "
                    f"top-left corner</span>{place}</div>",
                    unsafe_allow_html=True,
                )
        st.caption(f"Index: {INDEX_INFO[src]}")


def render_index_tab(res):
    names = [n for n in INDEX_ORDER if n in res["cells"]]
    choice = st.selectbox("Index", names, key="index_choice",
                          help="Choose which vegetation index to show on the grid.")
    st.caption(INDEX_INFO[choice] + " Grey areas have no usable data.")
    show_plots = bool(res.get("plots"))
    if show_plots:
        show_plots = st.checkbox("Show plot boundaries", value=True, key="index_plots")
    st.image(index_map_png(res["id"], choice, res["cells"][choice], res["rgb"],
                           res["plots"] if show_plots else None), width="stretch")


def render_overview_tab(res, meta):
    s = res["summary"]
    actions = list(res["flags"])
    st.image(overview_png(res["id"], s, actions), width="stretch")
    st.caption("The same spot can need more than one action, so these numbers can add up "
               "to more than 100%.")
    rows = []
    for a in actions:
        stat = s["actions"][a]
        rows.append({
            "Check": ACTIONS[a]["label"],
            "Based on": res["sources"][a],
            "Critical (m²)": round(stat["critical"] * meta["cell_area"], 1),
            "Moderate (m²)": round(stat["moderate"] * meta["cell_area"], 1),
            "Low (m²)": round(stat["low"] * meta["cell_area"], 1),
            "Total (m²)": round(stat["total"] * meta["cell_area"], 1),
            "% of field": round(stat["total"] / s["valid"] * 100, 2),
            "Priority": priority_of(stat, s["valid"]),
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={"% of field": st.column_config.NumberColumn(format="%.2f%%")},
    )
    skipped = [a for a, src in res["sources"].items() if src is None]
    if skipped:
        st.info("Not checked in this analysis: " + "; ".join(
            f"{ACTIONS[a]['label']} (needs {ACTIONS[a]['needs']})" for a in skipped))


def render_plots_tab(res, meta):
    plots = res["plots"]
    st.caption(f"The analysed area is split into {plots['rows']} x {plots['cols']} plots, "
               "named from the top-left. The score is the plot's own health score out of 100.")
    st.image(plot_map_png(res["id"], res["rgb"], plots), width="stretch")
    rows = plot_rows(res, meta)
    if not rows:
        st.info("No plot has usable data.")
        return
    st.markdown("**Plots, weakest first**")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={"Health score": st.column_config.NumberColumn(format="%.1f"),
                       "Healthy (%)": st.column_config.NumberColumn(format="%.1f%%"),
                       "Area (m²)": st.column_config.NumberColumn(format="%.1f"),
                       "Needs action (m²)": st.column_config.NumberColumn(format="%.1f")},
    )
    worst = rows[0]
    best = rows[-1]
    st.caption(f"Weakest plot: {worst['Plot']} at {worst['Health score']:.0f}/100. "
               f"Best plot: {best['Plot']} at {best['Health score']:.0f}/100.")

    with st.expander("Each check, plot by plot (m², with critical in brackets)"):
        table = []
        for i, label in enumerate(plots["labels"]):
            if int(plots["valid"][i]) == 0:
                continue
            row = {"Plot": label}
            for a in plots["action_total"]:
                total = plots["action_total"][a][i] * meta["cell_area"]
                crit = plots["action_critical"][a][i] * meta["cell_area"]
                row[ACTIONS[a]["label"]] = f"{total:,.1f} ({crit:,.1f})"
            table.append(row)
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Use this to see whether a plot has one clear problem or a mix. "
                   "The first number is the flagged area, the number in brackets is how much "
                   "of it is Critical.")


def render_cost_tab(res, meta):
    s = res["summary"]
    actions = list(res["flags"])
    st.caption("Enter your local rates. The cost is worked out from the area that needs "
               "each treatment.")
    col_in, col_out = st.columns([1, 1.15], gap="large")
    rates = {}
    with col_in:
        st.markdown("**Treatment rates (₹ per m²)**")
        c1, c2 = st.columns(2)
        for i, a in enumerate(actions):
            with (c1 if i % 2 == 0 else c2):
                rates[a] = kept_widget(
                    st.number_input, ACTIONS[a]["cost_label"], f"rate_{a}",
                    ACTIONS[a]["rate"], min_value=0.0, step=0.5,
                    help=f"Cost in ₹ to treat one square metre for {ACTIONS[a]['label'].lower()} "
                         "(material only, labour is added separately).",
                )
        c1, c2 = st.columns(2)
        with c1:
            labor = kept_widget(st.number_input, "Labour (₹ per m²)", "rate_labor", 1.5,
                                min_value=0.0, step=0.25, help=HELP["labor"])
        with c2:
            overhead = kept_widget(st.number_input, "Overhead (%)", "rate_overhead", 10.0,
                                   min_value=0.0, max_value=100.0, step=1.0, help=HELP["overhead"])
        st.markdown("**How much of each area will you treat?**")
        wc = kept_widget(st.slider, "Critical area treated (%)", "cover_c", 100,
                         min_value=0, max_value=100, step=5, help=HELP["cover_critical"])
        wm = kept_widget(st.slider, "Moderate area treated (%)", "cover_m", 75,
                         min_value=0, max_value=100, step=5, help=HELP["cover_moderate"])
        wl = kept_widget(st.slider, "Low area treated (%)", "cover_l", 50,
                         min_value=0, max_value=100, step=5, help=HELP["cover_low"])

    cost_rows = []
    for a in actions:
        stat = s["actions"][a]
        area = (stat["critical"] * wc + stat["moderate"] * wm + stat["low"] * wl) / 100 * meta["cell_area"]
        cost_rows.append({"action": a, "treatment": ACTIONS[a]["cost_label"],
                          "area_sqm": area, "cost": area * (rates[a] + labor)})
    subtotal = sum(r["cost"] for r in cost_rows)
    overhead_amt = subtotal * overhead / 100
    total = subtotal + overhead_amt
    st.session_state.cost = {
        "rows": cost_rows, "labor_per_sqm": labor, "overhead_pct": overhead,
        "coverage_pct": {"critical": wc, "moderate": wm, "low": wl},
        "subtotal": subtotal, "overhead": overhead_amt, "total": total,
    }

    with col_out:
        table = [{"Treatment": r["treatment"],
                  "Area treated (m²)": r["area_sqm"], "Cost (₹)": r["cost"]} for r in cost_rows]
        table.append({"Treatment": f"Overhead ({overhead:.0f}%)", "Area treated (m²)": None,
                      "Cost (₹)": overhead_amt})
        st.dataframe(
            pd.DataFrame(table), hide_index=True, width="stretch",
            column_config={"Area treated (m²)": st.column_config.NumberColumn(format="%.1f"),
                           "Cost (₹)": st.column_config.NumberColumn(format="₹ %.2f")},
        )
        st.markdown(f"<div class='gs-total'><div>Total estimated cost</div>"
                    f"<strong>₹{total:,.0f}</strong></div>", unsafe_allow_html=True)

        ranked = sorted((r for r in cost_rows if s["actions"][r["action"]]["total"] > 0),
                        key=lambda r: s["actions"][r["action"]]["critical"], reverse=True)
        if ranked:
            st.markdown("**Suggested order of work** (largest critical area first)")
            for i, r in enumerate(ranked, 1):
                stat = s["actions"][r["action"]]
                st.markdown(f"{i}. {ACTIONS[r['action']]['label']}: "
                            f"about {stat['critical'] * meta['cell_area']:,.1f} m² critical, "
                            f"around ₹{r['cost']:,.0f}")
        else:
            st.success("No treatment needed with the current limits.")


def slug(text, fallback="field"):
    cleaned = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return cleaned[:40] or fallback


def render_download_tab(res, meta):
    tag = f"{slug(meta.get('field_name'))}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    cost = st.session_state.cost
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c4:
        st.markdown("**Map pins (KML)**")
        kml = build_kml(res, meta)
        if kml:
            st.caption("Opens in Google Earth or any map app, with a pin on every critical patch.")
            st.download_button("Download KML", data=kml,
                               file_name=f"greenscan_{tag}_spots.kml",
                               mime="application/vnd.google-earth.kml+xml",
                               width="stretch", on_click="ignore")
        else:
            st.caption("Needs a photo with GPS data. Upload the original file from the drone "
                       "to get map pins.")
    with c1:
        st.markdown("**Full report (PDF)**")
        st.caption("Summary, cost estimate, notes, all maps and the main index map.")
        key = hashlib.sha1(json.dumps([res["id"], meta["notes"], cost], sort_keys=True,
                                      default=str).encode()).hexdigest()
        stored = st.session_state.pdf
        if stored and stored[0] == key:
            st.download_button("Download PDF", data=stored[1], file_name=f"greenscan_{tag}_report.pdf",
                               mime="application/pdf", type="primary",
                               width="stretch", on_click="ignore")
        elif st.button("Prepare PDF report", width="stretch"):
            with st.spinner("Preparing the PDF..."):
                try:
                    st.session_state.pdf = (key, build_pdf(res, meta, cost))
                except Exception as exc:
                    st.error("The PDF could not be created.")
                    with st.expander("Error details"):
                        st.exception(exc)
                    return
            st.rerun()
    with c2:
        st.markdown("**Grid data (CSV)**")
        st.caption("One row per grid square with index values and levels, for Excel or GIS.")
        st.download_button("Download CSV", data=build_csv(res["id"], res),
                           file_name=f"greenscan_{tag}_grid.csv", mime="text/csv",
                           width="stretch", on_click="ignore")
    with c3:
        st.markdown("**Summary (JSON)**")
        st.caption("Scores, settings and cost estimate for other software.")
        st.download_button("Download JSON", data=build_json(res, meta, cost),
                           file_name=f"greenscan_{tag}_summary.json", mime="application/json",
                           width="stretch", on_click="ignore")

    if meta["notes"].strip():
        st.divider()
        st.markdown("**Field notes**")
        st.markdown(f"<div class='gs-panel gs-notes'>{html.escape(meta['notes'])}</div>",
                    unsafe_allow_html=True)


def page_results():
    res = st.session_state.result
    meta = st.session_state.meta
    if res is None:
        st.markdown(
            "<div class='gs-hero'><h1>No report yet</h1>"
            "<p>Upload your field images and run the analysis. The report will appear here.</p></div>",
            unsafe_allow_html=True,
        )
        st.button("Go to upload", type="primary", on_click=go_to, args=(NAV_UPLOAD,))
        return

    render_report_header(res, meta)
    if res["mode"] == "rgb":
        st.info("This report uses only the RGB photo, so the results are approximate. "
                "Water stress, nitrogen and crop stress checks need multispectral bands.")
    if res.get("limit_mode") == "relative":
        pc, pm, pl = res["percents"]
        cuts = ", ".join(f"{ACTIONS[a]['label']} below {res['thresholds'][f'{a}:{src}'][0]:.2f} {src}"
                         for a, src in res["sources"].items()
                         if src and res["thresholds"].get(f"{a}:{src}"))
        st.info(f"Limits come from this field itself: weakest {pc}% marked Critical, "
                f"next {pm}% Moderate, next {pl}% Low. Critical cut-offs: {cuts}.")
    for note in res["notes"]:
        st.warning(note)
    if res.get("info"):
        with st.expander("How the images were processed"):
            for line in res["info"]:
                st.markdown(f"- {line}")

    actions = list(res["flags"])
    s = res["summary"]
    for col, a in zip(st.columns(len(actions)), actions):
        stat = s["actions"][a]
        col.metric(ACTIONS[a]["label"], f"{stat['total'] / s['valid'] * 100:.1f}%",
                   help=f"{stat['total'] * meta['cell_area']:,.1f} m² need action. "
                        f"Based on {res['sources'][a]}.",
                   border=True)

    labels = ([ACTIONS[a]["label"] for a in actions]
              + ["Index map", "Overview"]
              + (["Plots"] if res.get("plots") else [])
              + ["Cost estimate", "Download"])
    tabs = st.tabs(labels)
    for tab, a in zip(tabs, actions):
        with tab:
            render_action_tab(res, meta, a)
    n = len(actions)
    with tabs[n]:
        render_index_tab(res)
    with tabs[n + 1]:
        render_overview_tab(res, meta)
    n += 2
    if res.get("plots"):
        with tabs[n]:
            render_plots_tab(res, meta)
        n += 1
    with tabs[n]:
        render_cost_tab(res, meta)
    with tabs[n + 1]:
        render_download_tab(res, meta)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_state()
    progress = render_sidebar()
    if st.session_state.nav == NAV_RESULTS:
        page_results()
    else:
        page_upload(progress)
    render_footer()


main()