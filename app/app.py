import os
from pathlib import Path
from functools import reduce

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import cm

DATA_DIR = Path(__file__).absolute().parent / "data"

@st.cache
def load_ann_codes():
    return {
        "Malignancy": {
            1: "Highly Unlikely",
            2: "Moderately Unlikely",
            3: "Indeterminate",
            4: "Moderately Suspicious",
            5: "Highly Suspicious",
        }
    }

@st.cache
def load_meta():
    scan_meta_path = DATA_DIR / "scan_meta.csv"
    nod_meta_path = DATA_DIR / "nodule_meta.csv"
    
    if not scan_meta_path.exists() or not nod_meta_path.exists():
        st.error("Metadata files are missing!")
        st.stop()
    
    scan_df = pd.read_csv(scan_meta_path)
    nod_df = pd.read_csv(nod_meta_path)
    return scan_df, nod_df

@st.cache
def load_raw_img(pid):
    img_path = DATA_DIR / pid / "scan.npy"
    if not img_path.exists():
        st.error(f"CT scan file not found for patient '{pid}'")
        st.stop()
    return np.load(img_path)

@st.cache
def load_mask(pid):
    fnames = sorted((DATA_DIR / pid).glob('*_mask.npy'))
    if not fnames:
        st.warning(f"No mask files found for patient '{pid}'")
        return np.zeros((512, 512, 100), dtype=bool)
    masks = [np.load(fname) for fname in fnames]
    return reduce(np.logical_or, masks)

@st.cache
def load_nodule_img(pid, nid):
    img_path = DATA_DIR / pid / f"nodule_{nid:02d}_vol.npy"
    if not img_path.exists():
        st.warning(f"Nodule file not found for patient '{pid}', nodule '{nid}'")
        return np.zeros((64, 64, 64))
    return np.load(img_path)

@st.cache
def get_img_slice(img, z, window=(-600, 1500)):
    level, width = window
    img = np.clip(img, level - (width / 2), level + (width / 2))
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min)
    img_slice = img[:, :, z]
    return Image.fromarray(np.uint8(cm.gray(img_slice) * 255)).convert('RGBA')

@st.cache
def get_mask_slice(mask, z):
    return Image.fromarray((mask[:, :, z] * 96).astype(np.uint8), mode='L')

@st.cache
def get_overlay():
    arr = np.zeros((512, 512, 4), dtype=np.uint8)
    arr[:, :, 1] = 128
    arr[:, :, 3] = 128
    return Image.fromarray(arr, mode='RGBA')

# Load metadata
scan_df, nod_df = load_meta()

# Select Patient
patients = scan_df["PatientID"].unique().tolist()
pid = st.selectbox("Select Patient ID:", patients, index=0)

# Filter metadata for selected patient
scan = scan_df[scan_df["PatientID"] == pid].iloc[0]
nodules = nod_df[nod_df["PatientID"] == pid]

# Load Images
img_arr = load_raw_img(pid)
mask_arr = load_mask(pid)

codes = load_ann_codes()

st.header("Selected case for lung cancer detection application")
st.subheader("Patient Information")

st.write(f"**Patient ID:** {scan.PatientID}")
st.write(f"**Diagnosis:** {scan.Diagnosis}")
st.write(f"**Diagnosis Method:** {scan.DiagnosisMethod}")

st.subheader("CT Scan")

img_placeholder = st.empty()

col1, col2 = st.columns(2)
with col1:
    overlay_nodules = st.checkbox("Show nodule overlay", value=True)
    z = st.slider("Slice:", min_value=1, max_value=img_arr.shape[2], value=int(img_arr.shape[2]/2))
    level = st.number_input("Window level:", value=-600)
    width = st.number_input("Window width:", value=1500)

img = get_img_slice(img_arr, z - 1, window=(level, width))

if overlay_nodules:
    mask = get_mask_slice(mask_arr, z - 1)
    overlay = get_overlay()
    img_placeholder.image(Image.composite(overlay, img, mask), use_column_width=True)
else:
    img_placeholder.image(img, use_column_width=True)

st.subheader("Detected Nodules")

if nodules.empty:
    st.write("No nodules found for this patient.")
else:
    for _, nodule in nodules.iterrows():
        col1, col2, col3 = st.columns([1, 2, 3])
        with col1:
            img_arr = load_nodule_img(pid, nodule.NoduleID)
            img = get_img_slice(img_arr, img_arr.shape[2] // 2)
            st.image(img, caption=f"Nodule #{nodule.NoduleID}")
        with col2:
            st.write(f"**Diameter:** {nodule.Diameter:.2f} mm")
            st.write(f"**Surface Area:** {nodule.SurfaceArea:.2f} mm²")
            st.write(f"**Volume:** {nodule.Volume:.2f} mm³")
        with col3:
            malignancy_label = codes["Malignancy"].get(int(nodule.Malignancy), "Unknown")
            st.write(f"**Pred. Malignancy:** {malignancy_label}")
