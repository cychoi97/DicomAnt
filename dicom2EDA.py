#!/usr/bin/env python3
"""
dicom2EDA.py  (v1.0)

All-in-one DICOM metadata extraction + automated EDA.

Directory model assumed:
  <root>/
    <patient_dir>/          ← depth controlled by --patient-depth (default 1)
      <any subdirs>/
        *.dcm               ← multiple series can coexist in the same folder

Series are identified purely from DICOM headers:
  primary key  : SeriesInstanceUID
  human labels : SeriesNumber, SeriesDescription, ProtocolName

Pipeline:
  1. Discover all DICOM files under --dicom-dir
  2. Quick-read headers → build (patient_folder, series_uid) index
  3. Extract full metadata → DataFrame with derived columns:
       patient_folder, series_uid_label, series_key,
       SeriesSliceCount, SlicePositionInSeries (%)
  4. Save CSV / Parquet
  5. Generate EDA PDF + individual PNGs

EDA sections:
  A  Dataset overview table
  B  Missing-value rate per column
  C  Categorical feature distributions
  D  Numeric feature distributions
  E  Image geometry (Rows x Columns, BitsStored)
  F  Series-level analysis
       F1 - slice count histogram per series
       F2 - series count per patient
       F3 - top SeriesDescription distribution
       F4 - modality distribution per series
  G  Patient-level composition
       G1 - modality combinations per patient
       G2 - series count per patient box
  H  Patient demographics (Sex pie, Age histogram)
  I  SeriesDescription Gallery

Usage:
  python dicom2EDA.py --dicom-dir /path/to/root --out-dir ./eda_output
  python dicom2EDA.py --dicom-dir /path/to/root --out-dir ./eda_output --patient-depth 2 --keep-phi
  python dicom2EDA.py --dicom-dir /path/to/root --out-dir ./eda_output --one-slice-per-series --rep-policy auto

Dependencies:
  pip install pydicom pandas tqdm natsort matplotlib seaborn pyarrow
"""

import os
import sys
import hashlib
import argparse
import warnings
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    warnings.warn("seaborn not found - falling back to plain matplotlib.")

from tqdm import tqdm
from natsort import natsorted
from pydicom import dcmread


# ============================================================
# Tag definitions
# ============================================================

# Tags that are safe to store as-is (no PHI risk)
SAFE_TAGS = [
    ("Modality",                  str),
    ("BodyPartExamined",          str),
    ("ViewPosition",              str),
    ("PatientPosition",           str),
    ("Manufacturer",              str),
    ("ManufacturerModelName",     str),
    # Series-level identifiers (non-PHI)
    ("SeriesNumber",              int),
    ("SeriesDescription",         str),
    ("ProtocolName",              str),
    ("StudyDescription",          str),
    ("ImageType",                 str),   # multi-value → joined as string
    # Image geometry
    ("Rows",                      int),
    ("Columns",                   int),
    ("BitsStored",                int),
    ("PhotometricInterpretation", str),
    ("BurnedInAnnotation",        str),
    # Acquisition parameters
    ("KVP",                       float),
    ("Exposure",                  float),
    ("ExposureTime",              float),
    ("XRayTubeCurrent",           float),
    ("SliceThickness",            float),
    ("SpacingBetweenSlices",      float),
    ("ConvolutionKernel",         str),
    ("PixelSpacing",              list),        # flattened → PixelSpacingRow/Col
    ("ImagerPixelSpacing",        list),        # flattened → ImagerPixelSpacingRow/Col
    ("ReconstructionDiameter",    float),
    ("WindowCenter",              str),         # may be multi-value
    ("WindowWidth",               str),
]

# Tags with PHI risk – hashed by default
PHI_LIKE_TAGS = [
    ("PatientID",         str),
    ("PatientBirthDate",  str),
    ("PatientSex",        str),
    ("PatientAge",        str),   # e.g. '063Y'
    ("StudyInstanceUID",  str),
    ("SeriesInstanceUID", str),
    ("SOPInstanceUID",    str),
    ("AccessionNumber",   str),
    ("StudyDate",         str),
    ("StudyTime",         str),
    ("SeriesDate",        str),
    ("SeriesTime",        str),
]

# Numeric columns used in distribution / correlation plots
NUMERIC_COLS = [
    "filesize_bytes", "SeriesNumber",
    "Rows", "Columns", "BitsStored",
    "KVP", "Exposure", "ExposureTime", "XRayTubeCurrent",
    "SliceThickness", "SpacingBetweenSlices",
    "ReconstructionDiameter",
    "PixelSpacingRow", "PixelSpacingCol",
    "ImagerPixelSpacingRow", "ImagerPixelSpacingCol",
    "SlicePositionInSeries",
    "SeriesSliceCount",
]

# Categorical columns plotted in Section C
CATEGORICAL_COLS = [
    "Modality", "SeriesDescription", "ProtocolName", "StudyDescription",
    "BodyPartExamined", "ViewPosition", "PatientPosition",
    "Manufacturer", "ManufacturerModelName", "ConvolutionKernel",
    "PhotometricInterpretation", "BurnedInAnnotation",
    "ImageType", "PatientSex",
]

PALETTE = "Set2"   # seaborn palette


# ============================================================
# DICOM utilities
# ============================================================

def _hash(v):
    """SHA-256, truncated to 16 hex chars."""
    if v is None:
        return None
    return hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:16]


def _safe_get(ds, name, default=None):
    """Read a DICOM attribute with silent fallback."""
    try:
        v = getattr(ds, name)
    except Exception:
        return default
    return default if v is None else v


def extract_metadata(dcm_path: str, keep_phi: bool = False) -> dict:
    """
    Read DICOM header (pixels skipped) and return a flat metadata dict.
    Multi-value tags are joined to strings; PixelSpacing is flattened.
    """
    ds = dcmread(dcm_path, force=True, stop_before_pixels=True)
    out = {"path": dcm_path}

    # File size on disk
    try:
        out["filesize_bytes"] = os.path.getsize(dcm_path)
    except OSError:
        out["filesize_bytes"] = None

    # ---- Safe tags ----
    for tag, caster in SAFE_TAGS:
        val = _safe_get(ds, tag, None)
        if val is None:
            out[tag] = None
        elif tag in ("PixelSpacing", "ImagerPixelSpacing"):
            try:
                out[tag] = list(val)
            except Exception:
                out[tag] = None
        elif tag == "ImageType":
            # ImageType is a list like ['ORIGINAL','PRIMARY','AXIAL']
            try:
                out[tag] = "\\".join(str(x) for x in val)
            except Exception:
                try:
                    out[tag] = str(val)
                except Exception:
                    out[tag] = None
        else:
            try:
                out[tag] = caster(val)
            except Exception:
                out[tag] = str(val)

    # ---- PHI-like tags ----
    for tag, caster in PHI_LIKE_TAGS:
        val = _safe_get(ds, tag, None)
        if val is None:
            out[tag] = None
        else:
            v = str(val)
            if keep_phi:
                try:
                    out[tag] = caster(v)
                except Exception:
                    out[tag] = v
            else:
                if tag.endswith("UID") or tag in ("PatientID", "AccessionNumber"):
                    out[tag] = _hash(v)
                elif tag in ("StudyDate", "SeriesDate", "PatientBirthDate") and len(v) >= 6:
                    out[tag] = v[:6]   # YYYYMM
                elif tag in ("StudyTime", "SeriesTime") and len(v) >= 4:
                    out[tag] = v[:4]   # HHMM
                else:
                    out[tag] = v       # PatientSex, PatientAge – low risk

    # ---- Flatten PixelSpacing ----
    px = out.get("PixelSpacing")
    if isinstance(px, list) and len(px) >= 2:
        out["PixelSpacingRow"] = _to_float(px[0])
        out["PixelSpacingCol"] = _to_float(px[1])
    else:
        out["PixelSpacingRow"] = None
        out["PixelSpacingCol"] = None

    # ---- Flatten ImagerPixelSpacing ----
    ipx = out.get("ImagerPixelSpacing")
    if isinstance(ipx, list) and len(ipx) >= 2:
        out["ImagerPixelSpacingRow"] = _to_float(ipx[0])
        out["ImagerPixelSpacingCol"] = _to_float(ipx[1])
    else:
        out["ImagerPixelSpacingRow"] = None
        out["ImagerPixelSpacingCol"] = None

    return out


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


# ============================================================
# File scanning
# ============================================================

def scan_dicoms(root: str) -> list:
    """Recursively find DICOM files (.dcm, .dicom, or no extension)."""
    paths = []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            lower = f.lower()
            if lower.endswith((".dcm", ".dicom")) or "." not in f:
                paths.append(os.path.join(dp, f))
    return natsorted(paths)


# ============================================================
# Patient folder extraction
# ============================================================

def patient_folder_for_path(path: str, root: str, depth: int) -> str:
    """
    Return the sub-path at 'depth' levels below root as the patient identifier.
    Example:  root=/data  depth=1  path=/data/PAT001/CT/1.dcm  → 'PAT001'
              root=/data  depth=2  path=/data/site1/PAT001/1.dcm → 'site1/PAT001'
    If the file is shallower than depth, returns the root basename.
    """
    rel   = os.path.relpath(path, root)
    parts = rel.split(os.sep)          # ['PAT001', 'CT', '1.dcm']
    if len(parts) <= depth:
        # File is at or within the patient folder itself
        return os.sep.join(parts[:-1]) if len(parts) > 1 else os.path.basename(root)
    return os.sep.join(parts[:depth])  # Take first `depth` path components


# ============================================================
# Series grouping using DICOM headers
# ============================================================

def _read_series_key(path: str) -> tuple:
    """
    Return (patient_id_raw, study_uid, series_uid, series_number, series_desc)
    from DICOM header with graceful fallback.
    """
    try:
        ds = dcmread(path, force=True, stop_before_pixels=True)
        patient_id   = str(_safe_get(ds, "PatientID",         "NA"))
        study_uid    = str(_safe_get(ds, "StudyInstanceUID",  "NA"))
        series_uid   = str(_safe_get(ds, "SeriesInstanceUID", os.path.basename(path)))
        series_num   = _safe_get(ds, "SeriesNumber",      None)
        series_desc  = str(_safe_get(ds, "SeriesDescription", ""))
        return (patient_id, study_uid, series_uid, series_num, series_desc)
    except Exception:
        return ("NA", "NA", os.path.basename(path), None, "")


def build_series_index(paths: list, root: str, patient_depth: int) -> dict:
    """
    Quick-pass over all files to build:
      index[path] = {
          'patient_folder': str,
          'series_uid':     str,
          'series_number':  int|None,
          'series_desc':    str,
          'study_uid':      str,
      }
    Uses tqdm for progress display.
    """
    index = {}
    print("  [Index] Reading series keys from DICOM headers …")
    for p in tqdm(paths, desc="Building series index"):
        pat_id, study_uid, series_uid, series_num, series_desc = _read_series_key(p)
        index[p] = {
            "patient_folder": patient_folder_for_path(p, root, patient_depth),
            "series_uid":     series_uid,
            "series_number":  series_num,
            "series_desc":    series_desc,
            "study_uid":      study_uid,
        }
    return index


# ============================================================
# SlicePosition within series  (header-based, then position-based)
# ============================================================

def _series_sort_key(ds) -> tuple:
    """Sort tuple: (InstanceNumber, z, SliceLocation, filename)."""
    instance = _safe_get(ds, "InstanceNumber", None)
    try:
        instance = int(instance) if instance is not None else 10**9
    except Exception:
        instance = 10**9

    ipp = _safe_get(ds, "ImagePositionPatient", None)
    if isinstance(ipp, (list, tuple)) and len(ipp) >= 3 and ipp[2] is not None:
        try:
            z = float(ipp[2])
        except Exception:
            z = float("inf")
    else:
        sl = _safe_get(ds, "SliceLocation", None)
        try:
            z = float(sl) if sl is not None else float("inf")
        except Exception:
            z = float("inf")

    sl_v = _safe_get(ds, "SliceLocation", None)
    try:
        sl_v = float(sl_v) if sl_v is not None else float("inf")
    except Exception:
        sl_v = float("inf")

    fname = getattr(ds, "filename", "") or ""
    return (instance, z, sl_v, str(fname))


def compute_slice_position_in_series(paths: list, series_index: dict) -> dict:
    """
    Group files by (patient_folder, series_uid), sort by natural filename order,
    and assign SlicePositionInSeries as (rank+1)/N * 100 %.
    No pixel data is loaded.
    """
    # Group paths by series
    series_groups: dict[tuple, list] = defaultdict(list)
    for p in paths:
        info = series_index.get(p, {})
        key  = (info.get("patient_folder", "?"), info.get("series_uid", "?"))
        series_groups[key].append(p)

    pos_map = {}
    for key, plist in series_groups.items():
        plist_sorted = natsorted(plist)
        n = len(plist_sorted)
        for idx, p in enumerate(plist_sorted):
            pos_map[p] = round(((idx + 1) / n) * 100.0, 1)
    return pos_map


# ============================================================
# Representative slice selection  (for --one-slice-per-series)
# ============================================================

def choose_representatives(series_groups: dict, rep_policy: str = "auto") -> tuple:
    """
    series_groups: { (patient_folder, series_uid): [path, ...] }
    Returns: (selected_paths, slice_count_map)
    """
    selected     = []
    slice_counts = {}

    for key, plist in series_groups.items():
        n = len(plist)
        if n == 1:
            selected.append(plist[0])
            slice_counts[plist[0]] = 1
            continue

        if rep_policy == "first":
            chosen = natsorted(plist)[0]
        elif rep_policy == "last":
            chosen = natsorted(plist)[-1]
        else:
            # auto / middle – load minimal headers for proper ordering
            headers = []
            for p in plist:
                try:
                    ds = dcmread(p, force=True, stop_before_pixels=True)
                    headers.append((p, ds))
                except Exception:
                    headers.append((p, None))

            def sortkey(x):
                p, ds = x
                return (10**9, float("inf"), float("inf"), p) if ds is None else _series_sort_key(ds)

            headers_sorted = sorted(headers, key=sortkey)
            chosen = headers_sorted[len(headers_sorted) // 2][0]

        selected.append(chosen)
        slice_counts[chosen] = n

    return selected, slice_counts


# ============================================================
# Build DataFrame
# ============================================================

def build_dataframe(
    root:           str,
    paths:          list,
    series_index:   dict,
    slice_pos_map:  dict,
    one_slice:      bool,
    rep_policy:     str,
    keep_phi:       bool,
) -> pd.DataFrame:
    """
    Extract full metadata for each path and assemble the master DataFrame.
    Adds derived columns:
      patient_folder, series_uid_label, series_key,
      SeriesSliceCount, SlicePositionInSeries
    """
    # Build series groups for optional reduction
    series_groups: dict[tuple, list] = defaultdict(list)
    for p in paths:
        info = series_index.get(p, {})
        key  = (info.get("patient_folder", "?"), info.get("series_uid", "?"))
        series_groups[key].append(p)

    slice_counts_map = {}
    if one_slice:
        selected, slice_counts_map = choose_representatives(series_groups, rep_policy)
        # Preserve series slice counts for ALL paths (not just selected ones)
        # so that the count is available as a column value
        full_counts = {}
        for key, plist in series_groups.items():
            for p in plist:
                full_counts[p] = len(plist)
        # Replace paths with representative only
        paths = selected
        print(f"  [Series reduction] "
              f"{len(series_groups)} series → {len(paths)} representative slices")

    rows = []
    for p in tqdm(paths, desc="Extracting DICOM metadata"):
        info = series_index.get(p, {})
        pat  = info.get("patient_folder", "?")
        suid = info.get("series_uid",     "?")
        snum = info.get("series_number",  None)
        sdesc= info.get("series_desc",    "")

        try:
            row = extract_metadata(p, keep_phi=keep_phi)
        except Exception as exc:
            row = {"path": p, "error": str(exc)}

        # Derived structural columns
        row["patient_folder"] = pat
        row["series_uid"]     = suid

        # Human-readable series key:  "<SeriesNumber> – <SeriesDescription>"
        label_parts = []
        if snum is not None:
            label_parts.append(str(snum))
        if sdesc:
            label_parts.append(sdesc)
        row["series_label"] = " – ".join(label_parts) if label_parts else suid[:12]

        # Global series key for grouping:  patient_folder::series_uid
        row["series_key"] = f"{pat}::{suid}"

        # Slice position within the series (folder-rank based)
        row["SlicePositionInSeries"] = slice_pos_map.get(p)

        # Slice count for the series this file belongs to
        if one_slice:
            row["SeriesSliceCount"] = slice_counts_map.get(p, 1)
        else:
            # Even without reduction, annotate how many slices are in the series
            key = (pat, suid)
            row["SeriesSliceCount"] = len(series_groups.get(key, [p]))

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Column ordering ----
    priority = [
        "path", "patient_folder", "series_label", "series_uid", "series_key",
        "filesize_bytes", "SeriesSliceCount", "SlicePositionInSeries",
        "Modality", "SeriesNumber", "SeriesDescription", "ProtocolName",
        "StudyDescription", "BodyPartExamined", "ViewPosition", "PatientPosition",
        "Manufacturer", "ManufacturerModelName",
        "Rows", "Columns", "BitsStored", "PhotometricInterpretation",
        "BurnedInAnnotation", "ImageType",
        "KVP", "Exposure", "ExposureTime", "XRayTubeCurrent",
        "SliceThickness", "SpacingBetweenSlices", "ConvolutionKernel",
        "ReconstructionDiameter", "WindowCenter", "WindowWidth",
        "PixelSpacingRow", "PixelSpacingCol",
        "ImagerPixelSpacingRow", "ImagerPixelSpacingCol",
        "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
        "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "AccessionNumber",
        "StudyDate", "StudyTime", "SeriesDate", "SeriesTime",
    ]
    have  = [c for c in priority if c in df.columns]
    rest  = [c for c in df.columns if c not in have]
    df    = df[have + rest]

    return df


# ============================================================
# Plot helpers
# ============================================================

def _fig_title(fig, title, fontsize=13):
    fig.suptitle(title, fontsize=fontsize, fontweight="bold", y=1.02)


def _save_fig(fig, pdf, out_dir, stem):
    png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _barh(ax, vc, max_items=20):
    """Horizontal bar chart helper (top-N categories)."""
    vc = vc.head(max_items)
    if HAS_SNS:
        palette = sns.color_palette(PALETTE, len(vc))
        ax.barh(vc.index[::-1], vc.values[::-1], color=palette[::-1])
    else:
        ax.barh(vc.index[::-1], vc.values[::-1])
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)


# ============================================================
# EDA Section A – Dataset overview
# ============================================================

def plot_overview(df, pdf, out_dir):
    """A: High-level dataset overview table.

    Layout: widescreen (18 x 5.5 in) with a two-column table so the
    figure fills a 16:9 PPT slide without excessive white space.
    """
    total   = len(df)
    n_err   = int(df["error"].notna().sum()) if "error" in df.columns else 0
    valid   = total - n_err
    size_gb = df["filesize_bytes"].sum() / 1e9 if "filesize_bytes" in df.columns else float("nan")

    n_patients = df["patient_folder"].nunique()  if "patient_folder" in df.columns else "N/A"
    n_series   = df["series_key"].nunique()      if "series_key"     in df.columns else "N/A"
    n_studies  = df["StudyInstanceUID"].nunique() if "StudyInstanceUID" in df.columns else "N/A"
    n_mods     = df["Modality"].nunique()         if "Modality" in df.columns else "N/A"

    # Average series per patient
    if "patient_folder" in df.columns and "series_key" in df.columns:
        spp = df.groupby("patient_folder")["series_key"].nunique()
        avg_spp = f"{spp.mean():.1f} (min {spp.min()} - max {spp.max()})"
    else:
        avg_spp = "N/A"

    # Split into two side-by-side sub-tables for wider layout
    left_data = [
        ["Total DICOM files scanned",     f"{total:,}"],
        ["  - valid (parsed)",             f"{valid:,}"],
        ["  - parse errors",               f"{n_err:,}"],
        ["Total size on disk",             f"{size_gb:.2f} GB"],
        ["Columns in DataFrame",           str(len(df.columns))],
    ]
    right_data = [
        ["Unique patients (folder-based)", str(n_patients)],
        ["Unique studies",                 str(n_studies)],
        ["Unique series (by SeriesUID)",   str(n_series)],
        ["Unique modalities",              str(n_mods)],
        ["Avg series per patient",         str(avg_spp)],
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18.0, 5.5))

    def _render_table(ax, data, title):
        ax.axis("off")
        tbl = ax.table(
            cellText  = data,
            colLabels = [title, "Value"],
            cellLoc   = "left",
            loc       = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.3, 2.0)
        for col in (0, 1):
            tbl[0, col].set_facecolor("#2D4059")
            tbl[0, col].set_text_props(color="white", fontweight="bold")

    _render_table(ax_l, left_data,  "Dataset")
    _render_table(ax_r, right_data, "Counts")

    _fig_title(fig, "A · Dataset Overview", fontsize=13)
    _save_fig(fig, pdf, out_dir, "A_overview")
    print("  [A] Dataset overview saved.")


# ============================================================
# EDA Section B – Missing values
# ============================================================

def plot_missing_values(df, pdf, out_dir):
    """B: Missing-value rate per column (horizontal bar chart).

    Layout: fixed 18 x 7 in widescreen; up to 40 columns shown,
    bars drawn on the left half with annotation labels on the right.
    """
    # Exclude structural / derived columns from "expected" missing analysis
    skip = {"path", "patient_folder", "series_uid", "series_key",
            "series_label", "error", "PixelSpacing", "ImagerPixelSpacing"}
    cols = [c for c in df.columns if c not in skip]
    miss = df[cols].isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    if miss.empty:
        print("  [B] No missing values found - skipping.")
        return

    # Cap at 40 columns; fixed widescreen figure
    miss   = miss.head(40)
    fig, ax = plt.subplots(figsize=(18.0, 7.0))

    colors = ["#E84545" if v > 0.5 else "#F7A440" if v > 0.2 else "#4B8BBE"
              for v in miss.values]
    bars = ax.barh(miss.index[::-1], miss.values[::-1] * 100,
                   color=colors[::-1], height=0.65)
    ax.set_xlabel("Missing (%)", fontsize=10)
    ax.set_xlim(0, 118)
    ax.tick_params(axis="y", labelsize=8.5)
    for bar, val in zip(bars, miss.values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    _fig_title(fig, "B · Missing Value Rate per Column", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, pdf, out_dir, "B_missing_values")
    print("  [B] Missing value heatmap saved.")


# ============================================================
# EDA Section C – Categorical distributions
# ============================================================

def plot_categorical(df, pdf, out_dir):
    """C: Horizontal bar charts for key categorical columns.

    Layout: 4 columns across a widescreen figure (16:9 aspect ratio)
    so the output fits naturally on a standard PPT slide.
    """
    cats = [c for c in CATEGORICAL_COLS if c in df.columns and df[c].notna().any()]
    if not cats:
        print("  [C] No categorical columns found.")
        return

    # 4-column wide layout: each row is compact, figure grows vertically only
    # as needed.  Target ~2.8 in per row so the total height stays <= 10 in.
    ncols     = 4
    nrows     = math.ceil(len(cats) / ncols)
    row_h_in  = 2.8          # height per row (inches)
    fig_w_in  = 18.0         # fixed wide width (≈ PPT 16:9 scale)
    fig_h_in  = max(3.5, nrows * row_h_in)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w_in, fig_h_in))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cats):
        ax = axes[i]
        vc = df[col].value_counts()
        _barh(ax, vc, max_items=15)   # keep bars readable at smaller size
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7.5)

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _fig_title(fig, "C · Categorical Feature Distributions", fontsize=13)
    _save_fig(fig, pdf, out_dir, "C_categorical")
    print("  [C] Categorical distributions saved.")


# ============================================================
# EDA Section D – Numeric distributions
# ============================================================

def plot_numeric_dist(df, pdf, out_dir):
    """D: Histogram + KDE for each numeric column (1st-99th pct clipped).

    Layout: 5 columns in a single wide figure so every histogram fits
    in a PPT-friendly landscape frame without excessive vertical space.
    """
    nums = [c for c in NUMERIC_COLS if c in df.columns and df[c].notna().any()]
    nums = list(dict.fromkeys(nums))

    if not nums:
        print("  [D] No numeric columns found.")
        return

    # 5-column widescreen layout – row height kept compact
    ncols     = 5
    nrows     = math.ceil(len(nums) / ncols)
    row_h_in  = 2.8          # compact row height
    fig_w_in  = 18.0         # wide fixed width
    fig_h_in  = max(3.0, nrows * row_h_in)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w_in, fig_h_in))
    axes = np.array(axes).flatten()

    for i, col in enumerate(nums):
        ax     = axes[i]
        series = df[col].dropna()
        if series.empty:
            ax.set_visible(False)
            continue

        lo, hi  = series.quantile(0.01), series.quantile(0.99)
        clipped = series.clip(lo, hi)

        if HAS_SNS:
            sns.histplot(clipped, kde=True, ax=ax, color="#4B8BBE",
                         edgecolor="white", linewidth=0.4)
        else:
            ax.hist(clipped, bins=30, color="#4B8BBE",
                    edgecolor="white", linewidth=0.4)

        med = series.median()
        std = series.std()
        ax.axvline(med, color="red", linestyle="--", linewidth=1.2,
                   label=f"med={med:.3g}")
        ax.legend(fontsize=6.5)
        ax.set_title(f"{col}\n(n={len(series):,}  σ={std:.3g})",
                     fontsize=7.5, fontweight="bold")
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(nums), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _fig_title(fig, "D · Numeric Feature Distributions  (1st-99th pct clipped for display)",
               fontsize=12)
    _save_fig(fig, pdf, out_dir, "D_numeric_distributions")
    print("  [D] Numeric distributions saved.")


# ============================================================
# EDA Section E – Image geometry
# ============================================================

def plot_image_geometry(df, pdf, out_dir):
    """E: Rows x Columns hexbin scatter + BitsStored distribution.

    Layout: 18 x 6 in widescreen; hexbin scatter takes 70 % of width.
    """
    fig = plt.figure(figsize=(18.0, 6.0))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                            width_ratios=[2.0, 1])

    # E1 – Rows x Columns
    ax1 = fig.add_subplot(gs[0])
    if "Rows" in df.columns and "Columns" in df.columns:
        sub = df[["Rows", "Columns"]].dropna()
        if not sub.empty:
            if HAS_SNS:
                hb = ax1.hexbin(sub["Columns"], sub["Rows"],
                                gridsize=30, cmap="Blues", mincnt=1)
                plt.colorbar(hb, ax=ax1, label="Count")
            else:
                ax1.scatter(sub["Columns"], sub["Rows"],
                            alpha=0.3, s=10, color="#4B8BBE")
            ax1.set_xlabel("Columns"); ax1.set_ylabel("Rows")
            ax1.set_title("Image Size (Rows x Columns)", fontweight="bold")
            ax1.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center")

    # E2 - BitsStored
    ax2 = fig.add_subplot(gs[1])
    if "BitsStored" in df.columns and df["BitsStored"].notna().any():
        vc = df["BitsStored"].value_counts().sort_index()
        ax2.bar(vc.index.astype(str), vc.values, color="#E84545", edgecolor="white")
        ax2.set_xlabel("BitsStored"); ax2.set_ylabel("Count")
        ax2.set_title("BitsStored Distribution", fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        for xi, yi in enumerate(vc.values):
            ax2.text(xi, yi + max(vc.values) * 0.01, str(yi),
                     ha="center", fontsize=9)

    _fig_title(fig, "E · Image Geometry Summary")
    _save_fig(fig, pdf, out_dir, "E_image_geometry")
    print("  [E] Image geometry saved.")


# ============================================================
# EDA Section F – Series-level analysis
# ============================================================

def plot_series_analysis(df, pdf, out_dir):
    """
    F: Series-level analysis (4 sub-panels):
      F1 - Distribution of slice count per series
      F2 - Distribution of series count per patient
      F3 - Top-20 SeriesDescription frequency
      F4 - Modality distribution at series level
    """
    # 1-row x 4-col widescreen layout (18 x 7 in) so all sub-panels
    # fit on a single 16:9 PPT slide without vertical stacking.
    fig = plt.figure(figsize=(18.0, 7.0))
    gs  = gridspec.GridSpec(1, 4, figure=fig, hspace=0.0, wspace=0.38)

    # ----- F1: Slice count per series -----
    ax1 = fig.add_subplot(gs[0])
    if "SeriesSliceCount" in df.columns and df["SeriesSliceCount"].notna().any():
        ser_df = df.drop_duplicates(subset="series_key")[["series_key", "SeriesSliceCount"]] \
                   if "series_key" in df.columns else df[["SeriesSliceCount"]]
        counts = ser_df["SeriesSliceCount"].dropna()
        if HAS_SNS:
            sns.histplot(counts, bins=30, kde=False, ax=ax1,
                         color="#2ECC71", edgecolor="white")
        else:
            ax1.hist(counts, bins=30, color="#2ECC71", edgecolor="white")
        med = counts.median()
        ax1.axvline(med, color="red", linestyle="--", linewidth=1.5,
                    label=f"Median = {med:.0f}")
        ax1.legend(fontsize=8)
        ax1.set_xlabel("Slices per Series", fontsize=9)
        ax1.set_ylabel("Series Count", fontsize=9)
        ax1.set_title("F1 · Slice Count\nper Series", fontweight="bold", fontsize=9)
        ax1.grid(axis="y", alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "SeriesSliceCount N/A", ha="center", va="center")

    # ----- F2: Series count per patient -----
    ax2 = fig.add_subplot(gs[1])
    if "patient_folder" in df.columns and "series_key" in df.columns:
        spp = df.groupby("patient_folder")["series_key"].nunique()
        if HAS_SNS:
            sns.histplot(spp, bins=min(30, spp.nunique()), kde=False,
                         ax=ax2, color="#3498DB", edgecolor="white")
        else:
            ax2.hist(spp, bins=min(30, spp.nunique()),
                     color="#3498DB", edgecolor="white")
        med = spp.median()
        ax2.axvline(med, color="red", linestyle="--", linewidth=1.5,
                    label=f"Median = {med:.0f}")
        ax2.legend(fontsize=8)
        ax2.set_xlabel("Series per Patient", fontsize=9)
        ax2.set_ylabel("Patient Count", fontsize=9)
        ax2.set_title("F2 · Series Count\nper Patient", fontweight="bold", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "patient_folder N/A", ha="center", va="center")

    # ----- F3: Top-15 SeriesDescription -----
    ax3 = fig.add_subplot(gs[2])
    if "SeriesDescription" in df.columns and df["SeriesDescription"].notna().any():
        if "series_key" in df.columns:
            desc_df = df.drop_duplicates(subset="series_key")[["series_key", "SeriesDescription"]]
        else:
            desc_df = df[["SeriesDescription"]]
        vc = desc_df["SeriesDescription"].value_counts().head(15)
        _barh(ax3, vc, max_items=15)
        ax3.tick_params(axis="y", labelsize=7.5)
        ax3.set_title("F3 · Top-15 Series\nDescriptions (series-level)",
                      fontweight="bold", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "SeriesDescription N/A", ha="center", va="center")

    # ----- F4: Modality distribution at series level -----
    ax4 = fig.add_subplot(gs[3])
    if "Modality" in df.columns and df["Modality"].notna().any():
        if "series_key" in df.columns:
            mod_df = df.drop_duplicates(subset="series_key")[["series_key", "Modality"]]
        else:
            mod_df = df[["Modality"]]
        vc = mod_df["Modality"].value_counts().head(20)
        if HAS_SNS:
            palette = sns.color_palette(PALETTE, len(vc))
            ax4.bar(vc.index, vc.values, color=palette, edgecolor="white")
        else:
            ax4.bar(vc.index, vc.values, edgecolor="white")
        ax4.set_xlabel("Modality", fontsize=9)
        ax4.set_ylabel("Series Count", fontsize=9)
        ax4.set_title("F4 · Modality Distribution\n(series-level)", fontweight="bold", fontsize=9)
        ax4.grid(axis="y", alpha=0.3)
        for xi, (label, yi) in enumerate(zip(vc.index, vc.values)):
            ax4.text(xi, yi + max(vc.values) * 0.01, str(yi),
                     ha="center", fontsize=7.5)
        ax4.tick_params(axis="x", rotation=30, labelsize=8)
    else:
        ax4.text(0.5, 0.5, "Modality N/A", ha="center", va="center")

    fig.tight_layout()
    _fig_title(fig, "F · Series-Level Analysis", fontsize=13)
    _save_fig(fig, pdf, out_dir, "F_series_analysis")
    print("  [F] Series-level analysis saved.")


# ============================================================
# EDA Section G – Patient-level composition
# ============================================================

def plot_patient_composition(df, pdf, out_dir):
    """
    G: Patient-level multi-modality composition.
      G1 - Heatmap: patient x modality (series count)
      G2 - Top modality combinations per patient

    Layout: fixed widescreen (18 x 7 in) with two side-by-side panels.
    G1 heatmap is capped at 40 patients so it stays readable on a slide.
    """
    if "patient_folder" not in df.columns or "Modality" not in df.columns:
        print("  [G] patient_folder or Modality not available - skipping.")
        return

    # Build patient x modality series-count pivot
    if "series_key" in df.columns:
        pm = df.drop_duplicates("series_key")[["patient_folder", "Modality"]]
    else:
        pm = df[["patient_folder", "Modality"]].copy()

    pm = pm.dropna(subset=["Modality"])
    if pm.empty:
        print("  [G] Empty after dropna - skipping.")
        return

    pivot = pm.groupby(["patient_folder", "Modality"]).size().unstack(fill_value=0)

    # Fixed widescreen size: 18 in wide, 7 in tall (fits a 16:9 PPT slide)
    fig = plt.figure(figsize=(18.0, 7.0))
    # Give slightly more room to H1 heatmap (65 %) vs H2 bar chart (35 %)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40,
                            width_ratios=[1.85, 1])

    # H1 - Heatmap (cap at 40 patients for slide readability)
    ax1  = fig.add_subplot(gs[0])
    show = pivot.head(40)
    # Decide whether to annotate based on cell count
    do_annot = (len(show) * len(show.columns)) <= 200
    if HAS_SNS:
        sns.heatmap(show, annot=do_annot, fmt="d",
                    cmap="YlOrRd", linewidths=0.3, ax=ax1,
                    annot_kws={"size": 7})
    else:
        im = ax1.imshow(show.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax1, label="Series count")
        ax1.set_xticks(range(len(show.columns)))
        ax1.set_xticklabels(show.columns, rotation=45, ha="right", fontsize=8)
        ax1.set_yticks(range(len(show.index)))
        ax1.set_yticklabels(show.index, fontsize=7)

    ax1.set_xlabel("Modality", fontsize=9)
    ax1.set_ylabel("Patient Folder", fontsize=9)
    ax1.tick_params(axis="y", labelsize=max(5, 8 - len(show) // 10))
    ax1.set_title(
        f"G1 · Patient x Modality Series Count"
        f"\n(showing {len(show)} of {len(pivot)} patients)",
        fontweight="bold", fontsize=9,
    )

    # G2 - Top modality combinations  (e.g. "CT+MR", "CT only")
    ax2 = fig.add_subplot(gs[1])
    combos = pivot.apply(
        lambda row: " + ".join(sorted(col for col, v in row.items() if v > 0)),
        axis=1,
    )
    vc = combos.value_counts().head(20)
    _barh(ax2, vc, max_items=20)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.set_title("G2 · Top Modality Combinations per Patient",
                  fontweight="bold", fontsize=9)

    _fig_title(fig, "G · Patient-Level Modality Composition", fontsize=13)
    _save_fig(fig, pdf, out_dir, "G_patient_composition")
    print("  [G] Patient composition saved.")


# ============================================================
# EDA Section H – Patient demographics
# ============================================================

def plot_patient_demographics(df, pdf, out_dir):
    """H: PatientSex pie chart + PatientAge histogram.

    De-duplication priority for unique-patient counting:
      1. PatientID column (hashed DICOM tag) -- most accurate
      2. patient_folder -- fallback when PatientID is absent
    Each patient is counted only once, regardless of how many
    series / slices they have in the dataset.
    """
    has_sex = "PatientSex" in df.columns and df["PatientSex"].notna().any()
    has_age = "PatientAge" in df.columns and df["PatientAge"].notna().any()

    if not has_sex and not has_age:
        print("  [H] Demographics not available - skipping.")
        return

    # Determine the best column to use as a unique-patient key.
    # PatientID (even when hashed) is the most reliable identifier from the
    # DICOM standard.  patient_folder is a directory-structure approximation.
    if "PatientID" in df.columns and df["PatientID"].notna().any():
        patient_key_col = "PatientID"
        key_label = "DICOM PatientID"
    elif "patient_folder" in df.columns:
        patient_key_col = "patient_folder"
        key_label = "patient folder"
    else:
        patient_key_col = None
        key_label = "all rows (no patient key found)"

    def _dedup_by_patient(dataframe):
        """Return one row per unique patient, using the best available key."""
        if patient_key_col is not None:
            return dataframe.drop_duplicates(subset=[patient_key_col])
        return dataframe  # no key available - use all rows

    n_unique = (
        df[patient_key_col].nunique()
        if patient_key_col is not None
        else len(df)
    )
    print(f"  [H] Demographics: {n_unique} unique patients (keyed by {key_label})")

    # Widescreen layout: fixed 18 x 7 in regardless of how many sub-plots
    n_plots = int(has_sex) + int(has_age)
    fig, axes = plt.subplots(1, n_plots, figsize=(18.0, 7.0))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # ---- Sex pie ----
    if has_sex:
        ax = axes[idx]; idx += 1
        # One row per unique patient, then look at PatientSex
        sex_s = _dedup_by_patient(df)["PatientSex"].dropna()
        vc = sex_s.value_counts()
        ax.pie(
            vc.values,
            labels=vc.index.tolist(),
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4ECDC4", "#FF6B6B", "#FFE66D", "#95E1D3"][:len(vc)],
        )
        ax.set_title(
            f"PatientSex Distribution\n(n={len(sex_s)} unique patients, by {key_label})",
            fontweight="bold",
        )

    # ---- Age histogram ----
    if has_age:
        ax = axes[idx]; idx += 1

        def parse_age(v):
            """Convert DICOM age string (e.g. '063Y', '18M') to numeric years."""
            v = str(v).strip().upper()
            try:
                if v.endswith("Y"):  return float(v[:-1])
                if v.endswith("M"):  return float(v[:-1]) / 12.0
                if v.endswith("W"):  return float(v[:-1]) / 52.0
                return float(v)
            except Exception:
                return None

        # One row per unique patient
        age_raw = _dedup_by_patient(df)["PatientAge"]
        age_num = age_raw.map(parse_age).dropna()

        if not age_num.empty:
            if HAS_SNS:
                sns.histplot(age_num, bins=20, kde=True, ax=ax,
                             color="#F7A440", edgecolor="white")
            else:
                ax.hist(age_num, bins=20, color="#F7A440", edgecolor="white")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Patient Count")
            med = age_num.median()
            ax.axvline(med, color="red", linestyle="--",
                       label=f"Median = {med:.1f} yr")
            ax.legend(fontsize=9)
            ax.set_title(
                f"Patient Age Distribution\n(n={len(age_num)} unique patients, by {key_label})",
                fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Age cannot be parsed\n(may be hashed)",
                    ha="center", va="center")
            ax.set_title("PatientAge", fontweight="bold")

    fig.tight_layout()
    _fig_title(fig, "H · Patient Demographics")
    _save_fig(fig, pdf, out_dir, "H_demographics")
    print("  [H] Demographics saved.")


# ============================================================
# EDA Section I – SeriesDescription visual gallery
# ============================================================

def _load_pixel_thumbnail(path: str, wc_meta=None, ww_meta=None, size: int = 224) -> np.ndarray:
    """
    Load a DICOM pixel array, apply Rescue slope/intercept and windowing,
    then down-sample to (size x size) uint8 for thumbnail display.

    Window priority:
      1. wc_meta / ww_meta passed from the DataFrame (extracted at metadata time)
      2. WindowCenter / WindowWidth embedded in the pixel-data DICOM header
      3. 1st-99th percentile normalization as final fallback
    """
    ds  = dcmread(path, force=True)
    arr = ds.pixel_array

    # Handle multi-frame DICOM (take middle frame)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    elif arr.ndim > 3:
        arr = arr[0, arr.shape[1] // 2]

    arr = arr.astype(np.float32)

    # Apply RescaleSlope / RescaleIntercept (Hounsfield units for CT)
    slope     = float(getattr(ds, "RescaleSlope",     1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept

    # Attempt to parse window from metadata columns first
    wc, ww = None, None
    if wc_meta is not None and ww_meta is not None:
        try:
            # WindowCenter/Width may be stored as multi-value strings "40\\400"
            wc = float(str(wc_meta).split("\\")[0].strip())
            ww = float(str(ww_meta).split("\\")[0].strip())
        except Exception:
            pass

    # Fallback: read WindowCenter/Width directly from this file's header
    if wc is None or ww is None:
        ds_wc = getattr(ds, "WindowCenter", None)
        ds_ww = getattr(ds, "WindowWidth",  None)
        if ds_wc is not None and ds_ww is not None:
            try:
                # pydicom returns DSfloat or MultiValue
                wc = float(ds_wc) if not hasattr(ds_wc, "__iter__") \
                     else float(list(ds_wc)[0])
                ww = float(ds_ww) if not hasattr(ds_ww, "__iter__") \
                     else float(list(ds_ww)[0])
            except Exception:
                pass

    # Apply windowing or percentile normalization
    if wc is not None and ww is not None and ww > 0:
        lo  = wc - ww / 2.0
        hi  = wc + ww / 2.0
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / ww * 255.0
    else:
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)

    arr = arr.clip(0, 255).astype(np.uint8)

    # Down-sample to size x size using evenly-spaced index sampling
    h, w    = arr.shape
    row_idx = np.linspace(0, h - 1, size, dtype=int)
    col_idx = np.linspace(0, w - 1, size, dtype=int)
    return arr[np.ix_(row_idx, col_idx)]


def plot_series_description_gallery(df: pd.DataFrame, pdf, out_dir: str,
                                    max_preview: int = 24):
    """
    Section I: SeriesDescription visual mapping gallery.

    For each unique SeriesDescription (up to max_preview, ordered by
    frequency), selects one representative DICOM file, loads its pixel
    data, and renders a labelled thumbnail with acquisition metadata:
      - Modality
      - Number of series with this description in the dataset
      - SliceThickness, PixelSpacing
      - WindowCenter / WindowWidth used for display

    The gallery makes it immediately visible what anatomical region /
    image type corresponds to each SeriesDescription label.
    """
    if "SeriesDescription" not in df.columns:
        print("  [I] SeriesDescription not available - skipping gallery.")
        return

    # Work at the series level (one row per unique series_key)
    if "series_key" in df.columns:
        series_df = df.drop_duplicates("series_key").copy()
    else:
        series_df = df.copy()

    series_df = series_df.dropna(subset=["SeriesDescription"])
    if series_df.empty:
        print("  [I] No SeriesDescription data - skipping gallery.")
        return

    # Count how many unique series carry each description, then take top N
    desc_counts = series_df["SeriesDescription"].value_counts()
    top_descs   = desc_counts.head(max_preview).index.tolist()
    total_descs = len(desc_counts)

    # For each description, pick one representative row (middle of the group)
    rep_rows = []
    for desc in top_descs:
        subset = series_df[series_df["SeriesDescription"] == desc]
        rep    = subset.iloc[len(subset) // 2]   # median row
        rep_rows.append({
            "desc":          desc,
            "n_series":      len(subset),          # series count with this desc
            "n_files":       int(desc_counts[desc]), # same here (series-level dedup)
            "path":          rep["path"],
            "Modality":      rep.get("Modality"),
            "SliceThickness":rep.get("SliceThickness"),
            "PixelSpacingRow":rep.get("PixelSpacingRow"),
            "WindowCenter":  rep.get("WindowCenter"),
            "WindowWidth":   rep.get("WindowWidth"),
            "Rows":          rep.get("Rows"),
            "Columns":       rep.get("Columns"),
        })

    n     = len(rep_rows)
    # 6-column layout keeps total figure width at ~18 in for widescreen PPT
    ncols     = min(6, n)
    nrows     = math.ceil(n / ncols)
    thumb_in  = 18.0 / ncols        # thumbnail width per cell (inches)
    title_in  = 0.85                # extra height for sub-title text (inches)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(18.0, nrows * (thumb_in + title_in)),
    )
    axes = np.array(axes).flatten()

    print(f"  [I] Loading pixel thumbnails for {n} SeriesDescriptions …")
    success, failed = 0, 0

    for i, info in enumerate(tqdm(rep_rows, desc="Loading pixels")):
        ax   = axes[i]
        path = info["path"]

        try:
            thumb = _load_pixel_thumbnail(
                path,
                wc_meta = info["WindowCenter"],
                ww_meta = info["WindowWidth"],
                size    = 224,
            )
            ax.imshow(thumb, cmap="gray", aspect="equal",
                      interpolation="bilinear")
            ax.axis("off")
            success += 1
        except Exception as exc:
            # Show a placeholder with the error message
            ax.set_facecolor("#1a1a2e")
            ax.text(0.5, 0.5,
                    f"Load failed:\n{type(exc).__name__}",
                    ha="center", va="center", fontsize=7,
                    color="#e94560", transform=ax.transAxes, wrap=True)
            ax.axis("off")
            failed += 1

        # Build the subtitle: SeriesDescription + key metadata
        desc = info["desc"]
        mod  = info["Modality"] or "?"
        ns   = info["n_series"]
        st   = info["SliceThickness"]
        psr  = info["PixelSpacingRow"]
        rows = info["Rows"]
        cols = info["Columns"]

        line1 = desc[:40] + ("…" if len(desc) > 40 else "")
        line2_parts = [f"[{mod}]", f"n={ns} series"]
        if rows and cols:
            line2_parts.append(f"{int(rows)}x{int(cols)}px")
        line3_parts = []
        if st is not None:
            line3_parts.append(f"ST={st:.1f}mm")
        if psr is not None:
            line3_parts.append(f"Spc={psr:.3f}mm")

        title_lines = [line1, "  ".join(line2_parts)]
        if line3_parts:
            title_lines.append("  ".join(line3_parts))

        ax.set_title("\n".join(title_lines),
                     fontsize=7.5, fontweight="bold",
                     pad=4, loc="center")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _fig_title(
        fig,
        f"I · SeriesDescription Visual Gallery"
        f"  (top {n} of {total_descs} unique descriptions)"
        f"  |  loaded: {success}  failed: {failed}",
        fontsize=12,
    )
    _save_fig(fig, pdf, out_dir, "I_series_gallery")
    print(f"  [I] Gallery saved  ({success} thumbnails, {failed} failed).")


# ============================================================
# Text summary (stdout)
# ============================================================

def print_text_summary(df):
    sep = "=" * 72
    print(f"\n{sep}")
    print("DICOM EDA  -  TEXT SUMMARY")
    print(sep)
    print(f"Total rows (files): {len(df):,}")

    if "patient_folder" in df.columns:
        print(f"Unique patients    : {df['patient_folder'].nunique():,}")
    if "series_key" in df.columns:
        print(f"Unique series      : {df['series_key'].nunique():,}")
    if "Modality" in df.columns:
        print(f"Unique modalities  : {df['Modality'].nunique():,}")
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err:
            print(f"Parse errors       : {n_err:,}  (see 'error' column in CSV)")

    # Missing rates
    skip = {"path", "patient_folder", "series_uid", "series_key",
            "series_label", "error", "PixelSpacing", "ImagerPixelSpacing"}
    cols = [c for c in df.columns if c not in skip]
    miss = df[cols].isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if not miss.empty:
        print("\nMissing-value rates (top 10):")
        for col, rate in miss.head(10).items():
            print(f"  {col:<40} {rate*100:5.1f}%")

    # Numeric summary
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if num_cols:
        print("\nNumeric column statistics:")
        stat = df[num_cols].describe().T
        with pd.option_context("display.float_format", "{:.4g}".format,
                               "display.max_columns", 20, "display.width", 120):
            print(stat[["count", "mean", "std", "min", "50%", "max"]])

    # Categorical summaries
    for col in ["Modality", "SeriesDescription", "BodyPartExamined",
                "Manufacturer", "PatientSex"]:
        if col in df.columns and df[col].notna().any():
            vc = df[col].value_counts().head(10)
            print(f"\n{col} (top 10):")
            for k, v in vc.items():
                print(f"  {str(k):<45} {v:,}")

    # Series per patient
    if "patient_folder" in df.columns and "series_key" in df.columns:
        spp = df.groupby("patient_folder")["series_key"].nunique()
        print(f"\nSeries per patient – mean: {spp.mean():.1f}  "
              f"median: {spp.median():.0f}  "
              f"min: {spp.min()}  max: {spp.max()}")

    print(f"\n{sep}\n")


# ============================================================
# Argument parser
# ============================================================

def create_parser():
    p = argparse.ArgumentParser(
        description="DICOM metadata extraction + automated EDA  (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dicom-dir", required=True,
                   help="Root directory of DICOM files (or .npy path list)")
    p.add_argument("--out-dir", default="./eda_output",
                   help="Output directory for CSV, PDF, and PNGs (default: ./eda_output)")
    p.add_argument("--out-csv", default=None,
                   help="Explicit CSV path  [default: <out-dir>/metadata.csv]")
    p.add_argument("--out-parquet", default=None,
                   help="Optional Parquet output path (requires pyarrow)")
    p.add_argument("--patient-depth", type=int, default=1,
                   help=(
                       "Number of directory levels below --dicom-dir that define one patient. "
                       "E.g. 1 = <root>/<patient>/ (default), "
                       "2 = <root>/<site>/<patient>/"
                   ))
    p.add_argument("--keep-phi", action="store_true",
                   help="Store raw PatientID / UIDs / dates (not hashed)")
    p.add_argument("--one-slice-per-series", action="store_true",
                   help="Keep only one representative DICOM per (patient, SeriesUID)")
    p.add_argument("--rep-policy", default="auto",
                   choices=["auto", "first", "middle", "last"],
                   help="Slice selection policy when --one-slice-per-series is used")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip all plotting - only save the CSV / Parquet")
    p.add_argument("--max-series-preview", type=int, default=24,
                   help=(
                       "Maximum number of unique SeriesDescriptions shown in "
                       "the Section I visual gallery (default: 24). "
                       "Ordered by frequency (most common first)."
                   ))
    p.add_argument("--no-gallery", action="store_true",
                   help="Skip Section I pixel gallery (useful when pixel data is unavailable)")
    return p


# ============================================================
# Main
# ============================================================

def main():
    opt = create_parser().parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    # ---- 1. Discover files ----
    print(f"\n[Step 1] Scanning DICOM files in: {opt.dicom_dir}")
    if opt.dicom_dir.lower().endswith(".npy"):
        paths = np.load(opt.dicom_dir, allow_pickle=True).tolist()
        root  = os.path.commonpath(paths) if paths else "."
    else:
        root  = opt.dicom_dir
        paths = scan_dicoms(root)

    if not paths:
        print("No DICOM files found. Exiting.")
        sys.exit(1)
    print(f"  Found {len(paths):,} file(s).")

    # ---- 2. Build series index (quick header pass) ----
    print(f"\n[Step 2] Building series index  (patient-depth = {opt.patient_depth}) …")
    series_index = build_series_index(paths, root, opt.patient_depth)

    n_patients = len({v["patient_folder"] for v in series_index.values()})
    n_series   = len({v["series_uid"]     for v in series_index.values()})
    print(f"  Detected {n_patients} patient folder(s) / {n_series} unique series.")

    # ---- 3. SlicePosition (before any reduction) ----
    print("\n[Step 3] Computing slice position per series …")
    slice_pos_map = compute_slice_position_in_series(paths, series_index)

    # ---- 4. Extract full metadata ----
    print("\n[Step 4] Extracting full metadata …")
    df = build_dataframe(
        root          = root,
        paths         = paths,
        series_index  = series_index,
        slice_pos_map = slice_pos_map,
        one_slice     = opt.one_slice_per_series,
        rep_policy    = opt.rep_policy,
        keep_phi      = opt.keep_phi,
    )

    # ---- 5. Save CSV ----
    csv_path = opt.out_csv or os.path.join(opt.out_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Step 5] CSV saved → {csv_path}  ({len(df):,} rows x {len(df.columns)} cols)")

    # ---- Optional Parquet ----
    if opt.out_parquet:
        try:
            df.to_parquet(opt.out_parquet, index=False)
            print(f"  Parquet saved → {opt.out_parquet}")
        except Exception as e:
            print(f"  Parquet failed (install pyarrow): {e}")

    # ---- 6. Text summary ----
    print_text_summary(df)

    # ---- 7. EDA plots ----
    if opt.no_plots:
        print("[Step 7] --no-plots set; skipping visualisation.")
        return

    print("[Step 7] Generating EDA plots …")
    pdf_path = os.path.join(opt.out_dir, "eda_report.pdf")

    with PdfPages(pdf_path) as pdf:
        d = pdf.infodict()
        d["Title"]   = "DICOM EDA Report"
        d["Author"]  = "dicom2EDA v3"
        d["Subject"] = f"Automated EDA: {opt.dicom_dir}"

        plot_overview(df, pdf, opt.out_dir)              # A
        plot_missing_values(df, pdf, opt.out_dir)        # B
        plot_categorical(df, pdf, opt.out_dir)           # C
        plot_numeric_dist(df, pdf, opt.out_dir)          # D
        plot_image_geometry(df, pdf, opt.out_dir)        # E
        plot_series_analysis(df, pdf, opt.out_dir)       # F
        plot_patient_composition(df, pdf, opt.out_dir)   # G
        plot_patient_demographics(df, pdf, opt.out_dir)  # H

        # Section I requires pixel data - skip if --no-gallery is set
        if not opt.no_gallery:
            plot_series_description_gallery(
                df, pdf, opt.out_dir,
                max_preview=opt.max_series_preview,
            )  # I
        else:
            print("  [I] --no-gallery set; skipping pixel gallery.")

    print(f"\n✔  EDA report (PDF) → {pdf_path}")
    print(f"✔  PNGs            → {opt.out_dir}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
