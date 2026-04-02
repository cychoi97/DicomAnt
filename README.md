# DICOM2EDA

> **All-in-one DICOM metadata extraction + automated EDA pipeline**  
> Scan a directory tree of DICOM files, extract structured metadata into a tidy DataFrame, and generate a comprehensive EDA report (PDF + individual PNGs) — in a single command.

---

## Features

| Category | Details |
|----------|---------|
| **Privacy-safe by default** | PatientID, UID tags are SHA-256 hashed; dates coarsened to YYYYMM |
| **Hierarchy-aware** | Configurable patient-folder depth (`--patient-depth`) for nested datasets |
| **DICOM-header series grouping** | Series are discriminated by `SeriesInstanceUID`, `SeriesNumber`, `SeriesDescription` — independent of folder layout |
| **Mixed-series support** | Multiple modalities / series can coexist in the same directory |
| **Slice position tracking** | `SlicePositionInSeries` (0–100 %) computed per series without loading pixels |
| **9-section EDA report** | Auto-generated multi-page PDF + standalone PNGs at 300 DPI |
| **Visual gallery (Section J)** | Loads actual pixel data to display representative thumbnails per `SeriesDescription` |
| **Widescreen-ready figures** | **All** EDA sections (A–J) output at 300 DPI in 16:9 landscape format (fixed 18 in wide), optimized for direct insertion into PowerPoint slides |
| **CSV / Parquet output** | Flat metadata table saved alongside the report |

---

## Directory Model

```
<root>/                          ← --dicom-dir
  <patient_A>/                   ← level 1 (--patient-depth 1, default)
    <study_01>/
      series_CT_chest.dcm        ← CT series
      series_MR_brain.dcm        ← MR series in the same folder
    <study_02>/
      ...
  <patient_B>/
    ...
```

For multi-site datasets (e.g. `root/site/patient/`), set `--patient-depth 2`.

---

## EDA Sections

| Section | Title | Layout | Description |
|---------|-------|--------|-------------|
| **A** | Dataset Overview | 18 × 5.5 in · 2 side-by-side tables | File count, error count, total size, unique patients / studies / series / modalities, avg series per patient |
| **B** | Missing Value Rate | 18 × 7 in · single wide chart | Horizontal bar chart of missing-value (%) per column (up to 40), colour-coded by severity |
| **C** | Categorical Distributions | 18 in wide · **4-col** grid | Bar charts for Modality, SeriesDescription, BodyPart, Manufacturer, PatientSex, etc. |
| **D** | Numeric Distributions | 18 in wide · **5-col** grid | Histogram + KDE for KVP, SliceThickness, PixelSpacing, Exposure, etc. (1st–99th pct clipped) |
| **E** | Image Geometry | 18 × 6 in · 2-panel (70/30 split) | Rows × Columns hexbin scatter + BitsStored distribution |
| **F** | Series-Level Analysis | 18 × 7 in · **1-row × 4-col** | F1 slice count histogram · F2 series count per patient · F3 top-15 SeriesDescription · F4 modality distribution |
| **G** | Patient Composition | 18 × 7 in · 2-panel (65/35 split) | G1 patient × modality heatmap · G2 modality combination frequency |
| **H** | Patient Demographics | 18 × 7 in · 1 or 2 side-by-side | PatientSex pie chart + PatientAge histogram (de-duplicated by DICOM `PatientID`) |
| **I** | SeriesDescription Gallery | 18 in wide · **6-col** grid | Pixel thumbnails for top-N unique `SeriesDescription` labels with acquisition metadata overlay |

---

## Installation

```bash
pip install pydicom pandas tqdm natsort matplotlib seaborn pyarrow
```

> `seaborn` and `pyarrow` are optional but recommended.  
> Without `seaborn`, all plots fall back to plain matplotlib.  
> Without `pyarrow`, the `--out-parquet` option is unavailable.

---

## Quick Start

```bash
# Basic run (patient folders are direct children of the root)
python dicom2EDA.py --dicom-dir /path/to/data --out-dir ./eda_output

# Multi-site layout: root/site/patient/
python dicom2EDA.py --dicom-dir /path/to/data --out-dir ./eda_output \
    --patient-depth 2

# Keep only one representative slice per series (faster, metadata-only EDA)
python dicom2EDA.py --dicom-dir /path/to/data --out-dir ./eda_output \
    --one-slice-per-series --rep-policy auto

# Skip Section J pixel gallery (no pixel data access required)
python dicom2EDA.py --dicom-dir /path/to/data --out-dir ./eda_output \
    --no-gallery

# Retain raw PHI (research use only — ensure data governance compliance)
python dicom2EDA.py --dicom-dir /path/to/data --out-dir ./eda_output \
    --keep-phi

# Export Parquet in addition to CSV
python dicom2EDA.py --dicom-dir /path/to/data --out-dir ./eda_output \
    --out-parquet ./eda_output/metadata.parquet

# Metadata only (no plots)
python dicom2EDA.py --dicom-dir /path/to/data --out-dir ./eda_output \
    --no-plots
```

---

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--dicom-dir` | *(required)* | Root directory of DICOM files, or `.npy` file containing a list of paths |
| `--out-dir` | `./eda_output` | Output directory for CSV, PDF report, and PNG figures |
| `--out-csv` | `<out-dir>/metadata.csv` | Explicit path for the metadata CSV |
| `--out-parquet` | `None` | Optional Parquet output path (requires `pyarrow`) |
| `--patient-depth` | `1` | Directory levels below `--dicom-dir` that define one patient folder |
| `--keep-phi` | `False` | Store raw `PatientID` / UIDs / dates instead of hashing |
| `--one-slice-per-series` | `False` | Extract only one representative DICOM per `(patient, SeriesUID)` |
| `--rep-policy` | `auto` | Slice selection policy: `auto` (middle) · `first` · `middle` · `last` |
| `--no-plots` | `False` | Skip all plotting; save CSV/Parquet only |
| `--no-gallery` | `False` | Skip Section J pixel gallery |
| `--max-series-preview` | `24` | Max unique `SeriesDescription` thumbnails shown in Section J |

---

## Output Structure

```
eda_output/
  metadata.csv          ← flat per-file metadata table
  eda_report.pdf        ← multi-page EDA report (all sections)
  A_overview.png
  B_missing_values.png
  C_categorical.png
  D_numeric_distributions.png
  E_image_geometry.png
  F_series_analysis.png
  G_patient_composition.png
  H_demographics.png
  I_series_gallery.png  ← pixel thumbnails (skipped with --no-gallery)
```

---

## DataFrame Columns

### Derived columns (added by this script)

| Column | Description |
|--------|-------------|
| `patient_folder` | Relative path at `--patient-depth` levels below root |
| `series_uid` | Hashed (or raw) `SeriesInstanceUID` |
| `series_label` | Human-readable `"<SeriesNumber> – <SeriesDescription>"` |
| `series_key` | Unique grouping key: `patient_folder::series_uid` |
| `SeriesSliceCount` | Total DICOM files in the series |
| `SlicePositionInSeries` | Ordinal position within the series as a percentage (0–100 %) |

### Key DICOM tags extracted

**Series identifiers**
`SeriesNumber`, `SeriesDescription`, `ProtocolName`, `StudyDescription`, `ImageType`

**Acquisition parameters**
`Modality`, `KVP`, `Exposure`, `ExposureTime`, `XRayTubeCurrent`, `SliceThickness`, `SpacingBetweenSlices`, `ConvolutionKernel`, `ReconstructionDiameter`, `WindowCenter`, `WindowWidth`

**Image geometry**
`Rows`, `Columns`, `BitsStored`, `PhotometricInterpretation`, `PixelSpacingRow`, `PixelSpacingCol`

**PHI-like tags** *(hashed by default)*
`PatientID`, `PatientSex`, `PatientAge`, `PatientBirthDate`, `StudyInstanceUID`, `SeriesInstanceUID`, `SOPInstanceUID`, `AccessionNumber`, `StudyDate`, `SeriesDate`

---

## Privacy & Safety

By default:
- `PatientID`, `AccessionNumber`, and all `*UID` tags are replaced with a **16-character SHA-256 hex digest** — collision probability is negligible.
- Dates are truncated to **YYYYMM** (study/series/birth).
- Times are truncated to **HHMM**.
- `PatientSex` and `PatientAge` are stored as-is (low re-identification risk).

Use `--keep-phi` **only** in controlled research environments with proper data governance approval.

---

## Section I — SeriesDescription Visual Gallery

Section I loads actual pixel data to render one representative slice thumbnail per unique `SeriesDescription`.

**Windowing priority:**
1. `WindowCenter` / `WindowWidth` from the metadata DataFrame
2. Same tags read directly from the pixel-data DICOM header
3. 1st–99th percentile normalization (fallback)

CT images are automatically converted to Hounsfield Units via `RescaleSlope` / `RescaleIntercept`.

Each thumbnail is annotated with:
```
T1_MPRAGE
[MR]  n=15 series  512x512px
ST=1.0mm  Spc=0.488mm
```

Use `--no-gallery` to skip this section when pixel data is unavailable or access is slow.

---

## Notes

- DICOM files are detected by extension (`.dcm`, `.dicom`) **or by having no extension at all** (common in PACS exports).
- A `.npy` file containing a Python list of absolute paths can be passed to `--dicom-dir` for pre-filtered datasets.
- All figures are saved at **300 DPI** in a fixed **18-inch wide** landscape size (16:9). Each section height is either fixed or gently capped so no figure exceeds roughly 12 inches tall — making every output suitable for direct insertion into a 16:9 PowerPoint slide.
- Per-section layout summary:

  | Section | Width | Height | Grid |
  |---------|-------|--------|------|
  | A | 18 in | 5.5 in | 1 × 2 sub-tables |
  | B | 18 in | 7 in | 1-panel bar chart |
  | C | 18 in | dynamic (2.8 in × rows) | 4 col |
  | D | 18 in | dynamic (2.8 in × rows) | 5 col |
  | E | 18 in | 6 in | 1 × 2 (70/30) |
  | F | 18 in | 7 in | 1 × 4 (single row) |
  | G | 18 in | 7 in | 1 × 2 (65/35) |
  | H | 18 in | 7 in | 1 × 1 or 1 × 2 |
  | I | 18 in | dynamic | 6 col |
