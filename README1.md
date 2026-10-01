# 🛰️ SatQueryAI — Agentic Remote-Sensing Vision-Language Assistant

> An agentic, query-driven dashboard that lets a non-expert describe what they want to know about satellite imagery in plain English — and routes that request to the right remote-sensing specialist tool, single image or multi-image, optical or SAR.

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](#)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](#)
[![Model](https://img.shields.io/badge/Backbone-BLIP-yellow)](#)
[![Status](https://img.shields.io/badge/Status-Working%20Prototype-orange)](#)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [What SatQueryAI Is (Current Prototype)](#2-what-satqueryai-is-current-prototype)
3. [System Architecture](#3-system-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Setup](#5-installation--setup)
6. [How to Use the Dashboard](#6-how-to-use-the-dashboard)
7. [Module-by-Module Code Walkthrough](#7-module-by-module-code-walkthrough)
8. [Test Cases & Demonstration Prompts](#8-test-cases--demonstration-prompts)
9. [Requirements](#9-requirements)
10. [Alignment Matrix — Problem Statement vs. Current Prototype](#10-alignment-matrix--problem-statement-vs-current-prototype)
11. [Known Limitations of the Current Prototype](#11-known-limitations-of-the-current-prototype)
12. [Roadmap — What's Needed for 100% Alignment](#12-roadmap--whats-needed-for-100-alignment)
13. [Deliverables Checklist](#13-deliverables-checklist)

---

## 1. Problem Statement

### 1.1 Background

Remote-sensing imagery underpins agricultural monitoring, disaster management, urban planning, forest and water-resource assessment, infrastructure mapping, and environmental analysis. Most existing remote-sensing AI tools, however, are built as **single-purpose applications** — one model for land-cover classification, another for object detection, another for VQA, another for change detection. This forces a user to already understand satellite data characteristics, GIS workflows, and model selection before they can extract any information, which puts meaningful use of satellite imagery out of reach for non-expert users.

A further complication is that a single optical image is often **not enough** to answer real operational questions:

- **Optical / multispectral imagery** carries rich spectral and contextual detail but is blocked by cloud cover and unavailable at night.
- **Synthetic Aperture Radar (SAR)** provides complementary structural information and works day-or-night, through clouds — but lacks spectral richness.
- **Multitemporal pairs** (same location, two dates) are the only way to detect and interpret *change*.
- **Co-registered optical + SAR pairs** together give a more complete picture than either sensor alone.

Because of this, a **general-purpose LLM or VLM cannot be expected to reliably handle these specialised tasks** without remote-sensing adaptation — domain fine-tuning, sensor-aware preprocessing, and task-specific specialist components are required. A solution built on an un-adapted, generic VLM does **not** satisfy the brief.

### 1.2 The Ask: SatQuery AI

The Statement of Work calls for **SatQuery AI**, an **agentic, query-driven** vision-language assistant — not a single monolithic model, but a **controller** that:

1. Interprets a natural-language query and classifies the requested task.
2. Validates the number, modality, format, and compatibility of the supplied image(s).
3. Selects one or more specialist models/tools from a registry.
4. Executes the selected workflow with permitted parameters only.
5. Combines textual + spatial outputs, estimates confidence, and returns visual evidence.
6. Produces an **auditable execution summary** (selected task, model/tool names, key parameters) — the controller's *internal* reasoning is not evaluated, only this observable trace.

#### Defined Input Scope

| Input type | Description | Formats |
|---|---|---|
| **Single image** | One optical/multispectral or SAR image | Captioning, VQA, text-guided grounding |
| **Cross-modal pair** | Co-registered optical/multispectral + SAR of the same area | Joint / fusion analysis |
| **Bi-temporal pair** | Two images of the same area, different dates | Change detection / change VQA |

- **Primary formats:** GeoTIFF / TIFF for real geospatial imagery.
- **PNG / JPEG** are accepted **only** for the prescribed public benchmark datasets (not for the georeferenced evaluation set).

#### Mandatory Functional Scope

| # | Requirement | Mandatory? |
|---|---|---|
| 1 | Remote-sensing adaptation of at least one visual/VLM component (fine-tuned or otherwise adapted using **BigEarthNet.txt** or other open-source RS data) | ✅ Mandatory |
| 2 | Single-image **VQA** | ✅ Mandatory |
| 3 | One additional single-image task: **captioning/scene description** *or* **text-guided grounding** | ✅ Mandatory (pick one, or both) |
| 4 | Multi-image **change description** or **change-VQA** from a bi-temporal pair (spatial change map optional, only where reference masks exist) | ✅ Mandatory |
| 5 | **Cross-modal (optical + SAR)** joint information extraction | ✅ Mandatory |
| 6 | **Agentic orchestration** — automatic task classification, input validation, tool selection/sequencing, output fusion, confidence estimation, and an auditable execution trace | ✅ Mandatory |

#### Representative Queries the System Must Handle

- *"Describe the land-cover and major objects visible in this image."*
- *"Highlight the water body referred to in the query."*
- *"What changed between these two dates, and where did the change occur?"*
- *"Use the optical and SAR images together to identify built-up and water-covered regions."*
- *"Has the built-up area increased, decreased, or remained unchanged?"*

#### Evaluation

Final scoring uses **prescribed public benchmark test subsets** (BigEarthNet for image–text adaptation; **VRSBench** and **RSVQA** for single-image captioning/grounding/VQA; **CDVQA** for change-based VQA) plus a held-out **ISRO/SAC evaluation set** of pre-georeferenced, co-registered **Cartosat-2S (optical)** and **RISAT (SAR)** image pairs with task-specific reference answers/labels/boxes/masks. Scores are normalised before being combined. Evaluation annotations are never disclosed to teams — reproducible, deterministic behaviour matters as much as raw accuracy.

---

## 2. What SatQueryAI Is (Current Prototype)

**SatQueryAI** is the working prototype built against this brief. It is a single-page **Streamlit** dashboard backed by a lightweight **agentic controller** (`router.py`) and a **specialist toolbox** (`vision_tools.py`) built on top of a **BLIP** (Bootstrapped Language-Image Pretraining) vision-language backbone, with a **BigEarthNet-aware prompting layer** (`bigearthnet_adapter.py`) that injects remote-sensing vocabulary into queries before they reach the model.

At a glance, the prototype currently supports:

- 📷 **Single-image VQA** — free-form question answering over one optical/SAR image.
- 📝 **Single-image captioning** — automatic scene description ("A satellite view showing…").
- 🎯 **Text-guided visual grounding** — localises a salient region and draws a bounding box.
- 🕒 **Bi-temporal change detection** — pixel-difference-based change mask + change percentage across two dates of the same location.
- 🛰️ **Optical–SAR fusion** — builds an optical+SAR composite, runs VQA over it, and returns a domain-styled fusion summary.
- 🤖 **Rule-based agentic routing** — a keyword/heuristic classifier inspects the query text and image count and dispatches to the correct specialist function automatically.
- 📋 **Auditable execution trace** — every run produces a JSON report (timestamp, image count, query, selected task, "model" name, confidence score, output) that the user can download.

This satisfies the *shape* of the brief end-to-end (one interactive app, one entry point, all five task categories reachable, and an execution trace is emitted) — but, as detailed in [Section 10](#10-alignment-matrix--problem-statement-vs-current-prototype) and [Section 12](#12-roadmap--whats-needed-for-100-alignment) below, several of the *mandatory* requirements are currently implemented with **placeholder / classical computer-vision logic** rather than the remote-sensing-adapted deep-learning components the SOW calls for.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────┐
│         User: image(s)  +  natural-language query│
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              app.py  (Streamlit UI)              │
│         file upload  │  query box  │  results    │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│           router.py  ·  QueryRouter              │
│   inspects image count + query keywords          │
│   returns one task label ──────────────────────► │
└───────────────────────┬─────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          │             │              │
          ▼             ▼              ▼
   ┌────────────┐ ┌───────────┐ ┌───────────────────┐
   │ BLIP-based │ │  OpenCV   │ │      NumPy        │
   │  vision_   │ │ grounding │ │  pixel-diff       │
   │  tools.py  │ │ (contour) │ │  change mask      │
   └─────┬──────┘ └─────┬─────┘ └────────┬──────────┘
         │              │                │
         ▼              │                │
┌──────────────────┐    │                │
│  bigearthnet_    │    │                │
│  adapter.py      │    │                │
│  (RS prompt      │    │                │
│   enrichment)    │    │                │
└────────┬─────────┘    │                │
         │              │                │
         └──────────────┴────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│            Execution Trace  (JSON)               │
│  timestamp · task · model · confidence · output  │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│      Result rendered back in Streamlit UI        │
│      text answer  +  image evidence  +  download │
└─────────────────────────────────────────────────┘
```

**Task routing logic at a glance:**

| Images | Keyword cue in query | Task routed to | Specialist used |
|--------|----------------------|----------------|-----------------|
| 1 | `describe`, `caption`, `tell me about` | Captioning | BLIP conditional generation |
| 1 | `where`, `locate`, `find`, `highlight`, `box` | Grounding | OpenCV contour detector |
| 1 | anything else | VQA | BLIP VQA head |
| 2 | `sar`, `radar`, `optical`, `fusion`, `together` — or Cross-Modal mode | Optical–SAR Fusion | BLIP VQA on composite |
| 2 | anything else | Change Detection | NumPy pixel-diff |

**Control flow, in words:**

1. The user picks a workflow mode (*Auto-Detect* or *Cross-Modal*), uploads 1–2 images, and types a query.
2. `app.py` counts the images and calls `QueryRouter.route_query(query, num_images, mode)`.
3. The router inspects **image count first**, then **keywords in the query**, and returns one of five task labels: `captioning`, `vqa`, `grounding`, `change_detection`, `optical_sar_fusion`.
4. `app.py` dispatches to the matching method on `SatelliteTools`, all of which share one cached BLIP model instance (`@st.cache_resource`).
5. Every branch renders its result inline (text + image(s)) and assigns a confidence value.
6. Regardless of which task ran, `app.py` assembles an **execution_trace** dict and renders it in an expandable "Auditable Execution Summary & Report Card" panel, with a JSON download button.

---

## 4. Repository Structure

```
SatQueryAI/
├── .venv/                        # Local Python virtual environment (not committed)
├── data/
│   └── Test_case/                # Sample/benchmark imagery used for demonstration
├── local_blip_model/              # Locally cached BLIP checkpoint (loaded via from_pretrained)
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── src/
│   ├── __pycache__/
│   ├── app.py                    # Streamlit UI + orchestration glue + execution-trace panel
│   ├── router.py                 # QueryRouter — rule-based agentic task classifier
│   ├── vision_tools.py           # SatelliteTools — all specialist inference functions
│   └── bigearthnet_adapter.py    # BigEarthNet class list + domain-prompt enrichment
├── README.md
└── requirements.txt
```

> **Note:** `local_blip_model/` currently holds a **plain pretrained BLIP checkpoint** (standard `config.json` / `model.safetensors` / tokenizer files consistent with a stock `Salesforce/blip-*` export). There is no visible fine-tuning artifact (no LoRA/PEFT adapter weights, no training logs) distinguishing it from an off-the-shelf BLIP model — see [Section 11](#11-known-limitations-of-the-current-prototype).

---

## 5. Installation & Setup

```bash
# 1. Clone / unzip the project
cd SatQueryAI

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure the local BLIP checkpoint is present at ./local_blip_model
#    (config.json, model.safetensors, preprocessor_config.json,
#     tokenizer_config.json, vocab.txt)

# 5. Launch the dashboard
cd src
streamlit run app.py
```

The app will open in your browser (default `http://localhost:8501`).

---

## 6. How to Use the Dashboard

1. **Choose a workflow** in the sidebar:
   - *Auto-Detect (Single/Bi-temporal)* — for single-image tasks and change detection.
   - *Cross-Modal (Optical + SAR Fusion)* — forces optical–SAR fusion routing regardless of keywords.
2. **Upload image(s):**
   - `Primary Image` — always required (optical/multispectral image, or the "before" image for change detection, or the optical image for fusion).
   - `Secondary Image` — required only for change detection (the "after" image) or fusion (the SAR image).
3. **Type a natural-language query** (see [representative queries](#representative-queries-the-system-must-handle) above, or the [test-case prompts](#8-test-cases--demonstration-prompts) below).
4. Click **Run Analysis**. The agent controller reports which specialist module it routed to, then streams the result (text + image evidence).
5. Expand **📋 Auditable Execution Summary & Report Card** to inspect or download the JSON execution trace.

---

## 7. Module-by-Module Code Walkthrough

### `router.py` — `QueryRouter`

A deterministic, rule-based task classifier:

- If **two images** are supplied: routes to `optical_sar_fusion` when the mode flag is `cross_modal` *or* the query contains fusion-flavoured words (`sar`, `radar`, `optical`, `fusion`, `together`); otherwise defaults to `change_detection`.
- If **one image** is supplied: routes to `grounding` on locate-style keywords (`where`, `locate`, `find`, `point`, `highlight`, `box`), else to `captioning` on describe-style keywords (`describe`, `caption`, `tell me about`), else falls back to `vqa`.

This is the "agentic controller" of the brief in its current, minimal form — it performs task classification and tool selection, but does **not** currently perform input-compatibility/metadata/format validation, multi-tool sequencing, or output fusion across tools.

### `vision_tools.py` — `SatelliteTools`

Loads a single local BLIP checkpoint three ways (`BlipProcessor`, `BlipForQuestionAnswering`, `BlipForConditionalGeneration`) and exposes:

| Method | Task | Approach |
|---|---|---|
| `answer_question()` | VQA | BLIP VQA head over the domain-enriched question |
| `describe_image()` | Captioning | BLIP conditional generation, seeded with a hidden prompt ("A satellite view showing"), which is stripped from the output |
| `detect_changes()` | Change detection | Grayscale + resize both images to 256×256, take absolute pixel difference, threshold at 30, report the % of changed pixels |
| `ground_object()` | Grounding | OpenCV: grayscale → binary threshold (120) → contour detection → bounding box around the **largest contour by area** |
| `analyze_optical_sar()` | Cross-modal fusion | Builds a side-by-side (`hconcat`) optical+SAR composite, runs BLIP VQA on it with a fusion-flavoured prompt, then maps the raw yes/no/target answer to a canned, pre-written sentence via a lookup dictionary |

### `bigearthnet_adapter.py`

Holds the 17 BigEarthNet-19 land-cover class names and a `get_domain_prompt()` helper that appends a fixed clause — *"Focus on Sentinel-2 land cover categories such as urban fabric, arable land, forest, or water bodies."* — to the user's query whenever it contains land-cover-ish keywords (`land-cover`, `classify`, `describe`). This is **prompt engineering**, not model adaptation: no model weights are touched here.

### `app.py`

Streamlit entry point: file uploaders, query box, task dispatch, per-task rendering, and the mandatory execution-trace panel with JSON download.

---

## 8. Test Cases & Demonstration Prompts

These are the five scripted demonstrations the current prototype is built and verified against:

### Test Case 1 — Single-Image VQA & Captioning
- **Mode:** Auto-Detect
- **Input:** One Sentinel-2 optical patch (`.tif` / `.png` / `.jpg`)
- **Query:** `"Describe the land-cover and major objects visible in this image."`
- **Expected outcome:** Descriptive terrain text with a confidence score and an inline image render.

### Test Case 2 — Text-Guided Visual Grounding
- **Mode:** Auto-Detect
- **Input:** One optical urban/landscape image
- **Query:** `"Locate the target or building."`
- **Expected outcome:** An OpenCV contour pass isolates the highest-contrast region and overlays a bounding-box trace.

### Test Case 3 — Bi-Temporal Change Detection
- **Mode:** Auto-Detect
- **Inputs:** Baseline image (e.g. OSCD `im1`) + later image of the same location (e.g. OSCD `im2`)
- **Query:** `"What changed between these two dates?"`
- **Expected outcome:** A NumPy pixel-difference pass renders T1 (Before) / T2 (After) / Change Mask side by side with a structural-change percentage.

### Test Case 4 — Cross-Modal Optical–SAR Fusion
- **Mode:** Cross-Modal (Optical + SAR Fusion)
- **Inputs:** Co-registered optical `.tif` (`rgb` folder) + corresponding SAR `.tif` (`SAR/VV` folder)
- **Query:** `"Use the optical and SAR images together to identify built-up and water-covered regions."`
- **Expected outcome:** A horizontal optical+SAR composite, a joint-analysis pass, and a domain-styled fusion summary sentence.

### Test Case 5 — Auditable Execution Trace Verification
- **Action:** Expand **📋 Auditable Execution Summary & Report Card** after any run.
- **Verification:** Confirm the JSON payload includes timestamp, modality count, task routing path, "specialist model" metadata, confidence score, and output summary; confirm **Download Execution Report (JSON)** produces a valid local file.

---

## 9. Requirements

```
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
peft==0.6.2
streamlit==1.28.0
rasterio==1.3.9
Pillow==10.0.1
numpy==1.26.0
datasets
```

> `peft` and `rasterio` are already declared as dependencies but are **not yet exercised by the current code** — `peft` implies an intended LoRA/adapter fine-tuning path that hasn't been wired up yet, and `rasterio` implies intended GeoTIFF/geospatial-metadata handling that the app currently substitutes with plain `PIL.Image.open()` / `cv2.imread()`. Both are flagged in the roadmap below.

---

## 10. Alignment Matrix — Problem Statement vs. Current Prototype

| Mandatory Requirement | Status | Current Implementation | Gap |
|---|---|---|---|
| Remote-sensing adaptation of a visual/VLM component via BigEarthNet or open RS data | ⚠️ **Partial** | `bigearthnet_adapter.py` only enriches the *text prompt* with BigEarthNet class names; the BLIP weights themselves are not shown to be fine-tuned | No visible training/PEFT script or adapter weights; `local_blip_model` looks like a stock checkpoint |
| Single-image VQA | ✅ **Done** | `answer_question()` via BLIP VQA head | Uses generic BLIP VQA weights, not RS-fine-tuned |
| One additional single-image task (captioning **or** grounding) | ⚠️ **Partial** | Both are implemented, but grounding is classical CV, not a learned grounding model | `ground_object()` ignores the *semantic content* of the query entirely — any query returns the largest high-contrast contour, so "highlight the water body" and "highlight the road" behave identically |
| Multi-image change **description**/**VQA** | ⚠️ **Partial** | `detect_changes()` returns only a numeric % and a mask — no natural-language description of *what* changed or *where*, and no VQA over the pair | The SOW's example query *"Has the built-up area increased, decreased, or remained unchanged?"* cannot currently be answered — there is no semantic/class-aware change reasoning |
| Cross-modal (optical+SAR) joint analysis | ⚠️ **Partial** | `analyze_optical_sar()` runs VQA on a naive side-by-side composite, then maps the raw answer through a **hard-coded dictionary** of three canned sentences (`yes` / `no` / `target`) | Output is templated boilerplate rather than content genuinely conditioned on the fused optical+SAR signal; no true feature-level fusion |
| Agentic orchestration (task classification, input validation, tool selection, output fusion, confidence, audit trace) | ⚠️ **Partial** | `router.py` does keyword-based task classification; `app.py` emits a full execution trace | No format/CRS/band/metadata compatibility checks; no multi-tool sequencing/output fusion across specialists; confidence scores are **hard-coded constants**, not computed |
| GeoTIFF/TIFF as primary input format | ⚠️ **Partial** | File uploader accepts `.tif`, but reading is done with `PIL`/`cv2`, which do not correctly handle multi-band, 16-bit, or georeferenced GeoTIFFs | `rasterio` (already a dependency) is never imported/used |
| PNG/JPEG restricted to benchmark datasets only | ❌ **Not enforced** | Uploader accepts png/jpg/jpeg/tif for every workflow with no source/provenance check | No logic distinguishes "benchmark" uploads from "operational" uploads |
| Evidence-grounded, auditable execution summary | ✅ **Done** | `execution_trace` dict + `st.json` + JSON download button | `specialist_model` field is hard-coded to `"BlipForVisionDomainAdapted"` even for change-detection/grounding runs that never touch BLIP at all — the trace is not fully truthful about which component actually ran |
| Evaluation compatibility (VRSBench, RSVQA, CDVQA, BigEarthNet, ISRO/SAC Cartosat-2S+RISAT set) | ❌ **Not implemented** | No dataset loaders, no benchmark evaluation scripts, no Cartosat-2S/RISAT-specific band handling | Needed before the prescribed benchmark subsets or the ISRO/SAC set can be scored |

**Legend:** ✅ Done · ⚠️ Partial / superficial · ❌ Missing

---

## 11. Known Limitations of the Current Prototype

Being transparent about the current state is deliberate — these are the specific things a reviewer will notice:

1. **No demonstrated fine-tuning.** The SOW is explicit that *"a generic LLM or VLM without remote-sensing adaptation will not satisfy the requirements."* Today, remote-sensing "adaptation" is limited to string-concatenation prompt hints in `bigearthnet_adapter.py`; the BLIP model weights appear to be the stock pretrained checkpoint.
2. **Grounding is not text-guided.** `ground_object()` never actually reads the query — it draws a box around whichever contour has the largest area after a fixed intensity threshold. Two different queries on the same image always return the same box.
3. **Change "understanding," not change "description."** `detect_changes()` is a global grayscale pixel-difference threshold. It cannot say *what* changed (e.g., "new built-up area appeared in the north-east quadrant") or answer comparative questions about specific land-cover classes.
4. **Fusion output is templated.** `analyze_optical_sar()`'s `response_mapping` dictionary returns one of three fixed sentences regardless of the actual scene content — the "analysis" is a raw VQA yes/no answer wrapped in marketing-style copy, not a genuine multi-sensor synthesis.
5. **Confidence scores are constants**, not estimates derived from model logits, agreement between specialists, or annotation quality (e.g. `0.85`, `0.89`, `0.92`, `0.93` are hard-coded in `app.py`/`vision_tools.py`).
6. **No GeoTIFF-aware I/O.** Despite `rasterio` being listed as a dependency, all image reads go through `PIL.Image.open()` or `cv2.imread()`, which will silently mis-handle multi-band, 16-bit, or nodata-masked GeoTIFFs and drop all georeferencing/CRS metadata.
7. **No compatibility validation.** The router only checks *how many* images were uploaded — it does not check that a "cross-modal" pair is actually one optical + one SAR image, that a "bi-temporal" pair shares the same footprint, or that file formats/resolutions are compatible.
8. **Execution trace can misreport the model used.** `specialist_model` is always logged as `"BlipForVisionDomainAdapted"`, even when the executed task (`grounding`, `change_detection`) never invoked BLIP at all — this weakens the auditability the SOW asks for.
9. **No benchmark or evaluation harness.** There is no code yet that loads BigEarthNet, VRSBench, RSVQA, or CDVQA splits, or that would let the system be scored automatically against the ISRO/SAC Cartosat-2S/RISAT evaluation pairs.

---

## 12. Roadmap — What's Needed for 100% Alignment

### 12.1 Remote-Sensing Model Adaptation (highest priority)
- [ ] Fine-tune (or LoRA/PEFT-adapt, using the already-declared `peft` dependency) the BLIP image encoder/text-decoder on **BigEarthNet.txt** image–caption/label pairs to actually shift the visual representation toward multisensor RS imagery, rather than only editing prompts.
- [ ] Additionally fine-tune/evaluate on **VRSBench** and **RSVQA** for captioning/grounding/VQA, and on **CDVQA** for change-based VQA, per the SOW's evaluation plan.
- [ ] Persist and document adapter/checkpoint provenance (training data, steps, hyperparameters) so the "adaptation" claim is auditable, not just asserted.

### 12.2 True Text-Guided Grounding
- [ ] Replace the fixed-threshold/largest-contour heuristic with a real grounding model (e.g., a BLIP/GLIP/OWL-ViT-style referring-expression grounder, or a fine-tuned detector conditioned on the query text) so that different natural-language targets (e.g. "the water body" vs. "the building") produce different boxes/masks.
- [ ] Add IoU-based evaluation against VRSBench grounding annotations.

### 12.3 Semantic Change Description / Change-VQA
- [ ] Move beyond a global pixel-difference threshold to a learned change model (e.g., a Siamese/CDVQA-trained change-captioning or change-VQA head) that can answer *"what changed and where"* and *"has the built-up area increased, decreased, or remained unchanged"* in natural language, using BigEarthNet-style land-cover semantics.
- [ ] Where reference masks exist (e.g., OSCD), generate and evaluate a proper spatial change map (precision/recall/IoU against ground truth), not just a raw threshold mask.

### 12.4 Genuine Optical–SAR Fusion
- [ ] Replace the side-by-side composite + canned response-mapping dictionary with real feature-level or decision-level fusion (e.g., dual-encoder architecture combining optical and SAR branches, or a VLM fine-tuned directly on registered optical+SAR pairs from BigEarthNet's Sentinel-1/Sentinel-2 multimodal subset).
- [ ] Ensure the fusion output text is actually conditioned on both modalities' content, not templated from a 3-entry lookup table.

### 12.5 GeoTIFF-Native, Metadata-Aware I/O
- [ ] Read all imagery through `rasterio` (already a listed dependency) to correctly handle multi-band, 16-bit, and georeferenced GeoTIFFs, and to extract CRS/resolution/acquisition metadata for compatibility checks.
- [ ] Enforce the SOW's format rule: GeoTIFF/TIFF for real geospatial imagery; PNG/JPEG accepted only when the source is a named benchmark dataset.

### 12.6 Stronger Agentic Controller
- [ ] Extend `router.py` beyond keyword/image-count heuristics to validate modality (optical vs. SAR), spatial co-registration, band count, and resolution compatibility *before* dispatching a task, and to reject/flag malformed input combinations.
- [ ] Support multi-tool sequencing and output fusion (e.g., run grounding *and* VQA together when a query implies both), rather than a strict one-query-one-tool mapping.
- [ ] Compute confidence scores from actual model signals (e.g., softmax/logit confidence, cross-specialist agreement, or calibration against validation data) instead of hard-coded constants.
- [ ] Make the execution trace's `specialist_model` field accurately reflect whichever component(s) actually executed (e.g., `"OpenCV-ContourGrounding"`, `"NumPy-PixelDiff"`, `"BLIP-VQA-BigEarthNet-Adapted"`) instead of a single fixed label.

### 12.7 Evaluation & Benchmark Harness
- [ ] Add dataset loaders and scoring scripts for BigEarthNet, VRSBench, RSVQA, and CDVQA test splits, producing the normalised metrics the SOW says will be combined at evaluation time.
- [ ] Add a Cartosat-2S/RISAT-aware ingestion path (band order, radiometric scaling, co-registration assumptions) so the prototype can be pointed at the held-out ISRO/SAC evaluation set without code changes.

### 12.8 Deliverable Polish
- [ ] Ship the fine-tuned model weights/adapters alongside code (per "Codes and models including test and demonstration" in the Deliverables section).
- [ ] Extend the "downloadable report" beyond raw JSON to include the rendered visual evidence (boxes/masks/composites) bundled with the report, e.g. as a zipped compliance package.

---

## 13. Deliverables Checklist

| Deliverable (per SOW) | Current State |
|---|---|
| Interactive GUI/web app with agentic RS backend | ✅ Present (Streamlit + `QueryRouter` + `SatelliteTools`) |
| Codes | ✅ Present (`app.py`, `router.py`, `vision_tools.py`, `bigearthnet_adapter.py`) |
| Models (RS-adapted) | ⚠️ Local BLIP checkpoint present, but adaptation is currently prompt-level only — see [§12.1](#121-remote-sensing-model-adaptation-highest-priority) |
| Test & demonstration cases | ✅ Five scripted test cases implemented and documented (§8) |

---

*This README reflects the state of the codebase at the time of writing (BLIP-backed VQA/captioning, OpenCV-based grounding, NumPy-based change detection, composite-image optical–SAR fusion, and a keyword-based agentic router). It is intended both as onboarding documentation and as an honest gap analysis against the full problem statement, to guide the next iteration toward full compliance.*
