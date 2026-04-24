# Dental AI Agent - JSON Data Schema & Enumeration Standard

## Overview

This document defines the data structure and standardized vocabulary for the Dental AI Agent's output. It serves as the single source of truth for AI prompt engineering.

---

## Conversion Principles: Report to Structured JSON

When converting natural language OPG reports to structured JSON using this standard, follow these validated principles:

### 1. Abstraction Over Granularity
- **Generic Terms First:** Use abstract enumerations that cover multiple specific conditions
  - Example: `angled` covers mesioangular, distoangular, and other angular impactions
  - Example: `mucosal_change` covers mucosal thickening, retention cysts, polyps, and masses
  - Example: `degenerative` covers flattening, erosion, sclerosis, and other TMJ degenerative changes

### 2. Semantic Severity Levels
- **Replace Numerical Scales:** Convert numeric classification systems to semantic severity levels
  - ICDAS codes 1-6 → `early`, `moderate`, `advanced`
  - PAI scores 1-5 → `normal`, `mild_change`, `moderate_change`, `severe_change`
  - Periodontal severity → `mild`, `moderate`, `severe`

### 3. Snake_Case Standardization
- **All Enumeration Values:** Must use lowercase with underscores (snake_case)
  - Correct: `root_filled`, `supernumerary`, `air_fluid_level`
  - Incorrect: "Root Filled", "rootFilled", "Root-Filled"

### 4. Sparse Representation Priority
- **Omit Normal Findings:** Only include pathological or noteworthy findings
  - If a tooth is present and healthy → omit from `teeth` object
  - If TMJ is normal → omit entire `tmj` object
  - If sinuses are clear → omit entire `sinuses` object
- **Exception:** `dentition_summary` is always included

### 5. Anatomical Hierarchy (OPG-only findings)

**Strong constraint**: structured reports **must only include findings visible on the OPG (panoramic radiograph)**. If the source text also contains cephalometric / orthodontic measurements, those **must not** be written into the structured report (e.g. overbite, molar class, jaw relationships).

- **Top-Level Keys Only (OPG-only)**: only the following root keys are allowed:
  - `patient`, `dentition_summary`, `teeth`, `periodontium`, `tmj`, `sinuses`, `jaws`, `anatomical_variants`
- **FDI Notation:** Tooth-specific findings must use ISO 3950 notation as string keys
  - Example: `"18"`, `"46"`, `"85"` (not numeric integers)

### 6. Enum Coverage Validation (110-Report Tested)
- **Complete Coverage:** Current enumeration system covers 100% of OPG findings validated across 110 reports
- **Unique Enum Values:** 57 distinct enumeration codes (some values like `mild/moderate/severe` are reused across multiple anatomical contexts)
- **Context-Aware Usage:** Enumerations are applied based on their field context (e.g., `vertical` for impaction angle vs. bone loss pattern)
- **Edge Cases Included:** System handles:
  - Post-surgical states (coronectomy, extractions)
  - Complex impactions (third molars, canines)
  - Developmental anomalies (supernumerary teeth, transpositions)
  - Pathological findings (cysts, lesions, sinus pathology)
  - Anatomical variants (bone islands, stylohyoid ossification)

### 7. Clinical Context Preservation
- **Meaningful Grouping:** Related findings should be grouped under appropriate anatomical keys
  - Periapical pathology → `teeth.{FDI}.pai_score` or `teeth.{FDI}.apical_status`
  - Periodontal disease → `periodontium` (global or quadrant level)
  - Jaw lesions → `jaws.finding`

### 8. Consistency Rules
- **Bilateral Structures:** Use `left`/`right` sub-keys for TMJ and sinuses
- **Severity Modifiers:** Apply `severity` field consistently across multiple regions (sinuses, periodontium)
- **Status vs. Finding:** Use `status` for tooth presence/state, `finding` for pathological discoveries

### 9. No Free Text in Diagnostic Fields
- **Strict Enumeration:** All diagnostic values must match predefined enums
- **Notes Field Exception:** Free text allowed only in optional `notes` or `additional_info` fields (not for primary diagnoses)

### 10. Validation Checkpoint
Before finalizing structured output:
- ✓ All root keys are from the allowed 8 anatomical regions
- ✓ All diagnostic values are snake_case enums from Part II dictionary
- ✓ Normal findings are omitted (sparse representation)
- ✓ FDI notation used correctly for teeth (strings, not integers)
- ✓ No free text in diagnostic fields

---

## Part I: Schema Protocol

The data structure follows the **"Anatomical-Keyed & Sparse Representation"** principle to ensure efficiency and clarity.

### 1. Core Design Logic

* **Root Keys (Anatomical Grouping, OPG-only):**
Data is strictly categorized by anatomical regions. The allowed root keys are:
`dentition_summary`, `teeth`, `periodontium`, `tmj`, `sinuses`, `jaws`, `anatomical_variants`.
* **Keying Strategy:**
* **Teeth:** Use **FDI Notation** (ISO 3950) as string keys (e.g., `"18"`, `"46"`, `"85"`).
* **Bilateral/Single Structures:** Use specific anatomical names (e.g., `maxillary_sinus`, `nasopharyngeal`) or `left`/`right` sub-keys where applicable.


* **Omit If Normal (Sparse Representation):**
* **Field Level:** If a specific pathology (e.g., `caries`) is not detected, the field is omitted.
* **Object Level:** If an entire anatomical region (e.g., `tmj`) is clinically normal, the entire object is **omitted** from the JSON output.


* **Value Strategy:**
* All values must use **snake_case** enumeration codes.
* **No free text** is allowed for diagnostic fields.



### 2. JSON Structure Example

```json
{
  "dentition_summary": {
    "missing_teeth_fdi": ["28"]
  },
  "teeth": {
    "38": {
      "status": "impacted",
      "impaction_angle": "angled",
      "relationship": "approximates_iac"
    }
  },
  "sinuses": {
    "maxillary_sinus": {
      "finding": "mucosal_change",
      "severity": "mild"
    }
  }
  // Note: 'tmj', 'periodontium', etc., are omitted because they are normal.
}

```

---

## Part II: Enumeration Dictionary

The following tables define the strictly allowed standard codes (Enums) for AI generation.

### 1. Teeth Level

**Key Path:** `teeth.{FDI_Number}`

| Field | Category | Allowed Enums | Description |
| --- | --- | --- | --- |
| **status** | Presence | `present` | Default (can be omitted) |
|  |  | `missing` | Missing (congenital or extracted) |
|  |  | `unerupted` | Not yet erupted |
|  |  | `erupted` | erupted |
|  |  | `impacted` | Impacted tooth |
|  |  | `residual_root` | Retained root fragment |
|  |  | `implant` | Dental implant |
|  |  | `root_filled` | Endodontically treated |
|  |  | `supernumerary` | Extra tooth |
|  |  | `exfoliating` | Deciduous tooth near exfoliation |
| **winters_class** | Impaction Angle | `vertical` | Upright orientation |
|  |  | `angled` | Mesial/distal tilt |
|  |  | `horizontal` | Horizontal position |
| **icdas_code** | Caries Severity | `early` | Initial/enamel caries |
|  |  | `moderate` | Dentinal involvement |
|  |  | `advanced` | Cavitated/extensive |
| **pai_score** | Periapical Health | `normal` | Normal structures |
|  |  | `mild_change` | Minor changes |
|  |  | `moderate_change` | Visible radiolucency |
|  |  | `severe_change` | Extensive pathology |
| **relationship** | Vital Structures | `approximates_iac` | Close to IA canal |
|  |  | `approximates_sinus` | Close to sinus |
| **root_anomaly** | Root Issues | `abnormal_morphology` | Curvature/resorption/short |
|  |  | `root_rounding` | Blunting from orthodontics |
| **crown_anomaly** | Crown Issues | `abnormal_morphology` | Size/position/wear |
|  |  | `developmental_anomaly` | Talon cusp/invagination/other |
| **position_anomaly** | Position | `transposed` | Abnormal position exchange |
|  |  | `insufficient_space` | Inadequate eruption space |
|  |  | `ectopic_eruption` | Abnormal eruption path |
| **caries_location** | Caries Type | `recurrent` | Beneath restoration |
|  |  | `root_surface` | Root caries |
| **restoration_issue** | Restoration | `defective` | Deficiency/leakage |

### 2. Periodontal Level

**Key Path:** `periodontium` (Global or Quadrant level)
*Standards based on AAP/EFP 2017 Classification.*

| Field | Allowed Enums | Description |
| --- | --- | --- |
| **severity** | `mild` | Early/minor changes |
|  | `moderate` | Clear involvement |
|  | `severe` | Advanced disease |
| **bone_loss_pattern** | `horizontal` | Even bone loss |
|  | `vertical` | Angular defects |
| **findings** | `calculus` | Deposits present |
|  | `pericoronitis` | Crown inflammation |

### 3. Jaws & TMJ

**Key Path:** `jaws`, `tmj`

| Region | Field | Allowed Enums | Description |
| --- | --- | --- | --- |
| **TMJ** | **morphology** | `normal` | Normal appearance |
|  |  | `degenerative` | Flattening/erosion/arthropathy |
|  |  | `developmental` | Hypoplasia/bifid condyle |
| **Jaws** | **finding** | `sclerotic_lesion` | High density lesion |
|  |  | `lucent_lesion` | Low density lesion |
|  |  | `bone_variant` | Normal variant (tori/island) |
|  |  | `osteopenia` | Decreased bone density |
|  |  | `marrow_space_prominence` | Enlarged marrow cavity |
|  |  | `surgical_hardware` | Plates/screws/fixation |

### 4. Sinuses (OPG-only)

**Key Path:** `sinuses`

| Region | Field | Allowed Enums | Description |
| --- | --- | --- | --- |
| **Sinus** | **finding** | `mucosal_change` | Thickening/cyst/mass |
|  |  | `opacification` | Complete opacification |
|  |  | `air_fluid_level` | Acute sinusitis/fluid |
| **Generic** | **severity** | `mild` | Slight deviation |
|  |  | `moderate` | Clear pathology |
|  |  | `severe` | Significant pathology |

---

### 6. Anatomical Variants & Incidental Findings

**Key Path:** `anatomical_variants`

| Finding | Description |
| --- | --- |
| **variant_present** | Use specific descriptor (e.g., `bridged_sella`, `stylohyoid_ossification`, `retained_fragment`) |

---

### 7. Apical Pathology Extensions

**Key Path:** `teeth.{FDI_Number}`  

| Field | Allowed Enums | Description |
| --- | --- | --- |
| **apical_status** | `scar` | Non-active scar tissue |
|  | `active_pathology` | Active infection/lesion |

---