# CrackVision

CrackVision is a local React + FastAPI structural-inspection application. This repository now contains its detection, crack-segmentation, video, mapping, and evidence-grounded VQA backend, with the three inference checkpoints recovered from the sibling project archive.

## Model and label coverage

| Module | Model/evidence | Supported result |
|---|---|---|
| Image detection | `backend/app/models/detector.pt` | Detector boxes for `crack`, `fissures`, `rebar`, and `concrete spalling` |
| Crack segmentation | `backend/app/models/segmentation_v4_best.pth` | Pixel mask and pixel-space crack geometry |
| Video detection | `backend/app/models/video_detector_best_reconstructed.pt` | Frame-wise boxes for the same four detector labels |
| Crack mapping | V4 segmentation + `mapping_normalization_reference.json` | Crack geometry and an exploratory image-space priority index |
| Non-crack mask proposals | `backend/app/models/FastSAM-s.pt` + detector boxes | Class-agnostic FastSAM masks associated with `fissures`, `rebar`, and `concrete spalling` detector boxes |

The detector checkpoints expose all four classes above. The separate `damage_multiclass_v1.pth` checkpoint is not present. For non-crack labels, the app now uses the YOLO detector's label and box to select an overlapping class-agnostic FastSAM object proposal. These masks can support visual review and exploratory image-space mapping, but they are not supervised class-specific segmentations and their areas are not validated defect measurements. The dataset label `concrete crushing` is not in the detector classes and must not be reported as a supported prediction.

## VQA image pipeline

`POST /api/vqa/inspect-and-ask` saves an image inspection and runs image detection, crack segmentation, detector-guided FastSAM proposals, and crack mapping before generating the answer. Each stage is recorded with its own success/failure status. A failed stage is never represented as a measured zero. Detector class counts stay separate from crack-mask geometry and mapping values.

Offline mode answers from the saved model outputs and the source-linked evidence rules in `backend/app/services/vqa_screen.py`. It does not infer labels from pixels when there is no model evidence. Optional cloud image analysis is available through OpenAI Responses or Google Gemini when offline mode is disabled and the user supplies a key. Keys are used only for that request and are not stored in the backend database. Cloud observations are identified separately from CrackVision detector predictions.

Damage mapping returns a separate external-validation status for crack, fissures, spalling, and rebar. VQA includes the same class-wise benchmark summary in its answer, evidence, and saved report. The current external mask validation found limited crack-map performance, poor fissure-proxy results, and no matching ground truth to validate spalling or rebar; those statuses are shown explicitly instead of being inferred from detector boxes. See [`backend/validation_reports/README.md`](backend/validation_reports/README.md) and [`backend/validation_reports/class_validation_summary.json`](backend/validation_reports/class_validation_summary.json). These are benchmark-level results and do not validate an individual uploaded image.

### Class follow-up and severity

- **Crack:** document location, pattern, field-measured width, leakage, and change over time. ACI says significance depends on structure and crack characteristics; the image does not establish cause, depth, or activity. [ACI crack guidance](https://www.concrete.org/frequentlyaskedquestions.aspx?faqid=809)
- **Fissures:** handled as fine crack-like findings. Confirm at close range, map distribution, and check for movement/moisture before selecting a repair. [ACI 562 assessment and durability provisions](https://www.concrete.org/Portals/0/Files/PDF/562-16_Provisional.pdf)
- **Spalling:** qualified field inspection should map the area and check for unsound/delaminated concrete. Loose or overhead concrete calls for keeping people clear and prompt site inspection. FHWA’s bridge protocol records dimensions and exposed steel; its partial-depth repair procedure is pavement-specific. [FHWA bridge protocol](https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/047.cfm) · [FHWA pavement procedure](https://www.fhwa.dot.gov/Pavement/concrete/repair04.cfm)
- **Rebar:** prompt qualified inspection should document exposed steel and measure section loss against reliable original dimensions. A photo cannot establish remaining bar area or capacity; corrosion cause belongs in the repair design. [FHWA bridge protocol](https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/047.cfm) · [ACI 562](https://www.concrete.org/Portals/0/Files/PDF/562-16_Provisional.pdf)

The severity banner is qualitative action triage. It does not map pixel area, detector count, crack width, or the exploratory mapping index to structural severity or failure probability. The system does not certify safety or prescribe repair materials.

## Run locally

Requirements: Python 3.11 x64 and Node.js 20+.

On Windows, run `setup_windows.bat`, then `start_windows.bat`. Open `http://127.0.0.1:5173`; API documentation is at `http://127.0.0.1:8000/docs`.

Manual backend setup from the project root:

```powershell
py -3.11 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

For cloud VQA, enter an OpenAI or Gemini key in the VQA page and turn off offline mode. The image is sent to the selected provider.

## API routes

The backend also provides the project, dashboard, inspection-history, detection, crack-segmentation, video, and crack-mapping endpoints used by the frontend. VQA-specific routes are:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/vqa/config` | VQA engines and supported labels |
| `POST` | `/api/vqa/inspect-and-ask` | Run an image pipeline and answer a question |
| `POST` | `/api/vqa/ask` | Ask about a saved inspection or use the project-evidence VQA contract |
| `GET` | `/api/vqa/status` | Environment-configured project VQA provider status |
| `GET` | `/api/inspections/{id}/evidence` | Normalized saved observations |

`/api/vqa/ask` supports the React inspector's `inspection_id` request and the project VQA request with `project_id` and optional inspection scope. The project-evidence mode uses `VQA_API_BASE_URL`, `VQA_MODEL`, and optionally `VQA_API_KEY`; visual inputs also need `VQA_VISION_MODEL`.

## Verify

```powershell
backend\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
```

The VQA tests cover each detector label, source-linked class guidance, telemetry preservation, severity independence from pixel area, and the upload → saved inspection → ask flow. A live OpenAI/Gemini request requires the user's provider key.

Reference outputs under `assets/reference_outputs/` are archived examples, not current predictions. All pixel measurements are image-space unless explicitly calibrated. The V4 mapping index is exploratory and not a safety score.
