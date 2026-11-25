# External fissure and damage-map validation

## Dataset and scope

Validation uses NCCD-PF, a public set of real bridge/tunnel concrete images with paired expert-prepared pixel masks. Its target is narrow cracks below 0.3 mm, so it is a fissure-like proxy, not a dataset with an explicit `fissures` class. The dataset has 5,388 image/mask pairs; the source paper reports that an experienced bridge engineer prepared and reviewed the crack masks. Dataset: [Zenodo 10.5281/zenodo.8215100](https://zenodo.org/records/8215100), CC BY 4.0; data paper: [Scientific Data, DOI 10.1038/s41597-023-02839-z](https://www.nature.com/articles/s41597-023-02839-z).

No NCCD-PF images were copied into this repository. The archive was downloaded and unpacked under the system temp directory for these runs.

## Results

`nccdp_fissures_mapping.json` records a deterministic sample of 30 positive narrow-crack images and 20 clean controls through detection and crack mapping:

- Generic `crack` or `fissures` detection recall on positives: **12/30 (40%)**.
- Strict detector `fissures` recall on narrow-crack positives: **1/30 (3.3%)**.
- Specificity for either crack label on controls: **20/20 (100%)**.
- V4 crack-map mean IoU on positive images: **0.102**; mean Dice: **0.125**. Many positive masks were missed. The all-sample IoU of 0.461 is elevated by empty predictions correctly matching clean controls and should not be used as the positive-class score.
- One fissures detector-positive case (`image_2433`) had no matched FastSAM fissure proposal. V4's generic crack mask scored IoU **0.629** / Dice **0.772** against that case's expert mask; the strict fissures class map scored **0.000**.

`nccdp_damage_map.json` records the separate class-labelled damage-map endpoint on 8 positive narrow-crack images and 4 clean controls:

- Positive crack-or-fissures map mean IoU: **0.000**; strict fissures map mean IoU: **0.000**.
- All 4 controls correctly produced empty crack/fissures maps.
- The narrow-crack dataset cannot validate spalling or rebar classes; it has no such ground-truth labels.

The broad dataset search also identified MDMCS (1,200 images with pixel labels for cracking, spalling, corrosion and exposed rebar; [Mendeley Data, CC BY 4.0](https://data.mendeley.com/datasets/6x4dzzrs2h/2)) and DACL10k (bridge masks including crack, spalling and exposed rebars; [dataset toolkit, CC BY-NC 4.0](https://github.com/phiyodr/dacl10k-toolkit)). Neither annotation archive is installed in this validation run: MDMCS's advertised 195 MB file endpoint returned 404, and DACL10k's official development archive is 4.8 GB. Their published labels establish that stronger class-matched validation is possible, but are not evidence of this project's accuracy. No class-specific spalling or rebar scores are claimed.

These results validate execution and reveal poor recall for thin external cracks. The fresh multi-class run also found a false-map metric bug: an all-zero class-count dictionary is truthy in Python. The validator now checks actual matched mask counts; the corrected sample result is 0/6 false damage masks on controls. These results do **not** validate a distinct fissures class, class-specific FastSAM masks, spalling/rebar maps, physical crack measurements, severity, or structural safety. The existing repository's 1,200-image CrackVision12K mapping metrics are archived outputs from that model's research run; they are separate from this external NCCD-PF check. The module produces image-space damage maps, not geographic disaster maps.

`class_validation_summary.json` is the per-label status used at runtime. The `/api/mapping/damage/image` response includes this summary beside its per-class image measurements, and VQA adds the same benchmark-level status for crack, fissures, spalling, and rebar to its saved report and answer. Classes without matched ground-truth masks are explicitly reported as not independently validated; these benchmark figures do not validate the current uploaded image.

## Reproduce

Run the backend at port 8001, download/unpack the NCCD-PF archive from the Zenodo record, then from `backend/` run:

```powershell
..\.venv\Scripts\python.exe scripts\validate_nccdp_fissures_mapping.py `
  --dataset-root "<NCCD-PF_Dataset>\Dataset_for_semantic_segmentation" `
  --api http://127.0.0.1:8001 --positive-count 30 --negative-count 20

..\.venv\Scripts\python.exe scripts\validate_nccdp_damage_map.py `
  --dataset-root "<NCCD-PF_Dataset>\Dataset_for_semantic_segmentation" `
  --api http://127.0.0.1:8001 --positive-count 8 --negative-count 4
```
