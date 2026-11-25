import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import main
from app.database import Base
from app.entities import Inspection, Project
from app.services.class_validation import CLASS_ORDER, class_validation_summary
from app.services.vqa_screen import LABELS, _label_key, assessment, evidence_items, normalize_telemetry, offline_answer, present_labels, triage


class VqaEvidenceTests(unittest.TestCase):
    def test_all_detector_labels_have_class_specific_evidence(self):
        telemetry = normalize_telemetry({
            "class_counts": {"crack": 1, "concrete spalling": 2, "rebar": 3, "fissures": 4, "concrete crushing": 9},
            "class_metrics": {
                "crack": {"pixel_area": 120, "coverage_percent": 0.2},
                "concrete spalling": {"pixel_area": 80, "component_count": 1},
                "exposed rebar": {"pixel_area": 35},
                "fissures": {"pixel_area": 9},
                "concrete crushing": {"pixel_area": 1000},
            },
        })
        self.assertEqual(set(present_labels(telemetry)), set(LABELS))
        cards = evidence_items(telemetry)
        self.assertEqual({card["label"] for card in cards}, set(LABELS))
        self.assertTrue(all(card["follow_up"] and card["sources"] for card in cards))

    def test_detector_names_and_microcrack_aliases(self):
        self.assertEqual(_label_key("micro-cracks"), "fissures")
        self.assertEqual(_label_key("concrete crack"), "crack")
        self.assertEqual(_label_key("concrete spalling"), "spalling")

    def test_pixel_area_does_not_change_qualitative_triage(self):
        small = {"class_counts": {"spalling": 1}, "damage_classes": {"spalling": {"pixel_area": 1}}}
        large = {"class_counts": {"spalling": 1}, "damage_classes": {"spalling": {"pixel_area": 10_000_000}}}
        self.assertEqual(triage(small), triage(large))
        self.assertEqual(triage({"class_counts": {"crack": 1}})["severity"], "Field verification required")

    def test_findings_sort_by_pixel_coverage_without_severity_bands(self):
        telemetry = normalize_telemetry({
            "class_counts": {"crack": 1, "fissures": 1, "spalling": 1, "rebar": 1},
            "class_mapping": {
                "grid": [20, 20],
                "classes": {
                    "crack": {"pixel_area": 12, "coverage_percent": 1.2},
                    "fissures": {"pixel_area": 40, "coverage_percent": 4.0},
                    "spalling": {"pixel_area": 5, "coverage_percent": 0.5},
                    "rebar": {"pixel_area": 28, "coverage_percent": 2.8},
                },
            },
        })
        answer, findings, _ = offline_answer("Summarize findings", telemetry, "Bridge")
        self.assertEqual([item["label"] for item in findings], ["fissures", "rebar", "crack", "spalling"])
        self.assertEqual(findings[0]["pixel_extent"], {"pixel_area": 40, "coverage_percent": 4.0})
        self.assertIn("Image pixel extent (not structural severity)", answer)
        self.assertLess(answer.index("### Fissures"), answer.index("### Exposed / corroded reinforcement"))
        self.assertIn("Structural severity is not rated from pixels", answer)
        self.assertNotIn("1–<5%", answer)

    def test_severity_guidance_for_each_class_uses_field_checks_not_pixel_tiers(self):
        result = assessment(
            "Assess severity",
            {"class_counts": {"crack": 1}},
            "Bridge",
            True,
            None,
            None,
            "image/png",
        )
        severity_guidance = result["report"]["severity_guidance"]
        self.assertEqual(tuple(severity_guidance), CLASS_ORDER)
        self.assertTrue(all(item["status"] == "not_assessable_from_image" for item in severity_guidance.values()))
        self.assertIn("physical width", severity_guidance["crack"]["basis"])
        self.assertIn("not a separate engineering severity class", severity_guidance["fissures"]["basis"])
        self.assertIn("loose or overhead", severity_guidance["spalling"]["escalation"])
        self.assertIn("section loss", severity_guidance["rebar"]["field_checks"])
        self.assertTrue(all(item["sources"] for item in severity_guidance.values()))
        self.assertIn("Class-specific severity checks", result["answer"])
        self.assertIn("Structural severity is not rated from image pixels", result["answer"])

    def test_offline_answer_has_class_evidence_units_and_sources(self):
        telemetry = {
            "class_counts": {"crack": 1, "fissures": 1, "concrete spalling": 1, "rebar": 1},
            "damage_classes": {"crack": {"pixel_area": 10}, "fissures": {"pixel_area": 5}, "spalling": {"pixel_area": 7}, "rebar": {"pixel_area": 3}},
        }
        answer, labels, _ = offline_answer("Summarize findings", telemetry, "Concrete Beam")
        self.assertEqual(len(labels), 4)
        self.assertIn("Concrete Beam", answer)
        self.assertIn("concrete.org", answer)
        self.assertIn("fhwa.dot.gov", answer)
        self.assertIn("not a code-compliant condition rating", answer)

    def test_vqa_includes_source_linked_actions_for_every_supported_label(self):
        result = assessment(
            "What should I check?",
            {"class_counts": {"crack": 1}},
            "Bridge",
            True,
            None,
            None,
            "image/png",
        )
        solutions = result["report"]["evidence_based_solutions"]
        self.assertEqual(tuple(item["label"] for item in solutions), CLASS_ORDER)
        self.assertTrue(all(item["solution"] and item["sources"] for item in solutions))
        self.assertTrue(all(source["url"].startswith("https://") for item in solutions for source in item["sources"]))
        self.assertIn("Evidence-based field actions for supported labels", result["answer"])
        self.assertIn("not findings for this image", result["answer"])
        self.assertIn("sound concrete", result["answer"])
        self.assertIn("section loss", result["answer"])
        self.assertIn("severity_guidance", result["report"])

    def test_mapping_drives_class_specific_vqa_suggestions_with_image_cells(self):
        telemetry = normalize_telemetry({
            "class_counts": {"crack": 1, "fissures": 1, "concrete spalling": 1, "rebar": 1},
            "class_mapping": {
                "grid": [20, 20],
                "coordinate_note": "Image grid only.",
                "classes": {
                    name: {
                        "pixel_area": 48,
                        "coverage_percent": 0.0732,
                        "component_count": 1,
                        "mapped_cell_count": 1,
                        "top_cells": [{"row": index, "column": index + 1, "cell_coverage_percent": 5.0}],
                        "mask_method": "CRACKTO-SFNet V4" if name == "crack" else "Class-agnostic FastSAM proposal matched to detector box",
                    }
                    for index, name in enumerate(("crack", "fissures", "spalling", "rebar"), start=1)
                },
            },
        })
        findings = evidence_items(telemetry)
        by_label = {item["label"]: item for item in findings}
        self.assertEqual(set(by_label), set(LABELS))
        self.assertIn("row 1, column 2", by_label["crack"]["mapping_suggestion"])
        self.assertIn("crack gauge", by_label["crack"]["mapping_suggestion"])
        self.assertIn("row 3, column 4", by_label["spalling"]["mapping_suggestion"])
        self.assertIn("sound suspected concrete", by_label["spalling"]["mapping_suggestion"])
        self.assertIn("section loss", by_label["rebar"]["mapping_suggestion"])
        self.assertIn("class-agnostic FastSAM proposal", by_label["rebar"]["mapping_suggestion"])
        self.assertIn("not a separate engineering severity class", by_label["fissures"]["mapping_suggestion"])
        answer, _, _ = offline_answer("What should I check next?", telemetry, "Bridge")
        self.assertIn("Mapping-based suggestion", answer)
        self.assertIn("FHWA concrete bridge cracking field protocol", answer)
        self.assertIn("image grid cells are not site or geographic coordinates", " ".join(e["units"] for e in by_label["crack"]["evidence"] if e["type"] == "class_mapping"))

    def test_detector_label_without_mapped_pixels_is_not_reported_as_clear(self):
        telemetry = normalize_telemetry({"class_counts": {"spalling": 1}, "class_mapping": {"grid": [20, 20], "classes": {"spalling": {"pixel_area": 0, "coverage_percent": 0, "mapped_cell_count": 0, "top_cells": []}}}})
        item = evidence_items(telemetry)[0]
        self.assertIn("no mapped pixels", item["mapping_suggestion"])
        self.assertIn("not evidence that the defect is absent", item["mapping_suggestion"])

    def test_vqa_reports_separate_external_mapping_validation_for_all_classes(self):
        validation = class_validation_summary()
        self.assertEqual(tuple(validation["classes"]), CLASS_ORDER)
        answer, cards, _ = offline_answer("Summarize findings", {"class_counts": {"fissures": 1}}, "Bridge")
        self.assertIn("Class-wise mapping validation", answer)
        self.assertIn("strict fissures detections were 1/30", answer)
        self.assertIn("No class-matched pixel ground truth", answer)
        self.assertEqual(cards[0]["mapping_validation"]["status"], "poor_proxy_performance")

    def test_detector_guided_mask_provenance_is_clear_in_vqa_answer(self):
        telemetry = normalize_telemetry({
            "class_counts": {"concrete spalling": 1},
            "damage_classes": {"spalling": {
                "pixels": 256, "coverage_percent": 0.4,
                "mask_method": "FastSAM proposal matched to detector box",
            }},
        })
        cards = evidence_items(telemetry)
        self.assertEqual(cards[0]["evidence"][0]["type"], "detector_guided_mask")
        answer, _, _ = offline_answer("What is visible?", telemetry, "Bridge")
        self.assertIn("Class-agnostic FastSAM proposal", answer)
        self.assertIn("not a validated defect measurement", answer)

    def test_upload_runs_saved_pipeline_and_ask_returns_same_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_engine = create_engine(f"sqlite:///{(temp_path / 'test.sqlite').as_posix()}", connect_args={"check_same_thread": False})
            Base.metadata.create_all(test_engine)
            test_sessions = sessionmaker(bind=test_engine, expire_on_commit=False)
            with test_sessions() as session:
                session.add(Project(name="Test site", structure_type="Building"))
                session.commit()

            def override_db():
                session = test_sessions()
                try:
                    yield session
                finally:
                    session.close()

            main.app.dependency_overrides[main.get_db] = override_db
            detections = {
                "model_checkpoint": "detector.pt", "annotated_image_url": "/storage/test/annotated.jpg",
                "source_width": 24, "source_height": 24, "detection_count": 4,
                "class_counts": {"crack": 1, "fissures": 1, "rebar": 1, "concrete spalling": 1},
                "detections": [],
            }
            segmentation = {
                "mask_url": "/storage/test/mask.png", "overlay_url": "/storage/test/overlay.png",
                "source_width": 24, "source_height": 24, "crack_pixels": 12,
                "mask_coverage_percent": 2.08, "contour_count": 1,
                "total_contour_length_px": 10.0, "estimated_median_width_px": 1.0,
            }
            mapping = {"metrics": {"crack_area_ratio": 0.02, "crack_length": 10, "width_p95": 2}, "hotspot_count": 1, "max_priority": 0.8}
            damage_segmentation = {
                "class_metrics": {"spalling": {
                    "pixels": 5, "coverage_percent": 0.1, "component_count": 1,
                    "mask_method": "FastSAM proposal matched to detector box",
                }},
                "model": "CRACKTO-SFNet V4 + detector-guided FastSAM proposals",
                "overlay_url": "/storage/test/damage-overlay.png",
            }
            def fake_segmentation(input_path, output_dir, *args):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                mask = Image.new("L", (24, 24), 0)
                mask.putpixel((3, 3), 255)
                mask.save(output_dir / "segmentation_mask.png")
                return segmentation

            def fake_damage_segmentation(input_path, output_dir, *args):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                class_ids = Image.new("L", (24, 24), 0)
                class_ids.putpixel((12, 12), 2)
                class_ids.save(output_dir / "damage_class_ids.png")
                return damage_segmentation

            with patch.object(main, "STORAGE_DIR", temp_path / "storage"), \
                 patch.object(main, "storage_url", side_effect=lambda path: f"/storage/{Path(path).name}"), \
                 patch.object(main.detector, "detect_image", return_value=detections), \
                 patch.object(main.segmenter, "segment_image", side_effect=fake_segmentation), \
                 patch.object(main.damage_segmenter, "segment_image", side_effect=fake_damage_segmentation), \
                 patch.object(main.mapping_service, "map_image", return_value=mapping):
                (temp_path / "storage").mkdir()
                image = BytesIO()
                Image.new("RGB", (24, 24), "gray").save(image, "PNG")
                with TestClient(main.app) as client:
                    response = client.post(
                        "/api/vqa/inspect-and-ask",
                        data={"question": "Summarize all detected classes", "offline_mode": "true", "project_id": "1"},
                        files={"file": ("sample.png", image.getvalue(), "image/png")},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    data = response.json()
                    record_id = data["inspection_id"]
                    self.assertEqual(data["inspection"]["status"], "succeeded")
                    self.assertEqual(set(data["telemetry"]["class_counts"]), set(detections["class_counts"]))
                    self.assertIn("Fissures", data["answer"])
                    self.assertIn("Concrete spalling", data["answer"])
                    self.assertIn("Exposed / corroded reinforcement", data["answer"])
                    self.assertEqual(data["report"]["stage_statuses"]["detection"], "succeeded")
                    self.assertEqual(data["report"]["stage_statuses"]["damage_segmentation"], "succeeded")
                    self.assertEqual(data["report"]["stage_statuses"]["label_mapping"], "succeeded", data["stages"]["label_mapping"])
                    self.assertEqual(set(data["telemetry"]["class_mapping"]["classes"]), set(LABELS))
                    self.assertIn("Mapping-based suggestion", data["answer"])
                    spall_cell = data["telemetry"]["class_mapping"]["classes"]["spalling"]["top_cells"][0]
                    self.assertIn(f"row {spall_cell['row']}, column {spall_cell['column']}", data["answer"])
                    self.assertIn("Class-agnostic FastSAM proposal", data["answer"])
                    self.assertEqual(set(data["report"]["class_validation"]["classes"]), set(LABELS))
                    self.assertEqual(
                        {item["label"] for item in data["report"]["evidence_based_solutions"]},
                        set(LABELS),
                    )
                    self.assertIn("Class-wise mapping validation", data["answer"])
                    listed = client.get("/api/inspections?limit=10")
                    self.assertEqual(listed.status_code, 200)
                    asked = client.post("/api/vqa/ask", json={"inspection_id": record_id, "question": "Summarize labels", "offline_mode": True})
                    self.assertEqual(asked.status_code, 200, asked.text)
                    self.assertEqual(asked.json()["telemetry"]["class_counts"], data["telemetry"]["class_counts"])
                    cloud_without_key = client.post("/api/vqa/ask", json={"inspection_id": record_id, "question": "Summarize labels", "offline_mode": False})
                    self.assertEqual(cloud_without_key.status_code, 422)
            main.app.dependency_overrides.pop(main.get_db, None)
            test_engine.dispose()

    def test_aci_threshold_question_uses_combined_telemetry_and_calibrated_mm(self):
        telemetry = normalize_telemetry({
            "combined_modules": ["detection", "segmentation", "mapping"],
            "stages": {
                "detection": {"status": "succeeded", "result": {"detection_count": 2, "class_counts": {"crack": 2}, "maximum_confidence": 0.91}},
                "segmentation": {"status": "succeeded", "result": {"crack_pixels": 17463, "mask_coverage_percent": 4.657, "contour_count": 6, "total_contour_length_px": 1200.0, "estimated_median_width_px": 5.2}},
                "mapping": {"status": "succeeded", "result": {
                    "model": "CRACKTO-SFNet V4",
                    "grid": [20, 20],
                    "occupied_cells": 83,
                    "hotspot_count": 21,
                    "max_priority": 0.78,
                    "metrics": {"crack_pixels": 2785, "crack_area_ratio": 0.0424957, "crack_length": 826.1, "width_median": 4.0, "width_p95": 8.0, "width_max": 12.6, "junction_clusters": 7, "mask_components": 6},
                    "top_cells": [{"grid_row": 6, "grid_col": 1, "crack_area_ratio": 0.858, "priority_index": 0.78, "width_p95": 8.0, "junction_count": 1}],
                }},
            },
        })
        answer_px, _, _ = offline_answer("How do widths compare to ACI 224R threshold?", telemetry, "Concrete Beam")
        self.assertIn("ACI 224R-01", answer_px)
        self.assertIn("0.41 mm", answer_px)
        self.assertNotIn("coverage coverage not available", answer_px)

        summary_ans, _, _ = offline_answer("Summarize the combined detection, segmentation, and mapping findings.", telemetry, "Concrete Beam")
        self.assertIn("Detection (YOLO)", summary_ans)
        self.assertIn("Segmentation (V4)", summary_ans)
        self.assertIn("Damage Mapping", summary_ans)
        self.assertIn("2 box(es) [crack: 2]", summary_ans)

        calibrated = {**telemetry, "physical": {"width_median_mm": 0.22, "width_p95_mm": 0.35, "width_max_mm": 0.50, "crack_length_mm": 420.0}}
        answer_mm, _, _ = offline_answer("Check ACI threshold", calibrated, "Concrete Beam")
        self.assertIn("EXCEEDS", answer_mm)
        self.assertIn("within limit", answer_mm)

    def test_vqa_ask_combines_detection_segmentation_and_mapping_for_same_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_engine = create_engine(f"sqlite:///{(temp_path / 'test.sqlite').as_posix()}", connect_args={"check_same_thread": False})
            Base.metadata.create_all(test_engine)
            test_sessions = sessionmaker(bind=test_engine, expire_on_commit=False)
            storage_root = temp_path / "storage"
            storage_root.mkdir()
            with test_sessions() as session:
                session.add(Project(name="Test site", structure_type="Building"))
                session.commit()
                for insp_id, mod, res in (
                    ("det-1", "detection", {"detection_count": 2, "class_counts": {"crack": 2}, "maximum_confidence": 0.88}),
                    ("seg-1", "segmentation", {"crack_pixels": 17463, "mask_coverage_percent": 4.657, "contour_count": 4, "total_contour_length_px": 950.0, "estimated_median_width_px": 4.5}),
                    ("map-1", "mapping", {"kind": "image", "model": "CRACKTO-SFNet V4", "grid": [20, 20], "occupied_cells": 83, "hotspot_count": 21, "max_priority": 0.78, "metrics": {"crack_pixels": 2785, "crack_area_ratio": 0.0424957, "crack_length": 826.1, "width_median": 4.0, "width_p95": 8.0, "width_max": 12.6, "junction_clusters": 7, "mask_components": 6}, "top_cells": [{"grid_row": 6, "grid_col": 1, "crack_area_ratio": 0.858, "priority_index": 0.78, "width_p95": 8.0, "junction_count": 1}]}),
                ):
                    (storage_root / insp_id).mkdir()
                    Image.new("RGB", (24, 24), "gray").save(storage_root / insp_id / "original.png", "PNG")
                    import json as _json
                    session.add(Inspection(
                        id=insp_id, project_id=1, module=mod, status="succeeded", source_filename="2nd.png",
                        settings_json=_json.dumps({"asset_fingerprint": "same-sha256"}),
                        result_json=_json.dumps(res),
                    ))
                session.commit()

            def override_db():
                session = test_sessions()
                try:
                    yield session
                finally:
                    session.close()

            main.app.dependency_overrides[main.get_db] = override_db
            with patch.object(main, "STORAGE_DIR", storage_root):
                with TestClient(main.app) as client:
                    combined_resp = client.get("/api/vqa/combined-telemetry/map-1")
                    self.assertEqual(combined_resp.status_code, 200, combined_resp.text)
                    ct = combined_resp.json()["telemetry"]
                    self.assertEqual(ct["detection_count"], 2.0)
                    self.assertEqual(ct["class_counts"], {"crack": 2})
                    self.assertEqual(ct["crack_pixels"], 17463.0)
                    self.assertEqual(ct["hotspot_count"], 21.0)
                    self.assertEqual(ct["mapping_junctions"], 7.0)

                    asked = client.post("/api/vqa/ask", json={
                        "inspection_id": "map-1",
                        "question": "Summarize the combined detection, segmentation, and mapping findings.",
                        "structure_type": "Concrete Beam",
                        "offline_mode": True,
                    })
                    self.assertEqual(asked.status_code, 200, asked.text)
                    body = asked.json()
                    self.assertEqual(body["telemetry"]["detection_count"], 2.0)
                    self.assertEqual(body["telemetry"]["class_counts"], {"crack": 2})
                    self.assertIn("2 box(es) [crack: 2]", body["direct_answer"])
                    self.assertIn("17,463 px²", body["direct_answer"])
                    self.assertIn("83 occupied cells", body["direct_answer"])
                    self.assertNotIn("coverage coverage not available", body["answer"])
                    self.assertEqual(body["aci_level"], "MODERATE")
                    self.assertIn("Take care tomorrow", body["action_timeline"])
                    self.assertNotIn("Assessment scope", body["direct_answer"])
                    self.assertNotIn("Class-wise mapping validation", body["direct_answer"])

                    aci_resp = client.post("/api/vqa/ask", json={
                        "inspection_id": "map-1",
                        "question": "What is the ACI crack width threshold?",
                        "structure_type": "Concrete Beam",
                        "offline_mode": True,
                    })
                    self.assertEqual(aci_resp.status_code, 200)
                    self.assertIn("ACI 224R-01", aci_resp.json()["direct_answer"])
                    self.assertNotIn("Detection (YOLO)", aci_resp.json()["direct_answer"])

                    # Severity-only question should stay focused on ACI severity & timeline
                    sev_resp = client.post("/api/vqa/ask", json={
                        "inspection_id": "map-1",
                        "question": "Is this critical, moderate, or low? When should we fix it?",
                        "structure_type": "Concrete Beam",
                        "offline_mode": True,
                    })
                    self.assertEqual(sev_resp.status_code, 200)
                    self.assertIn("ACI Severity: MODERATE", sev_resp.json()["direct_answer"])
                    self.assertIn("Take care tomorrow", sev_resp.json()["direct_answer"])

                    # PDF report export should return valid PDF bytes
                    pdf_resp = client.post("/api/vqa/report/pdf", json={
                        "inspection_id": "map-1",
                        "structure_type": "Concrete Beam",
                        "aci_level": "MODERATE",
                        "action_timeline": body["action_timeline"],
                        "aci_basis": body["aci_basis"],
                        "telemetry": body["telemetry"],
                        "qa_items": [{"question": "What is the ACI crack width threshold?", "answer": body["direct_answer"]}],
                    })
                    self.assertEqual(pdf_resp.status_code, 200)
                    self.assertEqual(pdf_resp.headers["content-type"], "application/pdf")
                    self.assertTrue(pdf_resp.content.startswith(b"%PDF-"))
            main.app.dependency_overrides.pop(main.get_db, None)
            test_engine.dispose()

    def test_aci_triage_levels_critical_moderate_low(self):
        self.assertEqual(triage({"class_counts": {"rebar": 1}})["aci_level"], "CRITICAL")
        self.assertIn("Take care now", triage({"class_counts": {"spalling": 1}})["action_timeline"])
        self.assertEqual(triage({"class_counts": {"crack": 1}, "physical": {"width_p95_mm": 0.50}})["aci_level"], "CRITICAL")
        self.assertEqual(triage({"class_counts": {"crack": 1}, "physical": {"width_p95_mm": 0.25}})["aci_level"], "MODERATE")
        self.assertEqual(triage({"class_counts": {"crack": 1}})["aci_level"], "LOW")
        self.assertEqual(triage({"class_counts": {"crack": 1}})["label_risks"][0]["risk"], "LOW")
        self.assertEqual(triage({"class_counts": {"fissures": 1}})["aci_level"], "LOW")
        self.assertEqual(triage({"class_counts": {"rebar": 1, "crack": 1}})["label_risks"][0]["risk"], "CRITICAL")

    def test_multilabel_camera_calibration_mm_flows_through_all_labels_and_vqa(self):
        import numpy as np
        from app.services.damage_mapping_core import CLASS_IDS, measure_damage_map
        from app.services.mapping import build_calibration

        cal = build_calibration("sut_preset", None)
        self.assertTrue(cal["enabled"])
        scale_xy = (cal["scale_x_mm_per_px"], cal["scale_y_mm_per_px"])

        class_map = np.zeros((256, 256), dtype=np.uint8)
        class_map[20:24, 30:100] = CLASS_IDS["crack"]       # 4 px thick crack
        class_map[60:90, 60:90] = CLASS_IDS["spalling"]     # 30x30 spall patch
        class_map[120:124, 40:120] = CLASS_IDS["rebar"]     # exposed bar
        class_map[160:162, 50:110] = CLASS_IDS["fissures"]  # 2 px fine fissure

        analysis = measure_damage_map(class_map, scale_xy=scale_xy)
        cm = analysis["class_metrics"]
        self.assertIsNotNone(cm["crack"]["width_p95_mm"])
        self.assertIsNotNone(cm["fissures"]["width_p95_mm"])
        self.assertGreater(cm["spalling"]["equivalent_diameter_mm"], 0)
        self.assertGreater(cm["rebar"]["centerline_length_mm"], 0)
        self.assertIsNotNone(analysis["physical"]["width_p95_mm"])
        self.assertIsNotNone(analysis["physical"]["spalling_area_mm2"])
        self.assertIsNotNone(analysis["physical"]["rebar_length_mm"])

        telemetry = normalize_telemetry({
            "stages": {
                "detection": {"status": "succeeded", "result": {"detection_count": 4, "class_counts": {"crack": 1, "concrete spalling": 1, "rebar": 1, "fissures": 1}}},
                "multi_defect_mapping": {"status": "succeeded", "result": {
                    "units": "mm",
                    "calibration": cal,
                    "physical": analysis["physical"],
                    "class_metrics": cm,
                    "composite_summary": analysis["composite_summary"],
                }},
            }
        })
        self.assertAlmostEqual(telemetry["physical"]["width_p95_mm"], analysis["physical"]["width_p95_mm"], places=2)
        res = assessment("How do widths compare to ACI thresholds?", telemetry, "Concrete Beam", True, None, None, "image/png")
        self.assertEqual(res["aci_level"], "CRITICAL")
        self.assertIn("Measured (calibrated):", res["direct_answer"])
        self.assertIn("EXCEEDS", res["direct_answer"])


if __name__ == "__main__":
    unittest.main()
