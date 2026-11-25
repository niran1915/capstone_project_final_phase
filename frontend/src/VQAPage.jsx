import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, ArrowRight, Bot, Check, CheckCircle2, ChevronDown, Clock3,
  Cpu, Download, FileImage, Key, Layers, LoaderCircle, MessageSquareQuote, RefreshCw,
  Send, ShieldAlert, ShieldCheck, Sparkles, User, Wrench, X,
} from 'lucide-react';
import { api, formatDate, publicUrl } from './api.js';
import { EmptyState, ErrorNotice, ModuleHeader, UploadBox } from './shared.jsx';

const STRUCTURE_TYPES = [
  'Concrete Element', 'Concrete Beam', 'Concrete Column', 'Slab / Bridge Deck',
  'Retaining Wall', 'Building Facade', 'Foundation / Footing', 'Masonry Wall',
];

const QUICK_QUESTIONS = [
  'How do the measured crack widths compare to ACI 224R thresholds?',
  'Summarize the combined detection, segmentation, and mapping findings.',
  'What repair or injection action is appropriate under ACI 224.1R / ACI 562?',
  'Where are the highest-priority mapped hotspot cells to inspect first?',
];

function getTelemetryValues(result = {}, module = '') {
  const telemetry = result.telemetry && typeof result.telemetry === 'object'
    ? result.telemetry
    : result;
  const metrics = result.metrics && typeof result.metrics === 'object'
    ? result.metrics
    : {};
  const classCounts = telemetry.class_counts || result.class_counts || {};
  const damageClasses = telemetry.damage_classes || telemetry.class_metrics
    || result.damage_classes || result.class_metrics
    || result.stages?.multi_defect_mapping?.result?.class_metrics || {};
  const crackAreaRatio = metrics.crack_area_ratio ?? metrics.area_ratio;

  return {
    module: telemetry.source_module || result.source_type || module,
    combinedModules: telemetry.combined_modules || result.combined_modules || [],
    detections: telemetry.detection_count ?? telemetry.total_detections ?? result.detection_count ?? 0,
    classCounts,
    damageClasses,
    crackPixels: telemetry.crack_pixels ?? metrics.crack_pixels ?? 0,
    width: telemetry.mapping_width_p95
      ?? telemetry.estimated_width_px
      ?? telemetry.estimated_median_width_px
      ?? metrics.width_p95
      ?? 0,
    hotspots: telemetry.hotspot_count ?? result.hotspot_count ?? 0,
    priority: telemetry.max_priority ?? result.max_priority ?? 0,
    junctions: telemetry.mapping_junctions
      ?? telemetry.junction_count
      ?? metrics.junction_clusters
      ?? metrics.junction_count
      ?? 0,
    coverage: telemetry.coverage_percent
      ?? telemetry.mask_coverage_percent
      ?? result.mask_coverage_percent
      ?? (crackAreaRatio != null ? crackAreaRatio * 100 : 0),
    sampledFrames: telemetry.sampled_frame_count ?? result.processed_frame_count ?? 0,
    framesWithCracks: telemetry.frames_with_cracks ?? result.frames_with_cracks ?? 0,
  };
}

function mergeTelemetryValues(current, result, module = '') {
  const telemetry = result.telemetry && typeof result.telemetry === 'object'
    ? result.telemetry
    : result;
  const metrics = result.metrics && typeof result.metrics === 'object'
    ? result.metrics
    : {};
  const next = getTelemetryValues(result, module);
  const merged = { ...(current || {}) };
  const updateWhenPresent = (field, sources) => {
    if (sources.some(([object, key]) => Object.prototype.hasOwnProperty.call(object, key))) {
      merged[field] = next[field];
    }
  };

  updateWhenPresent('module', [[telemetry, 'source_module'], [result, 'source_type']]);
  updateWhenPresent('detections', [[telemetry, 'detection_count'], [telemetry, 'total_detections'], [result, 'detection_count']]);
  updateWhenPresent('classCounts', [[telemetry, 'class_counts'], [result, 'class_counts']]);
  updateWhenPresent('damageClasses', [
    [telemetry, 'damage_classes'], [telemetry, 'class_metrics'],
    [result, 'damage_classes'], [result, 'class_metrics'],
  ]);
  updateWhenPresent('crackPixels', [[telemetry, 'crack_pixels'], [metrics, 'crack_pixels']]);
  updateWhenPresent('width', [
    [telemetry, 'mapping_width_p95'], [telemetry, 'estimated_width_px'],
    [telemetry, 'estimated_median_width_px'], [metrics, 'width_p95'],
  ]);
  updateWhenPresent('hotspots', [[telemetry, 'hotspot_count'], [result, 'hotspot_count']]);
  updateWhenPresent('priority', [[telemetry, 'max_priority'], [result, 'max_priority']]);
  updateWhenPresent('junctions', [
    [telemetry, 'mapping_junctions'], [telemetry, 'junction_count'], [metrics, 'junction_count'],
  ]);
  updateWhenPresent('coverage', [
    [telemetry, 'coverage_percent'], [telemetry, 'mask_coverage_percent'],
    [result, 'mask_coverage_percent'], [metrics, 'crack_area_ratio'], [metrics, 'area_ratio'],
  ]);
  updateWhenPresent('sampledFrames', [[telemetry, 'sampled_frame_count'], [result, 'processed_frame_count']]);
  updateWhenPresent('framesWithCracks', [[telemetry, 'frames_with_cracks'], [result, 'frames_with_cracks']]);

  if (!current) return { ...next, ...merged, module: next.module || module };
  return merged;
}

function normalizeDamageClasses(classes = {}) {
  if (!classes || typeof classes !== 'object') return {};
  return Object.fromEntries(Object.entries(classes).sort(([a], [b]) => a.localeCompare(b)).map(([name, metric]) => {
    const value = metric && typeof metric === 'object' ? metric : {};
    return [name, {
      pixels: Number(value.pixel_area ?? value.pixels ?? 0),
      coverage: Number(value.coverage_percent ?? 0),
      components: Number(value.component_count ?? 0),
      perimeter: Number(value.perimeter_px ?? 0),
      centerline: Number(value.centerline_length_px ?? 0),
      diameter: Number(value.equivalent_diameter_px ?? 0),
    }];
  }));
}

function telemetryMatchesInspection(inspectionTelemetry, answerTelemetry) {
  if (!inspectionTelemetry || !answerTelemetry) return false;
  const numericFields = ['detections', 'crackPixels', 'width', 'hotspots', 'priority', 'junctions', 'coverage'];
  const numericMatch = numericFields.every(field => {
    const inspectionValue = Number(inspectionTelemetry[field] || 0);
    const answerValue = Number(answerTelemetry[field] || 0);
    return Number.isFinite(inspectionValue)
      && Number.isFinite(answerValue)
      && Math.abs(inspectionValue - answerValue) <= 0.01;
  });
  const inspectionClasses = JSON.stringify(
    Object.entries(inspectionTelemetry.classCounts || {}).sort(([a], [b]) => a.localeCompare(b)),
  );
  const answerClasses = JSON.stringify(
    Object.entries(answerTelemetry.classCounts || {}).sort(([a], [b]) => a.localeCompare(b)),
  );
  const inspectionDamage = JSON.stringify(normalizeDamageClasses(inspectionTelemetry.damageClasses));
  const answerDamage = JSON.stringify(normalizeDamageClasses(answerTelemetry.damageClasses));
  return numericMatch && inspectionClasses === answerClasses && inspectionDamage === answerDamage;
}

export default function VQAPage({ projectId, projects, notify, onInspectionComplete, pipeline, onStepComplete }) {
  const [sourceMode, setSourceMode] = useState('existing'); // 'existing' | 'upload'
  const [inspections, setInspections] = useState([]);
  const [selectedInspectionId, setSelectedInspectionId] = useState(pipeline?.mappingInspectionId || '');
  const [structureType, setStructureType] = useState('Concrete Element');
  const [uploadFile, setUploadFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [config, setConfig] = useState(null);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('crackvision-vqa-key') || '');
  const [offlineMode, setOfflineMode] = useState(true);
  const [showKeyDrawer, setShowKeyDrawer] = useState(false);

  useEffect(() => {
    if (pipeline?.mappingInspectionId) {
      setSourceMode('existing');
      setSelectedInspectionId(pipeline.mappingInspectionId);
    }
  }, [pipeline?.mappingInspectionId]);

  // Chat state
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Hello! I am **CrackVision AI Inspector**. Select an inspection or upload a photo, then ask about detector findings, crack-mask geometry, or evidence-based follow-up.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [exportingPdf, setExportingPdf] = useState(false);
  const [activeTelemetry, setActiveTelemetry] = useState(null);
  const [rawCombinedTelemetry, setRawCombinedTelemetry] = useState(null);

  const messagesEndRef = useRef(null);
  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollToBottom(); }, [messages, loading]);

  // Load VQA Config & Inspections list
  const loadInitialData = useCallback(async () => {
    try {
      const [cfg, inspList] = await Promise.all([
        api.get('/api/vqa/config'),
        api.get('/api/inspections?limit=30'),
      ]);
      setConfig(cfg);
      const successful = (inspList || []).filter(i => i.status === 'succeeded' && i.module !== 'video');
      setInspections(successful);
      if (successful.length > 0 && !selectedInspectionId) {
        setSelectedInspectionId(successful[0].id);
      }
    } catch (err) {
      // Non-critical if offline
    }
  }, [selectedInspectionId]);

  useEffect(() => { loadInitialData(); }, [loadInitialData]);

  // Handle local image preview
  useEffect(() => {
    if (!uploadFile) {
      if (sourceMode === 'upload') setPreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(uploadFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [uploadFile, sourceMode]);

  // Current inspection object
  const currentInspection = useMemo(() => {
    return inspections.find(i => i.id === selectedInspectionId) || null;
  }, [inspections, selectedInspectionId]);

  // Resolve preview image for existing inspection
  const existingImageUrl = useMemo(() => {
    if (!currentInspection) return '';
    const res = currentInspection.result || {};
    return publicUrl(
      res.vqa_frame_url
        || res.stages?.segmentation_and_mapping?.result?.peak_frame_url
        || res.stages?.segmentation?.result?.overlay_url
        || res.stages?.mapping?.result?.overlay_url
        || res.overlay_url
        || res.annotated_image_url
        || res.mask_url
        || `/storage/${currentInspection.id}/original.jpg`
    );
  }, [currentInspection]);

  // Update telemetry badge when selected inspection changes (fetch combined detection+segmentation+mapping telemetry)
  useEffect(() => {
    let cancelled = false;
    if (sourceMode !== 'existing' || !currentInspection) return undefined;
    setRawCombinedTelemetry(currentInspection.result || {});
    setActiveTelemetry(getTelemetryValues(currentInspection.result || {}, currentInspection.module));
    api.get(`/api/vqa/combined-telemetry/${currentInspection.id}`)
      .then(data => {
        if (!cancelled && data?.telemetry) {
          setRawCombinedTelemetry(data.telemetry);
          setActiveTelemetry(getTelemetryValues(data.telemetry, currentInspection.module));
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [sourceMode, currentInspection]);

  const handleSend = async (questionToSend) => {
    const q = (questionToSend || inputQuery).trim();
    if (!q || loading) return;

    if (sourceMode === 'existing' && !selectedInspectionId) {
      notify('Please select an inspection first.', 'error');
      return;
    }
    if (sourceMode === 'upload' && !uploadFile) {
      notify('Please select an image file to inspect.', 'error');
      return;
    }

    const userMsg = {
      id: String(Date.now()),
      role: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages(prev => [...prev, userMsg]);
    setInputQuery('');
    setLoading(true);

    try {
      let res;
      if (sourceMode === 'existing') {
        res = await api.postJson('/api/vqa/ask', {
          inspection_id: selectedInspectionId,
          question: q,
          structure_type: structureType,
          offline_mode: offlineMode,
          api_key: !offlineMode && apiKey.trim() ? apiKey.trim() : undefined,
        });
      } else {
        const form = new FormData();
        form.append('file', uploadFile);
        form.append('question', q);
        form.append('structure_type', structureType);
        form.append('offline_mode', String(offlineMode));
        if (projectId) form.append('project_id', String(projectId));
        if (!offlineMode && apiKey.trim()) form.append('api_key', apiKey.trim());
        res = await api.postForm('/api/vqa/inspect-and-ask', form);
        if (res.queued && res.inspection?.id) {
          let inspection = res.inspection;
          while (inspection.status === 'queued' || inspection.status === 'processing') {
            await new Promise(resolve => window.setTimeout(resolve, 2000));
            inspection = await api.get(`/api/inspections/${inspection.id}`);
          }
          if (inspection.status !== 'succeeded') {
            throw new Error(inspection.error_message || 'Unified video inspection failed.');
          }
          res = { ...inspection.result, inspection };
        }
        if (onInspectionComplete) onInspectionComplete();
        await loadInitialData();
      }

      const answerTelemetry = res.telemetry
        ? mergeTelemetryValues(activeTelemetry, res.telemetry, currentInspection?.module || 'vqa')
        : activeTelemetry;
      const telemetryMismatch = sourceMode === 'existing' && (
        !telemetryMatchesInspection(
          activeTelemetry,
          res.telemetry ? getTelemetryValues(res.telemetry, currentInspection?.module) : null,
        )
        || (res.structure_type && res.structure_type.toLowerCase() !== structureType.toLowerCase())
      );
      if (res.telemetry && !telemetryMismatch) {
        setRawCombinedTelemetry(res.telemetry);
        setActiveTelemetry(answerTelemetry);
      }

      const pipelineStages = res.stages || {};
      const stageResult = name => pipelineStages[name]?.result || {};
      const artifactCandidates = [
        ['Annotated detection video', res.annotated_video_url || stageResult('detection').annotated_video_url],
        ['YOLO detection CSV', stageResult('detection').detections_csv_url],
        ['Damage-mapped video', res.mapping_video_url || stageResult('segmentation_and_mapping').annotated_video_url],
        ['Sampled mapping frames CSV', stageResult('segmentation_and_mapping').frames_csv_url],
        ['Mapping summary JSON', stageResult('segmentation_and_mapping').summary_json_url],
        ['Representative analyzed frame', res.vqa_frame_url || stageResult('segmentation_and_mapping').peak_frame_url],
        ['Annotated detection image', stageResult('detection').annotated_image_url],
        ['Crack segmentation overlay', stageResult('segmentation').overlay_url],
        ['Crack segmentation mask', stageResult('segmentation').mask_url],
        ['Damage-mapping report (PDF)', stageResult('mapping').report_pdf_url],
      ];
      const artifacts = artifactCandidates
        .filter(([, url]) => Boolean(url))
        .map(([label, url]) => ({ label, url }));
      const botMsg = {
        id: String(Date.now() + 1),
        role: 'assistant',
        questionAsked: q,
        text: telemetryMismatch
          ? 'The generated assessment was withheld because the VQA service returned telemetry that does not match the selected inspection. Restart the backend so it loads the latest telemetry synchronization, then ask again.'
          : (res.direct_answer || res.answer),
        aciLevel: telemetryMismatch ? undefined : (res.aci_level || res.report?.aci_level),
        labelRisks: telemetryMismatch ? undefined : (res.label_risks || res.report?.label_risks || []),
        actionTimeline: telemetryMismatch ? undefined : (res.action_timeline || res.report?.action_timeline),
        aciBasis: telemetryMismatch ? undefined : (res.aci_basis || res.report?.aci_basis),
        aciBadgeColor: telemetryMismatch ? undefined : res.aci_badge_color,
        assessmentType: res.assessment_type,
        severityColor: telemetryMismatch ? undefined : (res.aci_badge_color || res.severity_color),
        urgency: telemetryMismatch ? undefined : (res.action_timeline || res.urgency),
        missingDirectAnswer: !telemetryMismatch && !res.direct_answer,
        engine: res.engine,
        repairSummary: telemetryMismatch ? undefined : res.repair_summary,
        notice: res.notice,
        telemetry: answerTelemetry,
        rawTelemetry: res.telemetry || rawCombinedTelemetry,
        inspectionIdUsed: res.inspection_id || selectedInspectionId || undefined,
        report: telemetryMismatch ? undefined : (res.report || res.inspection?.result?.report),
        artifacts: telemetryMismatch ? [] : artifacts,
        telemetryMismatch,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages(prev => [...prev, botMsg]);
      if (onStepComplete) onStepComplete('vqa');
    } catch (err) {
      notify(err.message || 'VQA assessment failed.', 'error');
      setMessages(prev => [
        ...prev,
        {
          id: String(Date.now() + 1),
          role: 'assistant',
          text: `⚠️ **Inspection Error**: ${err.message || 'Could not complete visual assessment.'}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const saveApiKey = (val) => {
    setApiKey(val);
    localStorage.setItem('crackvision-vqa-key', val);
    notify('API key saved locally.', 'success');
  };

  const handleExportPdf = async (singleMsg = null) => {
    const assistantMessages = singleMsg
      ? [singleMsg]
      : messages.filter(m => m.role === 'assistant' && m.id !== 'welcome' && !m.isError && !m.telemetryMismatch);
    if (assistantMessages.length === 0) {
      notify('Ask at least one question before exporting a PDF report.', 'error');
      return;
    }
    const latest = assistantMessages[assistantMessages.length - 1];
    setExportingPdf(true);
    try {
      const response = await fetch(publicUrl('/api/vqa/report/pdf'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inspection_id: latest.inspectionIdUsed || (sourceMode === 'existing' ? selectedInspectionId : undefined) || undefined,
          structure_type: structureType,
          aci_level: latest.aciLevel || 'MODERATE',
          action_timeline: latest.actionTimeline || latest.urgency || '',
          aci_basis: latest.aciBasis || '',
          telemetry: latest.rawTelemetry || rawCombinedTelemetry || {},
          qa_items: assistantMessages.map(m => ({
            question: m.questionAsked || '',
            answer: m.text || '',
          })),
        }),
      });
      if (!response.ok) {
        throw new Error(`PDF export failed (${response.status})`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `crackvision-vqa-${(latest.aciLevel || 'report').toLowerCase()}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      notify('VQA PDF report downloaded.', 'success');
    } catch (err) {
      notify(err.message || 'Could not export PDF report.', 'error');
    } finally {
      setExportingPdf(false);
    }
  };

  return (
    <div className="stack-lg">
      <ModuleHeader
        eyebrow="MODULE 05 · VISUAL QUESTION ANSWERING"
        title="AI Structural Inspector (VQA)"
        text="Image uploads run detection, segmentation, and per-label 20×20 image-grid mapping before VQA. The map points to regions in the photo; field suggestions cite FHWA/ACI inspection protocols and do not treat pixels as physical severity or site coordinates."
        icon={MessageSquareQuote}
        tag="VLM + Civil Engineering Engine"
      />

      <section className="panel" aria-labelledby="vqa-evidence-title" style={{ padding: 16 }}>
        <div className="panel-title-row">
          <div>
            <h3 id="vqa-evidence-title">Label-based map suggestions</h3>
            <p>VQA sorts image findings by pixel coverage but does not rate structural severity from pixels. It presents class-specific field measurements and escalation criteria from inspection guidance; it does not prescribe repairs.</p>
          </div>
          <span className="step-badge violet-step">FIELD GUIDANCE</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10, fontSize: 10, lineHeight: 1.55 }}>
          <div><b>Crack</b><br />At the mapped image cells, verify the actual member, mark and photograph crack ends, record length/orientation, and measure the widest opening with a crack gauge. Check moisture and compare repeat inspections at the same reference points. <a href="https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/049.cfm" target="_blank" rel="noreferrer">FHWA concrete cracking field protocol</a>.</div>
          <div><b>Fissures / micro-cracks</b><br />This model label means a fine crack-like finding, not a separate severity grade. Check mapped cells in person, use a scale-referenced close-up, note moisture/rust staining, and compare repeat observations; have an engineer determine cause before repair. <a href="https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/049.cfm" target="_blank" rel="noreferrer">FHWA concrete cracking field protocol</a> · <a href="https://www.concrete.org/frequentlyaskedquestions.aspx?faqid=809" target="_blank" rel="noreferrer">ACI crack guidance</a>.</div>
          <div><b>Spalling</b><br />At mapped cells, have a qualified inspector verify boundaries and dimensions and sound suspected concrete for hollow/delaminated areas. If material is loose overhead, keep people clear and request prompt site inspection. <a href="https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/048.cfm" target="_blank" rel="noreferrer">FHWA superstructure/substructure spall protocol</a>.</div>
          <div><b>Exposed / corroded rebar</b><br />At mapped cells, have a qualified inspector verify exposed steel, record its location and length, and assess section loss against reliable original dimensions. Photos cannot establish remaining bar area or capacity. <a href="https://www.fhwa.dot.gov/publications/research/infrastructure/structures/ltbp/16007/048.cfm" target="_blank" rel="noreferrer">FHWA exposed-steel field protocol</a>.</div>
        </div>
        <p style={{ margin: '12px 0 0', fontSize: 9, color: '#657487' }}><b>Severity rule:</b> escalate visible instability, falling/loose concrete, sudden change, significant displacement, or suspected reinforcement damage for immediate site-specific professional review and access controls. Otherwise document and arrange a qualified inspection. Never infer severity from pixel area, detector count, or the exploratory mapping index. The referenced model label set is crack, spalling, merged rebar, and fissures; concrete crushing is not a trained output.</p>
      </section>

      <div className="workbench-grid">
        {/* Left Column: Input and Context Configuration */}
        <div className="control-panel panel">
          <div className="panel-title-row">
            <div>
              <h3>Inspection Source</h3>
              <p>Choose a saved result or upload an image for detection, mapping, and evidence-based VQA.</p>
            </div>
            <span className="step-badge violet-step">STEP 1</span>
          </div>

          <div className="result-tabs mapping-mode-tabs" style={{ width: '100%', marginBottom: 15 }}>
            <button
              type="button"
              className={sourceMode === 'existing' ? 'active' : ''}
              onClick={() => setSourceMode('existing')}
            >
              Select Past Inspection
            </button>
            <button
              type="button"
              className={sourceMode === 'upload' ? 'active' : ''}
              onClick={() => {
                setActiveTelemetry(null);
                setSourceMode('upload');
              }}
            >
              Image VQA Screening
            </button>
          </div>

          {sourceMode === 'existing' ? (
            <div className="setting-field">
              <label style={{ fontSize: 10, fontWeight: 700, color: '#526274', display: 'block', marginBottom: 8 }}>
                Select Inspection Record
              </label>
              {inspections.length === 0 ? (
                <div className="empty-state" style={{ padding: '20px 10px' }}>
                  <FileImage size={24} style={{ opacity: 0.5 }} />
                  <b>No inspections found</b>
                  <p>Run a Detection, Segmentation, or Mapping job first.</p>
                </div>
              ) : (
                <select
                  className="select-control"
                  value={selectedInspectionId}
                  onChange={e => setSelectedInspectionId(e.target.value)}
                  style={{ marginBottom: 14 }}
                >
                  {inspections.map(i => (
                    <option key={i.id} value={i.id}>
                      {i.source_filename} ({i.module}) — {formatDate(i.created_at)}
                    </option>
                  ))}
                </select>
              )}

              {existingImageUrl && (
                <div className="result-image-frame" style={{ minHeight: 180, maxHeight: 220, marginBottom: 12 }}>
                  <img src={existingImageUrl} alt="Selected inspection preview" />
                </div>
              )}
            </div>
          ) : (
            <div style={{ marginBottom: 16 }}>
              <UploadBox
                accept="image/jpeg,image/png,image/webp,image/tiff,.jpg,.jpeg,.png,.webp,.tif,.tiff,.gif,.bmp"
                file={uploadFile}
                setFile={setUploadFile}
                label="Drop a concrete inspection image"
                help="JPEG, PNG, WEBP, TIFF, GIF, or BMP · maximum 15 MB"
              />
              {previewUrl && (
                <div className="result-image-frame" style={{ minHeight: 180, maxHeight: 220, marginTop: 10, marginBottom: 12 }}>
                  {uploadFile?.type.startsWith('video/') ? (
                    <video src={previewUrl} controls style={{ width: '100%', maxHeight: 220, objectFit: 'contain' }} />
                  ) : (
                    <img src={previewUrl} alt="New upload preview" />
                  )}
                </div>
              )}
            </div>
          )}

          <div className="settings-divider" />

          <div className="panel-title-row">
            <div>
              <h3>Structural Context</h3>
              <p>Civil engineering parameters to calibrate recommendations.</p>
            </div>
            <span className="step-badge green-step">STEP 2</span>
          </div>

          <div className="setting-field">
            <label style={{ fontSize: 10, fontWeight: 700, color: '#526274', display: 'block', marginBottom: 6 }}>
              Element Classification
            </label>
            <select
              className="select-control"
              value={structureType}
              onChange={e => setStructureType(e.target.value)}
            >
              {STRUCTURE_TYPES.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <small style={{ color: '#9aa5b0', fontSize: 9, marginTop: 4, display: 'block' }}>
              Informs whether shear, flexural, or shrinkage behavior dominates.
            </small>
          </div>

          {/* Active Model Telemetry Pill Box */}
          {activeTelemetry && (
            <div style={{ background: '#f5f7fa', border: '1px solid #e7ebf0', borderRadius: 10, padding: 12, marginTop: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, fontSize: 10, fontWeight: 800, color: '#44566c' }}>
                <Cpu size={14} style={{ color: '#2b8a6c' }} />
                <span>Synchronized Model Telemetry</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px 12px', fontSize: 9, color: '#687788' }}>
                <div>
                  YOLO Detections: <b style={{ color: '#2a3b4c' }}>{activeTelemetry.detections}</b>
                  {Object.keys(activeTelemetry.classCounts || {}).length > 0 && (
                    <span> ({Object.entries(activeTelemetry.classCounts).map(([name, count]) => `${name}: ${count}`).join(', ')})</span>
                  )}
                </div>
                <div>{activeTelemetry.module === 'video' ? 'Mean Crack Coverage / Frame' : activeTelemetry.module === 'mapping' ? 'Crack-map Coverage' : 'Full-image Crack Segmentation Coverage'}: <b style={{ color: '#2a3b4c' }}>{Number(activeTelemetry.coverage).toFixed(2)}%</b></div>
                <div>{activeTelemetry.module === 'video' ? 'Mean Crack Pixels / Frame' : activeTelemetry.module === 'mapping' ? 'Crack-map Pixels' : 'Full-image Crack Segmentation Pixels'}: <b style={{ color: '#2a3b4c' }}>{Number(activeTelemetry.crackPixels).toLocaleString()}</b></div>
                <div>{activeTelemetry.module === 'segmentation' ? 'Estimated Crack Width (source image)' : 'Crack Topology Map Width P95'}: <b style={{ color: '#2a3b4c' }}>{Number(activeTelemetry.width).toFixed(2)} px{activeTelemetry.module === 'segmentation' ? '' : ' @ 256×256'}</b></div>
                <div>Crack-map Hotspots: <b style={{ color: '#2a3b4c' }}>{activeTelemetry.hotspots}</b></div>
                <div>Risk Priority Index: <b style={{ color: '#2a3b4c' }}>{Number(activeTelemetry.priority).toFixed(2)}</b></div>
                {activeTelemetry.module === 'video' && (
                  <div>Crack Frames: <b style={{ color: '#2a3b4c' }}>{activeTelemetry.framesWithCracks}/{activeTelemetry.sampledFrames}</b></div>
                )}
              </div>
            </div>
          )}

          <div className="settings-divider" />

          {/* Engine & Key Configuration Drawer */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 10, color: '#677889' }}>
              <ShieldCheck size={15} style={{ color: '#249f76' }} />
              <span>Engine: <b>{!offlineMode && apiKey ? 'Cloud VLM' : 'Offline Evidence Engine'}</b></span>
            </div>
            <button
              type="button"
              className="text-button"
              style={{ fontSize: 9 }}
              onClick={() => setShowKeyDrawer(!showKeyDrawer)}
            >
              <Key size={12} /> {showKeyDrawer ? 'Close API Key' : 'API Key Setup'}
            </button>
          </div>

          <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 10, fontSize: 10, color: '#536476' }}>
            <input type="checkbox" checked={offlineMode} onChange={event => setOfflineMode(event.target.checked)} />
            Use offline VQA (saved model evidence only; no network inference)
          </label>

          {showKeyDrawer && (
            <div style={{ marginTop: 10, padding: 11, background: '#fafbfc', border: '1px solid #e4eaf0', borderRadius: 8 }}>
              <label style={{ fontSize: 9, fontWeight: 700, color: '#536476', display: 'block', marginBottom: 5 }}>
                Google Gemini or OpenAI API Key (Optional)
              </label>
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  type="password"
                  placeholder="Paste Gemini/OpenAI key..."
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  style={{ flex: 1, padding: '6px 8px', fontSize: 9, border: '1px solid #dbe2e8', borderRadius: 6 }}
                />
                <button
                  type="button"
                  className="button button-primary button-small"
                  onClick={() => saveApiKey(apiKey)}
                >
                  Save
                </button>
              </div>
              <small style={{ color: '#9ba6b1', fontSize: 8, display: 'block', marginTop: 5 }}>
                Leave blank for source-grounded offline triage from saved model evidence. Cloud visual screening requires a key and offline mode disabled.
              </small>
            </div>
          )}
        </div>

        {/* Right Column: VQA Conversational Workspace */}
        <div className="result-panel panel" style={{ display: 'flex', flexDirection: 'column', height: 680, padding: '18px 20px' }}>
          <div className="panel-title-row" style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
              <div style={{ width: 34, height: 34, borderRadius: 10, background: '#f0ecfb', color: '#6a55b8', display: 'grid', placeItems: 'center' }}>
                <Bot size={18} />
              </div>
              <div>
                <h3 style={{ margin: 0 }}>VQA Inspection Dialogue</h3>
                <p style={{ margin: 0 }}>Interactive structural diagnosis and repair consultation.</p>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button
                type="button"
                className="text-button"
                style={{ fontSize: 10, color: '#315b9b', fontWeight: 700 }}
                onClick={() => handleExportPdf(null)}
                disabled={exportingPdf || messages.filter(m => m.role === 'assistant' && m.id !== 'welcome' && !m.isError).length === 0}
                title="Download Q&A and ACI severity report as PDF"
              >
                {exportingPdf ? <LoaderCircle size={12} className="spin" /> : <Download size={12} />} Export PDF
              </button>
              <button
                type="button"
                className="text-button"
                style={{ fontSize: 10, color: '#7a8999' }}
                onClick={() => setMessages([messages[0]])}
                title="Clear conversation"
              >
                <RefreshCw size={12} /> Clear Chat
              </button>
            </div>
          </div>

          {/* Messages Container */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            paddingRight: 6,
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
            marginBottom: 12,
          }}>
            {messages.map(msg => (
              <div
                key={msg.id}
                style={{
                  display: 'flex',
                  gap: 10,
                  alignItems: 'flex-start',
                  flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                }}
              >
                {/* Avatar */}
                <div style={{
                  width: 30, height: 30, borderRadius: '50%',
                  display: 'grid', placeItems: 'center', flexShrink: 0,
                  background: msg.role === 'user' ? '#182b3d' : '#eaf8f2',
                  color: msg.role === 'user' ? '#d9e9f6' : '#1a8260',
                }}>
                  {msg.role === 'user' ? <User size={15} /> : <Bot size={16} />}
                </div>

                {/* Message Bubble */}
                <div style={{
                  maxWidth: '82%',
                  background: msg.role === 'user' ? '#192e42' : '#ffffff',
                  color: msg.role === 'user' ? '#f0f6fc' : '#223447',
                  border: msg.role === 'user' ? '1px solid #233e57' : '1px solid #e7ebf0',
                  borderRadius: 12,
                  padding: '12px 14px',
                  boxShadow: msg.role === 'user' ? 'none' : '0 2px 7px rgba(18, 38, 57, 0.02)',
                  fontSize: 11,
                  lineHeight: 1.6,
                }}>
                  {/* ACI Risk Banner (CRITICAL / MODERATE / LOW + per-label chips) */}
                  {msg.missingDirectAnswer && (
                    <div role="alert" style={{ marginBottom: 10, padding: 9, background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 7, color: '#9a3412', fontSize: 9.5 }}>
                      <b>Backend restart required:</b> Your running backend server is still on the older code. Restart the backend terminal, then ask again.
                    </div>
                  )}
                  {msg.aciLevel && (
                    <div style={{
                      background: msg.severityColor ? `${msg.severityColor}15` : '#fef2f2',
                      borderLeft: `3px solid ${msg.severityColor || '#dc2626'}`,
                      padding: '8px 10px',
                      borderRadius: 6,
                      marginBottom: 10,
                      fontSize: 10,
                      color: msg.severityColor || '#dc2626',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6, fontWeight: 800 }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <ShieldAlert size={14} />
                          ACI RISK: {msg.aciLevel}
                        </span>
                        {msg.actionTimeline && (
                          <span style={{ fontWeight: 700, fontSize: 9.5 }}>
                            {msg.actionTimeline}
                          </span>
                        )}
                      </div>
                      {msg.labelRisks?.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                          {msg.labelRisks.map(item => (
                            <span
                              key={item.label}
                              style={{
                                fontSize: 8.5,
                                fontWeight: 700,
                                padding: '2px 7px',
                                borderRadius: 999,
                                background: '#ffffff',
                                border: `1px solid ${item.color || '#cbd5e1'}`,
                                color: item.color || '#334155',
                              }}
                            >
                              {item.title}: {item.risk}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Structured & formatted message body */}
                  {msg.telemetryMismatch && (
                    <div role="alert" style={{ marginBottom: 10, padding: 9, background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 7, color: '#9a3412', fontSize: 9 }}>
                      <b>Assessment blocked to prevent a false report.</b>
                      <div style={{ marginTop: 4 }}>{msg.text}</div>
                    </div>
                  )}
                  {!msg.telemetryMismatch && (
                    <StructuredContent text={msg.text} isUser={msg.role === 'user'} repairSummary={msg.repairSummary} />
                  )}

                  {(msg.telemetry || msg.report) && !msg.telemetryMismatch && (
                    <details style={{ marginTop: 10, padding: '7px 10px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 7, fontSize: 9 }}>
                      <summary style={{ cursor: 'pointer', fontWeight: 700, color: '#475569' }}>
                        Combined module numbers &amp; technical details
                      </summary>
                      {msg.telemetry && (
                        <div style={{ marginTop: 8, padding: 8, background: '#f0f7f4', border: '1px solid #d6ebe1', borderRadius: 6 }}>
                          <b style={{ display: 'block', marginBottom: 4, color: '#23684f' }}>Model measurements used for this answer</b>
                          {msg.telemetry.source_module === 'vqa_visual_screening' ? (
                            <div>Detector / segmentation measurements: not available for this image screen.</div>
                          ) : <>
                          <div>YOLO detections: {msg.telemetry.detections}{Object.keys(msg.telemetry.classCounts || {}).length > 0 ? ` (${Object.entries(msg.telemetry.classCounts).map(([name, count]) => `${name}: ${count}`).join(', ')})` : ''}</div>
                          <div>{msg.telemetry.module === 'mapping' ? 'Crack-map coverage' : 'Full-image crack segmentation coverage'}: {Number(msg.telemetry.coverage).toFixed(2)}%</div>
                          <div>{msg.telemetry.module === 'mapping' ? 'Crack-map pixels' : 'Full-image crack segmentation pixels'}: {Number(msg.telemetry.crackPixels).toLocaleString()}</div>
                          <div>{msg.telemetry.module === 'segmentation' ? 'Segmentation estimated median crack width' : 'Crack topology-map width P95 (256×256)'}: {Number(msg.telemetry.width).toFixed(2)} px</div>
                          <div>Crack-map hotspots: {msg.telemetry.hotspots}</div>
                          <div>Risk priority index: {Number(msg.telemetry.priority).toFixed(2)}</div>
                          <div>Crack branching junctions: {msg.telemetry.junctions}</div>
                          </>}
                        </div>
                      )}
                      {msg.report && (
                        <div style={{ marginTop: 8, padding: 8, background: '#f6f8fb', borderRadius: 6 }}>
                          <b style={{ display: 'block', marginBottom: 4 }}>Inspection pipeline summary</b>
                          {msg.report.pipeline_status && <div>Pipeline status: {msg.report.pipeline_status}</div>}
                          {msg.report.aci_basis && <div>ACI basis: {msg.report.aci_basis}</div>}
                        </div>
                      )}
                    </details>
                  )}
                  {msg.artifacts?.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 9 }}>
                      {msg.artifacts.map(item => (
                        <a key={`${item.label}-${item.url}`} href={publicUrl(item.url)} target="_blank" rel="noreferrer" style={{ fontSize: 9 }}>
                          {item.label}
                        </a>
                      ))}
                    </div>
                  )}

                  {/* Engine footer badge */}
                  {msg.engine && (
                    <div style={{
                      marginTop: 10,
                      paddingTop: 8,
                      borderTop: '1px solid #edf1f5',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      fontSize: 8,
                      color: '#9aa6b2',
                    }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <ShieldCheck size={11} style={{ color: '#2b9c79' }} /> {msg.engine}
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        {!msg.isError && !msg.telemetryMismatch && (
                          <button
                            type="button"
                            className="text-button"
                            style={{ fontSize: 8.5, color: '#315b9b', padding: 0 }}
                            onClick={() => handleExportPdf(msg)}
                            disabled={exportingPdf}
                          >
                            <Download size={10} /> PDF
                          </button>
                        )}
                        <span>{msg.timestamp}</span>
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <div style={{ width: 30, height: 30, borderRadius: '50%', background: '#eaf8f2', color: '#1a8260', display: 'grid', placeItems: 'center' }}>
                  <Bot size={16} />
                </div>
                <div style={{ background: '#fff', border: '1px solid #e7ebf0', borderRadius: 12, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 8, fontSize: 10, color: '#7a8b9a' }}>
                  <LoaderCircle size={15} className="spin" style={{ color: '#258f6d' }} />
                  <span>{offlineMode ? 'Preparing source-grounded assessment...' : 'Screening image and preparing evidence...'}</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Prompts Bar */}
          <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 8, marginBottom: 8 }}>
            {QUICK_QUESTIONS.map(q => (
              <button
                key={q}
                type="button"
                className="button button-secondary button-small"
                style={{ fontSize: 9, whiteSpace: 'nowrap', borderRadius: 8, padding: '6px 10px', background: '#f7f9fa' }}
                onClick={() => handleSend(q)}
                disabled={loading}
              >
                <Sparkles size={11} style={{ color: '#6a55b8' }} /> {q}
              </button>
            ))}
          </div>

          {/* Input Box */}
          <form
            onSubmit={e => { e.preventDefault(); handleSend(); }}
            style={{ display: 'flex', gap: 8, alignItems: 'center' }}
          >
            <input
              type="text"
              className="num-input"
              style={{ flex: 1, height: 42, fontSize: 11, fontFamily: 'inherit' }}
              placeholder="Ask about crack depth, failure mechanism, repair injection, etc..."
              value={inputQuery}
              onChange={e => setInputQuery(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              className="button button-primary"
              style={{ height: 42, padding: '0 18px', background: '#6a55b8', borderColor: '#6a55b8' }}
              disabled={!inputQuery.trim() || loading}
            >
              {loading ? <LoaderCircle size={16} className="spin" /> : <Send size={15} />}
              <span>Ask AI</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

const APPENDIX_HEADINGS = new Set([
  'Evidence-based field actions for supported labels',
  'Class-specific severity checks',
  'Class-wise mapping validation',
  'Evidence sources',
]);

function StructuredContent({ text, isUser, repairSummary }) {
  if (!text) return null;
  if (isUser) {
    return <div style={{ fontSize: 11, lineHeight: 1.5 }}>{text}</div>;
  }

  const lines = text.split('\n');
  const blocks = [];
  const appendixBlocks = [];
  let currentList = [];
  let inAppendix = false;

  const pushBlock = (node) => {
    if (inAppendix) appendixBlocks.push(node);
    else blocks.push(node);
  };

  const flushList = (key) => {
    if (currentList.length > 0) {
      pushBlock(
        <ul key={`list-${key}`} style={{ margin: '6px 0 10px', paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 5 }}>
          {currentList.map((item, idx) => (
            <li key={idx} style={{ color: '#384858', fontSize: 10.5, lineHeight: 1.5 }}>
              {parseInlineFormatting(item)}
            </li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    if (trimmed.startsWith('###')) {
      flushList(index);
      const title = trimmed.replace(/^###+\s*/, '');
      if (APPENDIX_HEADINGS.has(title)) {
        inAppendix = true;
      } else if (inAppendix) {
        inAppendix = false;
      }
      pushBlock(
        <div key={`h-${index}`} style={{ display: 'flex', alignItems: 'center', gap: 7, margin: '12px 0 6px', paddingBottom: 5, borderBottom: '1px solid #edf1f5' }}>
          <Sparkles size={13} style={{ color: '#6a55b8' }} />
          <h4 style={{ margin: 0, fontSize: 12, fontWeight: 800, color: '#1a2a3b', letterSpacing: '-0.2px' }}>{title}</h4>
        </div>
      );
    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      currentList.push(trimmed.replace(/^[-*]\s+/, ''));
    } else if (/^\d+\.\s+/.test(trimmed)) {
      flushList(index);
      const stepNum = trimmed.match(/^\d+/)[0];
      const stepContent = trimmed.replace(/^\d+\.\s+/, '');
      pushBlock(
        <div key={`step-${index}`} style={{ display: 'flex', gap: 9, alignItems: 'flex-start', margin: '5px 0', background: '#f7f9fb', padding: '7px 10px', borderRadius: 8, border: '1px solid #e9edf2' }}>
          <span style={{ minWidth: 19, height: 19, borderRadius: '50%', background: '#6a55b8', color: '#fff', fontSize: 9.5, fontWeight: 800, display: 'grid', placeItems: 'center', flexShrink: 0, marginTop: 1 }}>
            {stepNum}
          </span>
          <div style={{ flex: 1, fontSize: 10.5, color: '#334455', lineHeight: 1.5 }}>
            {parseInlineFormatting(stepContent)}
          </div>
        </div>
      );
    } else if (trimmed === '') {
      flushList(index);
    } else {
      flushList(index);
      if (trimmed.startsWith('**Prescribed Action:**') || trimmed.startsWith('**Primary Repair Technique:**')) {
        pushBlock(
          <div key={`act-${index}`} style={{ background: '#f0f9f5', border: '1px solid #d2ece0', borderRadius: 8, padding: '9px 12px', margin: '8px 0', display: 'flex', alignItems: 'flex-start', gap: 8 }}>
            <Wrench size={15} style={{ color: '#188661', flexShrink: 0, marginTop: 2 }} />
            <div style={{ fontSize: 10.5, color: '#203930', lineHeight: 1.5 }}>
              {parseInlineFormatting(trimmed)}
            </div>
          </div>
        );
      } else {
        pushBlock(
          <p key={`p-${index}`} style={{ margin: '4px 0', color: '#374758', fontSize: 10.5, lineHeight: 1.6 }}>
            {parseInlineFormatting(trimmed)}
          </p>
        );
      }
    }
  });

  flushList('end');

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {blocks}

      {appendixBlocks.length > 0 && (
        <details style={{ marginTop: 10, padding: '8px 10px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8 }}>
          <summary style={{ cursor: 'pointer', fontSize: 10, fontWeight: 700, color: '#475569' }}>
            ACI / FHWA Reference Protocols &amp; Benchmark Validation
          </summary>
          <div style={{ marginTop: 6 }}>
            {appendixBlocks}
          </div>
        </details>
      )}

      {repairSummary && (
        <div style={{ marginTop: 10, background: '#f4fbf8', border: '1px solid #d3eee1', borderRadius: 9, padding: '10px 12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 9.5, fontWeight: 800, color: '#19805e' }}>
              <Wrench size={13} /> REPAIR PROTOCOL HIGHLIGHT
            </span>
            <span style={{ fontSize: 8.5, fontWeight: 700, color: '#567167', background: '#e1f4ec', padding: '2px 6px', borderRadius: 4 }}>
              {repairSummary.code}
            </span>
          </div>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#1e322b' }}>
            {repairSummary.method}
          </div>
          <div style={{ fontSize: 9, color: '#59796c', marginTop: 2 }}>
            Urgency: <b>{repairSummary.urgency}</b>
          </div>
        </div>
      )}
    </div>
  );
}

function parseInlineFormatting(str) {
  if (!str) return '';
  const parts = str.split(/(\*\*.*?\*\*|`.*?`|\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g);
  return parts.map((part, i) => {
    const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
    if (link) {
      return <a key={i} href={link[2]} target="_blank" rel="noreferrer" style={{ color: '#315b9b', textDecoration: 'underline' }}>{link[1]}</a>;
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={i} style={{ color: '#182738', fontWeight: 800 }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} style={{ background: '#edf2f7', padding: '2px 5px', borderRadius: 4, fontFamily: 'monospace', fontSize: 9.5, color: '#1e6853' }}>
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}
