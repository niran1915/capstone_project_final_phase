import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle, ArrowRight, Check, Clock3, Database, Download, FileImage, Grid3x3, HardDrive,
  Image as ImageIcon, LoaderCircle, MapPinned, ShieldCheck, SlidersHorizontal, Sparkles, Video, X,
} from 'lucide-react';
import { api, formatDate, formatNumber, publicUrl } from './api.js';
import { EmptyState, ErrorNotice, MiniMetric, ModuleHeader, ResultPlaceholder, SettingField, UploadBox } from './shared.jsx';

const GRID = 20;
const SCENARIOS = [
  ['equal', 'Equal weights', 'All four features count the same (notebook baseline).'],
  ['length_emphasis', 'Length emphasis', 'Crack length density counts most.'],
  ['width_emphasis', 'Width emphasis', 'Width P95 proxy counts most.'],
  ['area_emphasis', 'Area emphasis', 'Crack area ratio counts most.'],
  ['topology_emphasis', 'Topology emphasis', 'Junction density counts most.'],
];
const IMAGE_METRICS = [
  ['priority', 'Priority index'], ['crack_area_ratio', 'Crack area'], ['length_density', 'Length density'],
  ['width_p95', 'Width P95'], ['junction_density', 'Junctions'],
];
const DAMAGE_IMAGE_METRICS = [
  ['crack', 'Crack geometry'],
  ['spalling', 'Concrete spalling'],
  ['rebar', 'Exposed/corroded rebar'],
  ['fissures', 'Fissures / Micro-cracks'],
].map(([key, label]) => [key, `${key}_area_ratio`, `${key}_relative_hotspot`, label]);
const VIDEO_METRICS = [['hotspot_persistence', 'Hotspot persistence'], ['mean_priority', 'Mean index'], ['max_priority', 'Peak index']];
const STOPS = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'];

function hexToRgb(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
function colorFor(t) {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x)); const f = x - i;
  const a = hexToRgb(STOPS[i]); const b = hexToRgb(STOPS[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`;
}
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d));
const sci = (v) => (v === null || v === undefined ? '—' : Number(v) !== 0 && Math.abs(v) < 0.001 ? Number(v).toExponential(2) : Number(v).toFixed(4));

function usePreview(file) {
  const [preview, setPreview] = useState('');
  useEffect(() => {
    if (!file || file.type.startsWith('video')) { setPreview(''); return undefined; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return preview;
}

/** Interactive 20x20 heat grid. `matrices[metric]` is a 20x20 array (null = no data). */
function GridMap({ matrices, metric, hotspot, selected, onSelect, label }) {
  const m = matrices?.[metric];
  const max = useMemo(() => {
    let v = 0; (m || []).forEach(r => r.forEach(x => { if (x !== null && x > v) v = x; })); return v || 1;
  }, [m]);
  if (!m) return null;
  return <div className="grid-map-wrap">
    <div className="grid-map" role="grid" aria-label={label}>
      {m.map((row, r) => row.map((val, c) => {
        const isHot = hotspot && hotspot[r]?.[c] > 0; const isSel = selected && selected[0] === r && selected[1] === c;
        return <button type="button" key={`${r}-${c}`} className={`grid-cell ${isHot ? 'hot' : ''} ${isSel ? 'sel' : ''} ${val === null ? 'nodata' : ''}`}
          style={val === null ? undefined : { background: colorFor(val / max) }} onClick={() => onSelect([r, c])}
          title={`row ${r}, col ${c}: ${val === null ? 'no data' : num(val, 4)}`} aria-label={`row ${r} column ${c}`} />;
      }))}
    </div>
    <div className="grid-legend"><span>0</span><div className="grid-legend-bar" style={{ background: `linear-gradient(90deg, ${STOPS.join(',')})` }} /><span>{num(max, 3)}</span>{hotspot && <span className="legend-hot"><i /> relative hotspot</span>}</div>
  </div>;
}

function Timeline({ series }) {
  if (!series?.length) return null;
  const W = 620; const H = 170; const pad = { l: 40, r: 12, t: 12, b: 26 };
  const tMax = Math.max(...series.map(p => p.timestamp_seconds), 0.001);
  const aMax = Math.max(...series.map(p => p.crack_area_ratio), 1e-6);
  const pMax = Math.max(...series.map(p => p.max_priority), 1e-6);
  const X = t => pad.l + (t / tMax) * (W - pad.l - pad.r);
  const Y = (v, mx) => pad.t + (1 - v / mx) * (H - pad.t - pad.b);
  const line = (key, mx) => series.map(p => `${X(p.timestamp_seconds).toFixed(1)},${Y(p[key], mx).toFixed(1)}`).join(' ');
  return <div className="timeline-chart">
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Crack area and peak index over time">
      {[0, 0.5, 1].map(f => <line key={f} x1={pad.l} x2={W - pad.r} y1={pad.t + f * (H - pad.t - pad.b)} y2={pad.t + f * (H - pad.t - pad.b)} stroke="#e5e9ee" />)}
      <polyline fill="none" stroke="#3276e5" strokeWidth="2" points={line('crack_area_ratio', aMax)} />
      <polyline fill="none" stroke="#df8a46" strokeWidth="2" points={line('max_priority', pMax)} />
      <text x={pad.l} y={H - 6} fontSize="10" fill="#798798">0 s</text>
      <text x={W - pad.r} y={H - 6} fontSize="10" fill="#798798" textAnchor="end">{num(tMax, 1)} s</text>
      <text x={4} y={pad.t + 8} fontSize="9" fill="#3276e5">{(aMax * 100).toFixed(1)}%</text>
    </svg>
    <div className="timeline-legend"><span><i style={{ background: '#3276e5' }} /> Crack area ratio (scaled to its max)</span><span><i style={{ background: '#df8a46' }} /> Peak cell index (scaled to its max)</span></div>
  </div>;
}

function Downloads({ items }) {
  const list = items.filter(i => i.url);
  return <div className="download-row">{list.map(i => <a key={i.label} className="download-link" href={publicUrl(i.url)} download><i.icon size={15} /> {i.label} <Download size={13} /></a>)}</div>;
}

export default function MappingPage({ projectId, notify, onInspectionComplete, pipeline, onProceedToVQA, onStepComplete }) {
  const isPipelineLocked = Boolean(pipeline?.locked && pipeline?.targetMode);
  const initialMode = pipeline?.targetMode || 'crack';
  const [tab, setTab] = useState('image');
  const [mapMode, setMapMode] = useState(initialMode);
  const [file, setFile] = useState(pipeline?.file || null);
  const preview = usePreview(file);
  const [threshold, setThreshold] = useState(0.525);
  const [scenario, setScenario] = useState('equal');
  const [tta, setTta] = useState(false);
  const [calib, setCalib] = useState('none');
  const [camera, setCamera] = useState({ height: '672', focal: '5.4', sw: '7.3', sh: '5.4', swap: false });
  const [frameStep, setFrameStep] = useState(5);
  const [maxFrames, setMaxFrames] = useState(0);
  const [loading, setLoading] = useState(false);
  const [inspection, setInspection] = useState(null);
  const [error, setError] = useState('');
  const [view, setView] = useState('overlay');
  const [metric, setMetric] = useState(initialMode === 'damage' ? 'spalling' : 'priority');
  const [selected, setSelected] = useState(null);

  const isVideo = tab === 'video';
  const isDamageMap = !isVideo && mapMode === 'damage';

  useEffect(() => {
    if (pipeline?.file) {
      setFile(pipeline.file);
      if (pipeline.targetMode) {
        const nextMode = pipeline.targetMode || 'crack';
        setMapMode(nextMode);
        setMetric(nextMode === 'damage' ? 'spalling' : 'priority');
      }
      setInspection(null);
    }
  }, [pipeline]);

  useEffect(() => { setFile(pipeline?.file || null); setInspection(null); setError(''); setSelected(null); setLoading(false); setView('overlay'); setMetric(tab === 'video' ? 'hotspot_persistence' : mapMode === 'damage' ? 'spalling' : 'priority'); }, [tab]);

  useEffect(() => {
    if (!inspection?.id || !['queued', 'processing'].includes(inspection.status)) return undefined;
    let stopped = false;
    const tick = async () => {
      try {
        const next = await api.get(`/api/inspections/${inspection.id}`);
        if (stopped) return;
        setInspection(next);
        if (['succeeded', 'failed'].includes(next.status)) {
          setLoading(false);
          if (next.status === 'succeeded') { notify('Video damage mapping completed.'); onInspectionComplete(); }
          else { setError(next.error_message || 'Video mapping failed.'); notify(next.error_message || 'Video mapping failed.', 'error'); }
        }
      } catch (err) { if (!stopped) { setLoading(false); setError(err.message); } }
    };
    const timer = window.setInterval(tick, 1800); tick();
    return () => { stopped = true; window.clearInterval(timer); };
  }, [inspection?.id, inspection?.status, notify, onInspectionComplete]);

  const run = async (e) => {
    e.preventDefault();
    if (!file) { setError(`Choose ${isVideo ? 'a video' : 'an image'} first.`); return; }
    setLoading(true); setError(''); setInspection(null); setSelected(null);
    const form = new FormData();
    form.append('file', file); form.append('calibration', calib);
    if (!isDamageMap) { form.append('threshold', String(threshold)); form.append('weight_scenario', scenario); }
    if (calib === 'camera') {
      form.append('camera_height_mm', camera.height); form.append('focal_length_mm', camera.focal);
      form.append('sensor_width_mm', camera.sw); form.append('sensor_height_mm', camera.sh); form.append('swap_axes', String(camera.swap));
    }
    if (projectId) form.append('project_id', String(projectId));
    try {
      if (isVideo) {
        form.append('frame_step', String(frameStep)); form.append('max_frames', String(maxFrames));
        const job = await api.postForm('/api/mapping/video', form); setInspection(job); notify('Video queued for damage mapping.');
      } else if (isDamageMap) {
        const result = await api.postForm('/api/mapping/damage/image', form); setInspection(result); setView('overlay');
        if (onStepComplete) onStepComplete('mapping');
        notify('Multi-class damage map completed successfully.'); onInspectionComplete(); setLoading(false);
      } else {
        form.append('tta', String(tta));
        const result = await api.postForm('/api/mapping/image', form); setInspection(result); setView('overlay');
        if (onStepComplete) onStepComplete('mapping');
        notify('Damage mapping completed successfully.'); onInspectionComplete(); setLoading(false);
      }
    } catch (err) { setLoading(false); setError(err.message); notify(err.message, 'error'); }
  };

  const data = inspection?.result || {};
  const active = loading && inspection && ['queued', 'processing'].includes(inspection.status);
  const done = inspection?.status === 'succeeded' && (isVideo ? data.kind === 'video' : data.kind === 'image' || data.kind === 'damage_image');

  return <div className="stack-lg">
    <ModuleHeader eyebrow="MODULE 04 · SPATIAL DAMAGE MAPPING" title={isDamageMap ? 'Map detector-guided damage masks' : 'Map where the cracking concentrates'} icon={MapPinned} tag={isDamageMap ? 'V4 + YOLO/FASTSAM · 20×20 IMAGE GRID' : 'V4 · 20×20 GRID · EXPLORATORY INDEX'}
      text={isDamageMap ? 'Map cracks, spalling, rebar, and fissures by image region. V4 predicts crack masks; non-crack masks are class-agnostic FastSAM proposals selected using YOLO boxes and require visual review.' : 'Segment cracks with CRACKTO-SFNet V4, measure their skeleton length, width proxy and junctions, then rank a 20×20 grid of regions for follow-up inspection. This is image-space mapping — not a geographic map or a safety score.'} />
    <div className="result-tabs mapping-mode-tabs">
      {[['image', 'Image mapping', ImageIcon], ['video', 'Video mapping', Video], ['reference', 'Research reference', Sparkles]].map(([id, label, Icon]) =>
        <button key={id} type="button" className={tab === id ? 'active violet-active' : ''} onClick={() => setTab(id)}><Icon size={13} /> {label}</button>)}
    </div>

    {tab === 'reference' ? <MappingReference /> :
    <div className="workbench-grid">
      <form className="panel control-panel" onSubmit={run}>
        <div className="panel-title-row"><div><h3>{isVideo ? 'Video mapping setup' : 'Mapping setup'}</h3><p>{isVideo ? 'Sampled frames are mapped and aggregated.' : 'Upload one image of a surface to map.'}</p></div><span className="step-badge violet-step">04</span></div>
        {isVideo
          ? <UploadBox key="v" accept="video/*,.mp4,.mov,.avi,.mkv,.webm" file={file} setFile={setFile} kind="video" label="Drop an inspection video" help="MP4, MOV, AVI, MKV or WEBM · max 250 MB" />
          : <UploadBox key="i" accept="image/*" file={file} setFile={setFile} kind="mapimage" label="Drop a surface image" help="JPG, PNG, WEBP or TIFF · max 250 MB" />}
        {!isVideo && <><div className="settings-divider" /><div className="settings-heading"><SlidersHorizontal size={16} /><span>Image mapping model</span></div>
          <div className="segmented-options">{[['crack', 'Crack geometry'], ['damage', 'All detector-guided labels']].map(([id, label]) => <button type="button" key={id} className={mapMode === id ? 'selected' : ''} disabled={isPipelineLocked && mapMode !== id} title={isPipelineLocked && mapMode !== id ? 'Locked by pipeline workflow' : ''} onClick={() => { setMapMode(id); setInspection(null); setError(''); setSelected(null); setView('overlay'); setMetric(id === 'damage' ? 'spalling' : 'priority'); }}>{label}</button>)}</div>
          {isDamageMap&&<p className="measurement-note"><AlertTriangle size={15}/><span>Crack masks come from V4. For other labels, YOLO supplies the class and FastSAM supplies a class-agnostic mask proposal. Review results against the source photo; the grid is not a severity score.</span></p>}
          {isPipelineLocked && (
            <span className="pipeline-locked-notice">
              <ShieldCheck size={12} /> Model locked to {mapMode === 'crack' ? 'Crack Geometry' : 'Non-crack labels'} from Detection step
            </span>
          )}
        </>}
        <div className="settings-divider" /><div className="settings-heading"><SlidersHorizontal size={16} /><span>Mapping configuration</span></div>
        {!isDamageMap && <SettingField label="Mask threshold" value={threshold.toFixed(3)} help="0.525 is the threshold selected on the validation split in the research notebook.">
          <input className="range-input violet-range" type="range" min="0.15" max="0.8" step="0.025" value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
          <div className="range-ends"><span>More cracks</span><span>Stricter</span></div>
        </SettingField>}
        {!isDamageMap && <SettingField label="Index weighting" value={SCENARIOS.find(s => s[0] === scenario)?.[1]} help={SCENARIOS.find(s => s[0] === scenario)?.[2]}>
          <select className="select-control" value={scenario} onChange={e => setScenario(e.target.value)}>{SCENARIOS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select>
        </SettingField>}
        {isVideo ? <>
          <SettingField label="Frame step" value={`every ${frameStep}${frameStep === 1 ? 'st' : 'th'} frame`} help="Larger steps run faster. Each sampled frame needs one V4 inference.">
            <input className="range-input violet-range" type="range" min="1" max="30" step="1" value={frameStep} onChange={e => setFrameStep(Number(e.target.value))} />
          </SettingField>
          <SettingField label="Maximum sampled frames" value={maxFrames ? formatNumber(maxFrames) : 'Whole video'} help="Set a cap to bound processing time on long clips.">
            <select className="select-control" value={maxFrames} onChange={e => setMaxFrames(Number(e.target.value))}>{[[0, 'Whole video'], [60, '60 frames'], [150, '150 frames'], [300, '300 frames'], [600, '600 frames']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
          </SettingField></> :
          !isDamageMap && <label className="toggle-row"><span className="toggle-copy"><b>Flip test-time augmentation</b><small>Average four orientations. Slower, may stabilise faint cracks.</small></span><input type="checkbox" checked={tta} onChange={e => setTta(e.target.checked)} /><span className="toggle-ui" /></label>
        }
        <SettingField label="Physical calibration" value={calib === 'none' ? 'Off (analysis px)' : calib === 'sut_preset' ? 'SUT preset' : 'Custom camera'} help="Off by default, as in the notebook. Only enable it when the camera geometry is verified.">
          <div className="segmented-options">{[['none', 'Off'], ['sut_preset', 'SUT preset'], ['camera', 'Custom']].map(([id, l]) => <button type="button" key={id} className={calib === id ? 'selected' : ''} onClick={() => setCalib(id)}>{l}</button>)}</div>
        </SettingField>
        {calib === 'sut_preset' && <p className="calib-note">Notebook camera: 672 mm height, 5.4 mm focal length, 7.3 × 5.4 mm sensor. Valid only for images taken with that setup.</p>}
        {calib === 'camera' && <div className="calib-grid">
          {[['height', 'Camera height (mm)'], ['focal', 'Focal length (mm)'], ['sw', 'Sensor width (mm)'], ['sh', 'Sensor height (mm)']].map(([k, l]) =>
            <label key={k}><span>{l}</span><input className="num-input" type="number" min="0.01" step="any" value={camera[k]} onChange={e => setCamera({ ...camera, [k]: e.target.value })} /></label>)}
          <label className="calib-swap"><input type="checkbox" checked={camera.swap} onChange={e => setCamera({ ...camera, swap: e.target.checked })} /> Swap sensor axes (portrait capture)</label>
        </div>}
        {error && <ErrorNotice message={error} />}
        <button className="button button-primary button-wide" type="submit" disabled={loading || !file}>
          {active ? <><LoaderCircle size={17} className="spin" /> Mapping frames…</> : loading ? <><LoaderCircle size={17} className="spin" /> {isVideo ? 'Starting job…' : 'Mapping damage…'}</> : <>{isVideo ? 'Start video mapping' : isDamageMap ? 'Map non-crack labels' : 'Generate crack map'} <ArrowRight size={16} /></>}
        </button>
        <p className="panel-footnote"><ShieldCheck size={13} /> {isDamageMap ? 'Uses V4 crack masks plus YOLO labels and FastSAM proposals for non-crack labels. Review proposal masks.' : 'Uses the V4 crack checkpoint plus fixed validation-derived normalization.'}</p>
      </form>

      <div className="panel result-panel">
        <div className="panel-title-row"><div><h3>{isVideo ? 'Video mapping output' : 'Damage map'}</h3>
          <p>{inspection ? `${inspection.source_filename} · ${inspection.status}${inspection.completed_at ? ` · ${formatDate(inspection.completed_at)}` : ''}` : 'Overlay, heat grid, ranked regions and exports will appear here.'}</p></div>
          {inspection && <span className={`status-badge ${inspection.status}`}>{inspection.status === 'succeeded' ? <Check size={12} /> : inspection.status === 'failed' ? <AlertTriangle size={12} /> : <LoaderCircle size={12} className="spin" />}{inspection.status}</span>}</div>
        {active ? <div className="job-progress"><div className="job-progress-icon"><LoaderCircle size={24} className="spin" /></div><h3>Mapping sampled frames</h3><p>Each frame is segmented, measured and scored on the 20×20 grid.</p>
          <div className="progress-track"><span style={{ width: `${Number(data.progress_percent || 0) || 3}%` }} /></div>
          <div className="progress-labels"><span>{data.processed_frames ? `${formatNumber(data.processed_frames)} frames mapped` : 'Preparing video reader'}</span><b>{data.progress_percent ? `${data.progress_percent}%` : 'In progress'}</b></div>
          <div className="job-note"><Clock3 size={14} /> Time depends on clip length, frame step and available GPU.</div></div>
        : inspection?.status === 'failed' ? <ErrorNotice large message={inspection.error_message || error || 'The mapping job failed.'} />
        : done && !isVideo ? data.kind === 'damage_image'
          ? <DamageImageResult data={data} view={view} setView={setView} metric={metric} setMetric={setMetric} selected={selected} setSelected={setSelected} />
          : <ImageResult data={data} view={view} setView={setView} metric={metric} setMetric={setMetric} selected={selected} setSelected={setSelected} />
        : done && isVideo ? <VideoResult data={data} metric={metric} setMetric={setMetric} selected={selected} setSelected={setSelected} />
        : <ResultPlaceholder file={isVideo ? null : file} preview={preview} icon={MapPinned} title={isVideo ? 'Awaiting video mapping' : 'Awaiting damage mapping'}
            text={isVideo ? 'Upload a clip to map sampled frames and see which grid regions are flagged most persistently.' : 'Generate a map to see the segmentation overlay, a 20×20 heat grid and the highest-ranked regions.'} />}
        {done && onProceedToVQA && (
          <div className="pipeline-next-step-card">
            <div className="pipeline-next-step-info">
              <span className="pipeline-step-badge">FINAL STEP</span>
              <h4>Consult AI Inspector (VQA)</h4>
              <p>Send the mapped measurements and structural telemetry to the civil engineering AI assistant.</p>
            </div>
            <button
              type="button"
              className="button button-primary pipeline-next-btn"
              onClick={() => onProceedToVQA({
                inspectionId: inspection.id,
                file,
                sourceModule: isVideo ? 'video' : isDamageMap ? 'damage_mapping' : 'mapping',
              })}
            >
              <span>Open AI Inspector (VQA)</span>
              <ArrowRight size={16} />
            </button>
          </div>
        )}
      </div>
    </div>}
  </div>;
}

function CellInspector({ matrices, selected, rows }) {
  if (!selected) return <p className="cell-hint"><Grid3x3 size={14} /> Select a grid cell on the overlay or heat grid to inspect it.</p>;
  const [r, c] = selected;
  const g = (k) => matrices?.[k]?.[r]?.[c];
  return <div className="cell-inspector"><b>Cell row {r}, col {c}</b>
    {rows.map(([k, label, f]) => <span key={k}>{label}<strong>{f(g(k))}</strong></span>)}
    {matrices?.hotspot && <span>Relative hotspot<strong>{g('hotspot') > 0 ? 'Yes' : 'No'}</strong></span>}</div>;
}

function DamageImageResult({ data, view, setView, metric, setMetric, selected, setSelected }) {
  const classes = data.classes || [];
  const metrics = data.class_metrics || {};
  const activeMetric = DAMAGE_IMAGE_METRICS.find(([key]) => key === metric) || DAMAGE_IMAGE_METRICS[0];
  const [, matrixKey, hotspotKey] = activeMetric;
  const clickOverlay = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setSelected([
      Math.min(GRID - 1, Math.floor(((event.clientY - rect.top) / rect.height) * GRID)),
      Math.min(GRID - 1, Math.floor(((event.clientX - rect.left) / rect.width) * GRID)),
    ]);
  };
  const downloads = [
    { label: 'Color class mask', url: data.mask_url, icon: FileImage },
    { label: 'Numeric class IDs', url: data.class_ids_url, icon: Database },
    { label: 'Model support image (not calibrated)', url: data.probability_url, icon: FileImage },
    { label: 'Contours', url: data.contours_url, icon: FileImage },
    { label: 'Regions CSV', url: data.regions_csv_url, icon: HardDrive },
    { label: 'Summary JSON', url: data.summary_json_url, icon: Database },
  ];
  const tabs = [['overlay', 'Overlay'], ['grid', 'Class heat grid'], ['mask', 'Class mask'], ['confidence', 'Model support']];
  const imageForView = view === 'overlay' ? data.overlay_url : view === 'mask' ? data.mask_url : data.probability_url;
  return <>
    <div className="result-tabs">{tabs.map(([id, label]) => <button key={id} type="button" className={view === id ? 'active violet-active' : ''} onClick={() => setView(id)}>{label}</button>)}</div>
    {view === 'grid' ? <>
      <div className="metric-chips">{DAMAGE_IMAGE_METRICS.map(([key, , , label]) => <button key={key} type="button" className={metric === key ? 'active' : ''} onClick={() => setMetric(key)}>{label}</button>)}</div>
      <GridMap matrices={data.matrices} metric={matrixKey} hotspot={data.matrices?.[hotspotKey]} selected={selected} onSelect={setSelected} label={`${activeMetric[3]} 20 by 20 image grid`} />
    </> : <div className="result-image-frame map-frame">
      {view === 'overlay' ? <div className="map-stage" onClick={clickOverlay}>
        <img src={publicUrl(imageForView)} alt="Multi-class damage map overlay" draggable="false" />
        {selected && <span className="map-selected" style={{ left: `${selected[1] * 100 / GRID}%`, top: `${selected[0] * 100 / GRID}%`, width: `${100 / GRID}%`, height: `${100 / GRID}%` }} />}
      </div> : <img src={publicUrl(imageForView)} alt={view === 'mask' ? 'Multi-class damage mask' : 'Pixel confidence image'} />}
      <a className="image-download" href={publicUrl(imageForView)} download><Download size={14} /> Download image</a>
    </div>}
    <div className="class-legend" style={{ display: 'flex', flexWrap: 'wrap', gap: 12, margin: '12px 0' }}>{classes.map(item => <span key={item.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}><i style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: `rgb(${item.color.join(',')})` }} />{item.label}</span>)}</div>
    <p className="surface-note">Detector boxes by class: {Object.entries(data.class_detection_counts || {}).filter(([, count]) => count > 0).map(([label, count]) => `${label}: ${count}`).join(' · ') || 'none'}.</p>
    {data.class_validation?.classes && <div className="surface-note" style={{ marginTop: 8 }}>
      <b>External per-class validation (not this image)</b>
      {Object.entries(data.class_validation.classes).map(([label, item]) => <div key={label} style={{ marginTop: 4 }}>
        <b>{label.replaceAll('_', ' ')} — {String(item.status || 'unavailable').replaceAll('_', ' ')}</b>: {item.summary}
      </div>)}
      {data.class_validation.dataset_url && <a href={data.class_validation.dataset_url} target="_blank" rel="noreferrer">{data.class_validation.dataset || 'Validation dataset'}</a>}
    </div>}
    {view === 'grid' && <p className="surface-note">Cell values are the share of the 256×256 analysis mask assigned to each class. Orange outlines mark relative within-image hotspots, not an absolute severity threshold.</p>}
    <div className="result-stat-row four-stats">{DAMAGE_IMAGE_METRICS.map(([key, , , label]) => {
      const item = metrics[key] || {};
      return <MiniMetric key={key} label={`${label} area`} value={`${num(item.coverage_percent, 2)}%`} />;
    })}</div>
    <div className="measurement-grid">{DAMAGE_IMAGE_METRICS.map(([key, , , label]) => {
      const item = metrics[key] || {};
      const isCalibrated = data.units === 'mm' && item.area_mm2 != null;
      const area = isCalibrated ? `${num(item.area_mm2)} mm²` : `${formatNumber(item.pixel_area || 0)} px²`;
      const perimeter = isCalibrated && item.perimeter_mm != null ? `${num(item.perimeter_mm)} mm` : `${num(item.perimeter_px)} px`;
      const extraBits = [];
      if (isCalibrated && item.width_p95_mm != null) extraBits.push(`Width P95: ${num(item.width_p95_mm)} mm`);
      else if (item.width_p95_px != null) extraBits.push(`Width P95: ${num(item.width_p95_px)} px`);
      if (isCalibrated && item.centerline_length_mm != null) extraBits.push(`Length: ${num(item.centerline_length_mm)} mm`);
      else if (item.centerline_length_px != null) extraBits.push(`Length: ${num(item.centerline_length_px)} px`);
      if (isCalibrated && item.equivalent_diameter_mm != null) extraBits.push(`Equiv. dia: ${num(item.equivalent_diameter_mm)} mm`);
      else if (item.equivalent_diameter_px != null) extraBits.push(`Equiv. dia: ${num(item.equivalent_diameter_px)} px`);
      return <div key={key}><span>{label} · area / components</span><b>{area} / {formatNumber(item.component_count || 0)}</b><small>Perimeter: {perimeter}{extraBits.length ? ` · ${extraBits.join(' · ')}` : ''}</small></div>;
    })}</div>
    {data.units === 'mm' && data.calibration?.scale_x_mm_per_px && (
      <div className="measurement-note" style={{ marginBottom: 10 }}>
        <ShieldCheck size={15} />
        <span>Calibrated at 256×256: {num(data.calibration.scale_x_mm_per_px, 3)} × {num(data.calibration.scale_y_mm_per_px, 3)} mm/px (FOV {num(data.calibration.field_of_view_mm?.[0], 1)} × {num(data.calibration.field_of_view_mm?.[1], 1)} mm).</span>
      </div>
    )}
    <CellInspector matrices={data.matrices} selected={selected} rows={[
      [activeMetric[1], `${activeMetric[3]} coverage`, value => `${num(Number(value) * 100, 2)}%`],
    ]} />
    <Downloads items={downloads} />
    <div className="measurement-note"><AlertTriangle size={15} /><span>{data.measurement_note} Grid coordinates are image-space only; outputs are not geographic locations or a structural-safety assessment.</span></div>
  </>;
}

function ImageResult({ data, view, setView, metric, setMetric, selected, setSelected }) {
  const m = data.metrics || {}; const mm = data.units === 'mm' && data.physical;
  const clickOverlay = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setSelected([Math.min(GRID - 1, Math.floor(((e.clientY - rect.top) / rect.height) * GRID)), Math.min(GRID - 1, Math.floor(((e.clientX - rect.left) / rect.width) * GRID))]);
  };
  const tabs = [['overlay', 'Overlay'], ['grid', 'Heat grid'], ['surface', '2.5D surface'], ['card', 'Evidence card']];
  const pct = data.test_set_percentiles || {};
  const exportItems = [
    { label: 'Evidence card', url: data.evidence_card_url, icon: FileImage }, { label: 'Regions CSV', url: data.regions_csv_url, icon: HardDrive },
    { label: 'Arrays NPZ', url: data.arrays_npz_url, icon: Database }, { label: 'Summary JSON', url: data.summary_json_url, icon: Database },
    { label: 'PDF report', url: data.report_pdf_url, icon: FileImage },
  ];
  return <>
    <div className="result-tabs">{tabs.map(([id, l]) => <button key={id} type="button" className={view === id ? 'active violet-active' : ''} onClick={() => setView(id)}>{l}</button>)}</div>
    {view === 'overlay' && <div className="result-image-frame map-frame"><div className="map-stage" onClick={clickOverlay}>
      <img src={publicUrl(data.overlay_url)} alt="Damage map overlay" draggable="false" />
      {selected && <span className="map-selected" style={{ left: `${selected[1] * 100 / GRID}%`, top: `${selected[0] * 100 / GRID}%`, width: `${100 / GRID}%`, height: `${100 / GRID}%` }} />}
    </div><a className="image-download" href={publicUrl(data.overlay_url)} download><Download size={14} /> Download overlay</a></div>}
    {view === 'grid' && <><div className="metric-chips">{IMAGE_METRICS.map(([k, l]) => <button key={k} type="button" className={metric === k ? 'active' : ''} onClick={() => setMetric(k)}>{l}</button>)}</div>
      <GridMap matrices={data.matrices} metric={metric} hotspot={data.matrices?.hotspot} selected={selected} onSelect={setSelected} label="Damage heat grid" /></>}
    {view === 'surface' && <div className="result-image-frame"><img src={publicUrl(data.surface_url)} alt="2.5D inspection-index surface" /><a className="image-download" href={publicUrl(data.surface_url)} download><Download size={14} /> Download</a></div>}
    {view === 'card' && <div className="result-image-frame"><img src={publicUrl(data.evidence_card_url)} alt="Evidence card" /><a className="image-download" href={publicUrl(data.evidence_card_url)} download><Download size={14} /> Download</a></div>}
    <p className="surface-note">{view === 'surface' ? 'Vertical axis is the derived index, not physical depth.' : 'Orange cells are the top quartile of non-zero cells within this image — a relative ranking, not an absolute threshold.'}</p>

    <CellInspector matrices={data.matrices} selected={selected} rows={[
      ['priority', 'Priority index', v => num(v, 3)], ['crack_area_ratio', 'Crack area ratio', v => `${num(100 * v, 2)}%`],
      ['length_density', 'Length density', v => sci(v)], ['width_p95', 'Width P95 (px)', v => num(v, 1)], ['junction_density', 'Junction density', v => sci(v)]]} />

    <div className="result-stat-row four-stats">
      <MiniMetric label="Hotspot cells" value={`${data.hotspot_count} / ${data.total_cells}`} />
      <MiniMetric label="Crack area" value={`${num(100 * m.crack_area_ratio, 2)}%`} />
      <MiniMetric label={mm ? 'Crack length' : 'Crack length (px)'} value={mm ? `${formatNumber(Math.round(data.physical.crack_length_mm))} mm` : formatNumber(Math.round(m.crack_length))} />
      <MiniMetric label="Max index" value={num(data.max_priority, 3)} />
    </div>
    <div className="measurement-grid">
      <div><span>Width median / P95 / max {mm ? '(mm)' : '(px)'}</span><b>{mm ? `${num(data.physical.width_median_mm)} / ${num(data.physical.width_p95_mm)} / ${num(data.physical.width_max_mm)}` : `${num(m.width_median)} / ${num(m.width_p95)} / ${num(m.width_max)}`}</b></div>
      <div><span>Endpoints / junction clusters</span><b>{m.endpoints} / {m.junction_clusters}</b></div>
      <div><span>Mask / skeleton components</span><b>{m.mask_components} / {m.skeleton_components}</b></div>
      <div><span>Orientation (principal axis)</span><b>{m.orientation_deg === null ? '—' : `${num(m.orientation_deg, 1)}°`}</b></div>
      <div><span>Threshold · weighting</span><b>{num(data.threshold, 3)} · {data.weight_scenario}</b></div>
      <div><span>Ambiguous probability (0.4–0.6)</span><b>{num(100 * m.ambiguous_probability_fraction, 2)}%</b></div>
    </div>
    {data.units === 'mm' && <div className="measurement-note"><AlertTriangle size={15} /><span>Physical values are estimates at the 256×256 analysis resolution ({num(data.physical.scale_x_mm_per_px, 2)} × {num(data.physical.scale_y_mm_per_px, 2)} mm per analysis pixel). The index is still computed in analysis pixels.</span></div>}

    {Object.values(pct).some(v => v !== null) && <div className="pct-row"><span>Versus the 1,200 CrackVision-12K test images:</span>
      {[['crack_area_ratio', 'area'], ['crack_length', 'length'], ['width_p95', 'width P95'], ['junction_clusters', 'junctions']].map(([k, l]) => pct[k] !== null && pct[k] !== undefined && <b key={k} className="pct-chip">{l}: {pct[k]}th pct</b>)}</div>}

    <div className="result-subheading"><b>Highest-ranked regions</b><span>{data.top_cells?.length || 0} shown</span></div>
    {data.top_cells?.length ? <div className="detections-table-wrap"><table className="data-table rank-table"><thead><tr><th>#</th><th>Cell</th><th>Index</th><th>Area</th><th>Width P95</th><th>Junctions</th></tr></thead>
      <tbody>{data.top_cells.map(c => <tr key={c.rank} className={selected && selected[0] === c.grid_row && selected[1] === c.grid_col ? 'row-selected' : ''} onClick={() => { setSelected([c.grid_row, c.grid_col]); setView('overlay'); }}>
        <td>{c.rank}</td><td className="mono-text">r{c.grid_row} · c{c.grid_col}{c.hotspot && <span className="hot-dot" title="Relative hotspot" />}</td>
        <td><div className="confidence-cell"><div className="confidence-track"><span style={{ width: `${Math.min(100, c.priority_index * 100 / Math.max(data.max_priority, 1e-6))}%` }} /></div><b>{num(c.priority_index, 2)}</b></div></td>
        <td>{num(100 * c.crack_area_ratio, 1)}%</td><td>{num(c.width_p95, 1)}</td><td>{c.junction_count}</td></tr>)}</tbody></table></div>
      : <EmptyState icon={Grid3x3} title="No crack regions found" body="Nothing exceeded the mask threshold. Try a lower threshold or a sharper, closer image." />}

    <details className="notebook-reference sens-details"><summary><Sparkles size={14} /> How stable is the ranking if the weights change?</summary>
      <table className="data-table"><thead><tr><th>Weighting</th><th>Rank correlation vs equal</th><th>Top-10% overlap (IoU)</th></tr></thead>
        <tbody>{(data.weight_sensitivity || []).map(s => <tr key={s.scenario}><td>{s.scenario}</td><td>{num(s.spearman_vs_equal, 3)}</td><td>{num(s.top10pct_overlap_iou, 2)}</td></tr>)}</tbody></table>
      <p>Computed for this image. Lower overlap means the highest-ranked cells depend on how the features are weighted.</p></details>
    <Downloads items={exportItems} />
    <details className="notebook-reference"><summary><AlertTriangle size={14} /> Interpretation limits</summary><ul className="caveat-list">{(data.caveats || []).map(c => <li key={c}>{c}</li>)}</ul></details>
  </>;
}

function VideoResult({ data, metric, setMetric, selected, setSelected }) {
  const exportItems = [
    { label: 'Annotated video', url: data.annotated_video_url, icon: Video }, { label: 'Per-frame CSV', url: data.frames_csv_url, icon: HardDrive },
    { label: 'Summary JSON', url: data.summary_json_url, icon: Database }, { label: 'Persistence map', url: data.persistence_heatmap_url, icon: FileImage },
  ];
  return <>
    <div className="video-output-frame"><video controls playsInline src={publicUrl(data.annotated_video_url)} /><a className="image-download" href={publicUrl(data.annotated_video_url)} download><Download size={14} /> Download annotated video</a></div>
    <div className="result-stat-row four-stats">
      <MiniMetric label="Frames mapped" value={formatNumber(data.processed_frame_count)} /><MiniMetric label="Frames with cracks" value={formatNumber(data.frames_with_cracks)} />
      <MiniMetric label="Peak index" value={num(data.peak_max_priority, 3)} /><MiniMetric label="Max crack area" value={`${num(100 * data.max_crack_area_ratio, 2)}%`} />
    </div>
    <div className="result-subheading"><b>Change over time</b><span>every {data.frame_step}th frame · {num(data.fps, 1)} fps</span></div>
    <Timeline series={data.series} />
    <div className="result-subheading"><b>Where it recurs</b><span>frame coordinates</span></div>
    <div className="metric-chips">{VIDEO_METRICS.map(([k, l]) => <button key={k} type="button" className={metric === k ? 'active' : ''} onClick={() => setMetric(k)}>{l}</button>)}</div>
    <GridMap matrices={data.matrices} metric={metric} selected={selected} onSelect={setSelected} label="Aggregate video heat grid" />
    <CellInspector matrices={data.matrices} selected={selected} rows={[['hotspot_persistence', 'Frames flagged', v => `${num(100 * v, 0)}%`], ['mean_priority', 'Mean index', v => num(v, 3)], ['max_priority', 'Peak index', v => num(v, 3)]]} />
    {data.peak_frame_url && <div className="result-image-frame"><img src={publicUrl(data.peak_frame_url)} alt="Peak frame" /><span className="image-download static">Peak frame · t = {num(data.peak_timestamp_seconds, 2)} s</span></div>}
    {data.persistent_cells?.length > 0 && <div className="detections-table-wrap"><table className="data-table"><thead><tr><th>Cell</th><th>Frames flagged</th><th>Mean index</th><th>Peak index</th></tr></thead>
      <tbody>{data.persistent_cells.map(c => <tr key={`${c.grid_row}-${c.grid_col}`} onClick={() => setSelected([c.grid_row, c.grid_col])}><td className="mono-text">r{c.grid_row} · c{c.grid_col}</td><td>{num(100 * c.persistence, 0)}%</td><td>{num(c.mean_priority, 3)}</td><td>{num(c.max_priority, 3)}</td></tr>)}</tbody></table></div>}
    <Downloads items={exportItems} />
    <details className="notebook-reference"><summary><AlertTriangle size={14} /> Interpretation limits</summary><ul className="caveat-list">{(data.caveats || []).map(c => <li key={c}>{c}</li>)}</ul></details>
  </>;
}

function MappingReference() {
  const [ref, setRef] = useState(null); const [err, setErr] = useState(''); const [open, setOpen] = useState(null);
  const load = useCallback(async () => { try { setRef(await api.get('/api/mapping/reference')); } catch (e) { setErr(e.message); } }, []);
  useEffect(() => { load(); }, [load]);
  if (err) return <ErrorNotice large message={err} />;
  if (!ref) return <div className="loading-rows"><LoaderCircle className="spin" size={22} /><span>Loading saved research outputs…</span></div>;
  if (!ref.available) return <EmptyState icon={Sparkles} title="Reference outputs unavailable" body={ref.error || 'The assets/reference_outputs/mapping folder was not found.'} />;
  const meta = ref.metadata || {};
  return <div className="stack-lg">
    <div className="panel ref-panel"><div className="panel-heading"><div><h3>V4 segmentation quality</h3><p>Saved evaluation behind the mapping (CrackVision-12K).</p></div><span className="subtle-tag">NOTEBOOK OUTPUT</span></div>
      <table className="data-table"><thead><tr><th>Split</th><th>Threshold</th><th>IoU</th><th>Dice</th><th>Precision</th><th>Recall</th></tr></thead>
        <tbody>{(ref.segmentation_metrics || []).map(r => <tr key={r.Split}><td>{r.Split}</td><td>{num(r.Threshold, 3)}</td><td>{num(r.IoU, 4)}</td><td>{num(r.Dice, 4)}</td><td>{num(r.Precision, 4)}</td><td>{num(r.Recall, 4)}</td></tr>)}</tbody></table></div>
    <div className="ref-two-col">
      <div className="panel ref-panel"><div className="panel-heading"><div><h3>Method</h3><p>{meta.method_version}</p></div></div>
        <div className="measurement-grid one-col"><div><span>Grid</span><b>{meta.grid?.join(' × ')}</b></div><div><span>Analysis size</span><b>{meta.analysis_size} px</b></div><div><span>Validation threshold</span><b>{meta.selected_validation_threshold}</b></div>
          <div><span>Physical units</span><b>{meta.physical_units_enabled ? 'On' : 'Off by default'}</b></div>
          {Object.entries(meta.normalization_scales_from_validation_only || {}).map(([k, v]) => <div key={k}><span>Scale · {k}</span><b>{num(v, 4)}</b></div>)}</div></div>
      <div className="panel ref-panel"><div className="panel-heading"><div><h3>Prediction vs ground-truth geometry</h3><p>{ref.prediction_vs_gt?.images} test images</p></div></div>
        {ref.prediction_vs_gt && <div className="measurement-grid one-col"><div><span>Median |area ratio| error</span><b>{sci(ref.prediction_vs_gt.median_area_ratio_abs_error)}</b></div><div><span>Median |length density| error</span><b>{sci(ref.prediction_vs_gt.median_length_density_abs_error)}</b></div>
          <div><span>Median |width P95| error (px)</span><b>{num(ref.prediction_vs_gt.median_width_p95_abs_error)}</b></div><div><span>Median index rank correlation</span><b>{num(ref.prediction_vs_gt.median_priority_spearman, 3)}</b></div><div><span>Median top-10% region IoU</span><b>{num(ref.prediction_vs_gt.median_top10_region_iou, 3)}</b></div></div>}
        <p className="surface-note">{ref.prediction_vs_gt?.note}</p></div>
    </div>
    <div className="panel ref-panel"><div className="panel-heading"><div><h3>Sensitivity diagnostics</h3><p>How much the ranking changes under different choices (50-image subsets).</p></div></div>
      <div className="ref-two-col">
        <div><b className="ref-sub">Index weighting</b><table className="data-table"><thead><tr><th>Scenario</th><th>Rank corr.</th><th>Top-10% IoU</th></tr></thead><tbody>{(ref.weight_sensitivity || []).map(r => <tr key={r.scenario}><td>{r.scenario}</td><td>{num(r.median_spearman_vs_equal, 3)}</td><td>{num(r.median_top10_iou_vs_equal, 2)}</td></tr>)}</tbody></table></div>
        <div><b className="ref-sub">Grid resolution (vs 20×20)</b><table className="data-table"><thead><tr><th>Grid</th><th>Rank corr.</th><th>Top-decile IoU</th></tr></thead><tbody>{(ref.grid_sensitivity || []).map(r => <tr key={r.comparison_grid}><td>{r.comparison_grid}</td><td>{num(r.median_spearman, 3)}</td><td>{num(r.median_top_decile_iou, 2)}</td></tr>)}</tbody></table></div>
      </div>
      {ref.threshold_sensitivity && <p className="surface-note">Threshold {ref.threshold_sensitivity.selected} vs {ref.threshold_sensitivity.alternate}: median rank correlation {num(ref.threshold_sensitivity.median_priority_rank_spearman, 3)}, median top-10% overlap {num(ref.threshold_sensitivity.median_top10_iou, 2)}, median area change {sci(ref.threshold_sensitivity.median_area_ratio_abs_change)}.</p>}
    </div>
    <div className="panel ref-panel"><div className="panel-heading"><div><h3>Saved evidence cards</h3><p>Twelve representative test images from the research run.</p></div></div>
      <div className="gallery">{(ref.visuals || []).map(v => <button type="button" key={v.id} className="gallery-item" onClick={() => setOpen(v)}><img loading="lazy" src={v.surface} alt={`2.5D index ${v.id}`} /><span>Image {v.id}</span></button>)}</div></div>
    <div className="panel ref-panel"><div className="panel-heading"><div><h3>Downloads</h3><p>Saved notebook tables.</p></div></div>
      <div className="reference-links">{(ref.downloads || []).map(d => <a key={d.url} href={d.url} download>{d.label} <Download size={13} /></a>)}</div></div>
    <div className="measurement-note"><AlertTriangle size={15} /><span>{(meta.caveats || []).join(' ')}</span></div>
    {open && <div className="modal-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) setOpen(null); }}><section className="modal-card wide-modal"><header className="modal-header"><div><span className="eyebrow dark-eyebrow">SAVED EVIDENCE CARD</span><h2>Image {open.id}</h2></div><button className="icon-button" onClick={() => setOpen(null)}><X size={18} /></button></header>
      <div className="modal-body"><img className="modal-result-image" src={open.evidence_card} alt={`Evidence card ${open.id}`} /><img className="modal-result-image" src={open.surface} alt={`2.5D surface ${open.id}`} /></div></section></div>}
  </div>;
}
