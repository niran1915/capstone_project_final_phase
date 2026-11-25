import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowRight, BarChart3, Boxes,
  Building2, Check, CheckCircle2, ChevronDown, CircleHelp, Clock3, CloudUpload,
  Command, Database, Download, FileImage, FileVideo, Filter, Gauge, HardDrive,
  History, Layers3, LoaderCircle, Menu, Moon, MoreHorizontal, Plus, Scan, ScanLine,
  Search, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Sun, Target, Trash2,
  TrendingUp, Video, Waves, X, Zap, MapPinned, MessageSquareQuote,
} from 'lucide-react';
import { api, formatDate, formatNumber, publicUrl } from './api.js';
import { EmptyState, ErrorNotice, MiniMetric, ModuleHeader, ResultPlaceholder, SettingField, UploadBox } from './shared.jsx';
import MappingPage from './MappingPage.jsx';
import VQAPage from './VQAPage.jsx';

const NAV = [
  { group: 'WORKSPACE', items: [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'detection', label: 'Defect detection', icon: ScanLine },
    { id: 'segmentation', label: 'Segmentation lab', icon: Layers3 },
    { id: 'video', label: 'Video inspection', icon: Video },
    { id: 'mapping', label: 'Damage mapping', icon: MapPinned },
    { id: 'vqa', label: 'AI Inspector (VQA)', icon: MessageSquareQuote },
  ]},
  { group: 'MANAGE', items: [
    { id: 'projects', label: 'Projects & sites', icon: Building2 },
    { id: 'history', label: 'Inspection history', icon: History },
    { id: 'system', label: 'System status', icon: Activity },
  ]},
];

const PAGE_META = {
  overview: ['Overview', 'Your structural inspection workspace at a glance.'],
  detection: ['Defect detection', 'Locate and classify visible defects in a single image.'],
  segmentation: ['Segmentation lab', 'Segment cracks with V4 or build detector-guided masks for other labels.'],
  video: ['Video inspection', 'Run frame-by-frame defect detection over an inspection clip.'],
  mapping: ['Damage mapping', 'Map crack and detector-guided damage masks onto an image grid.'],
  vqa: ['AI Inspector (VQA)', 'Ask visual questions and receive civil engineering structural assessments.'],
  projects: ['Projects & sites', 'Organize inspections by structure, site, and project.'],
  history: ['Inspection history', 'Review source files, results, and completed model runs.'],
  system: ['System status', 'Check model artifacts and inference dependencies.'],
};

export default function App() {
  const [page, setPage] = useState('overview');
  const [theme, setTheme] = useState(() => localStorage.getItem('crackvision-theme') || 'light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('crackvision-theme', theme);
  }, [theme]);

  const [projects, setProjects] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(() => localStorage.getItem('crackvision-project') || '');
  const [dashboard, setDashboard] = useState(null);
  const [health, setHealth] = useState(null);
  const [toast, setToast] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [mobileMenu, setMobileMenu] = useState(false);

  // Guided Pipeline State
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const onStepComplete = useCallback((stepId) => {
    setCompletedSteps(prev => new Set([...prev, stepId]));
  }, []);

  const [pipeline, setPipeline] = useState({
    file: null,
    targetMode: 'crack', // 'crack' | 'damage'
    locked: false,
    detectionResult: null,
    segmentationInspection: null,
    mappingInspectionId: '',
  });

  const onProceedToSegmentation = useCallback(({ file, mode, detectionResult }) => {
    setPipeline(prev => ({
      ...prev,
      file,
      targetMode: mode,
      locked: true,
      detectionResult,
    }));
    setCompletedSteps(prev => new Set([...prev, 'detection']));
    setPage('segmentation');
  }, []);

  const onProceedToMapping = useCallback(({ file, mode, segmentationInspection }) => {
    setPipeline(prev => ({
      ...prev,
      file,
      targetMode: mode,
      locked: true,
      segmentationInspection,
    }));
    setCompletedSteps(prev => new Set([...prev, 'segmentation']));
    setPage('mapping');
  }, []);

  const onProceedToVQA = useCallback(({ inspectionId, file }) => {
    setPipeline(prev => ({
      ...prev,
      mappingInspectionId: inspectionId,
      file,
    }));
    setCompletedSteps(prev => new Set([...prev, 'mapping']));
    setPage('vqa');
  }, []);

  const notify = useCallback((message, kind = 'success') => {
    setToast({ message, kind });
    window.setTimeout(() => setToast(null), 4200);
  }, []);

  const refreshWorkspace = useCallback(async () => {
    try {
      const [projectList, dashboardInfo, healthInfo] = await Promise.all([
        api.get('/api/projects'), api.get('/api/dashboard'), api.get('/api/health'),
      ]);
      setProjects(projectList);
      setDashboard(dashboardInfo);
      setHealth(healthInfo);
      const chosen = selectedProjectId && projectList.some(p => String(p.id) === String(selectedProjectId))
        ? String(selectedProjectId) : String(projectList[0]?.id || '');
      setSelectedProjectId(chosen);
      if (chosen) localStorage.setItem('crackvision-project', chosen);
    } catch (error) {
      setHealth({ status: 'offline', error: error.message, modules: {} });
    }
  }, [selectedProjectId]);

  useEffect(() => { refreshWorkspace(); }, [refreshWorkspace, refreshKey]);
  useEffect(() => {
    if (selectedProjectId) localStorage.setItem('crackvision-project', selectedProjectId);
  }, [selectedProjectId]);

  const onInspectionComplete = useCallback(() => setRefreshKey(value => value + 1), []);
  const projectId = selectedProjectId ? Number(selectedProjectId) : undefined;
  const title = PAGE_META[page]?.[0] || 'Overview';
  const online = health?.status === 'ok';

  const renderPage = () => {
    const shared = { projectId, projects, notify, onInspectionComplete, onStepComplete };
    switch (page) {
      case 'detection': return <DetectionPage {...shared} onProceedToSegmentation={onProceedToSegmentation} />;
      case 'segmentation': return <SegmentationPage {...shared} pipeline={pipeline} onProceedToMapping={onProceedToMapping} />;
      case 'video': return <VideoPage {...shared} />;
      case 'mapping': return <MappingPage {...shared} pipeline={pipeline} onProceedToVQA={onProceedToVQA} />;
      case 'vqa': return <VQAPage {...shared} pipeline={pipeline} />;
      case 'projects': return <ProjectsPage projects={projects} notify={notify} onRefresh={() => setRefreshKey(v => v + 1)} />;
      case 'history': return <HistoryPage projects={projects} onRefresh={onInspectionComplete} notify={notify} />;
      case 'system': return <SystemPage health={health} onRefresh={() => setRefreshKey(v => v + 1)} />;
      default: return <DashboardPage dashboard={dashboard} health={health} projects={projects} onNavigate={setPage} />;
    }
  };

  return <div className={`app-shell theme-${theme}`}>
    <aside className={`sidebar ${mobileMenu ? 'sidebar-open' : ''}`}>
      <div className="brand-row">
        <div className="brand-mark"><Waves size={22} strokeWidth={2.4} /></div>
        <div className="brand-copy"><span>CrackVision</span><small>STRUCTURAL INTELLIGENCE</small></div>
        <button className="icon-button mobile-close" aria-label="Close menu" onClick={() => setMobileMenu(false)}><X size={18} /></button>
      </div>
      <div className="workspace-switcher">
        <div className="workspace-avatar"><Building2 size={17} /></div>
        <div className="workspace-text"><span>Inspection workspace</span><small>{projects.length} {projects.length === 1 ? 'project' : 'projects'}</small></div>
        <ChevronDown size={15} className="muted" />
      </div>
      <nav className="side-nav">
        {NAV.map(group => <div className="nav-group" key={group.group}>
          <div className="nav-label">{group.group}</div>
          {group.items.map(item => {
            const Icon = item.icon;
            const isCompleted = completedSteps.has(item.id);
            const isActive = page === item.id;
            return <button key={item.id} className={`nav-item ${isActive ? 'active' : ''} ${isCompleted ? 'step-completed' : ''}`} onClick={() => { setPage(item.id); setMobileMenu(false); }}>
              <Icon size={18} strokeWidth={1.9} />
              <span>{item.label}</span>
              {isCompleted ? (
                <span className="step-done-badge" title="Stage completed">
                  <Check size={11} strokeWidth={2.8} />
                </span>
              ) : (
                item.id === 'detection' && <span className="nav-dot" />
              )}
            </button>;
          })}
        </div>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="model-ready-card">
          <div className="ready-icon"><ShieldCheck size={18} /></div>
          <div><b>Model workspace</b><span>{online ? 'API connected' : 'Connection check'}</span></div>
          <span className={`status-pip ${online ? 'good' : 'warn'}`} />
        </div>
        <div className="sidebar-footer"><span className="footer-avatar">CV</span><span><b>Engineer workspace</b><small>Local environment</small></span><MoreHorizontal size={17} className="muted" /></div>
      </div>
    </aside>
    {mobileMenu && <button className="mobile-overlay" aria-label="Close navigation" onClick={() => setMobileMenu(false)} />}
    <main className="main-area">
      <header className="topbar">
        <div className="topbar-left">
          <button className="icon-button mobile-menu-button" onClick={() => setMobileMenu(true)} aria-label="Open navigation"><Menu size={20} /></button>
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><b>{title}</b></div>
        </div>
        <div className="topbar-actions">
          <button
            type="button"
            className="theme-toggle-button"
            onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            aria-label="Toggle light or dark theme"
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          <label className="project-select-wrap"><Building2 size={15} /><select value={selectedProjectId} onChange={e => setSelectedProjectId(e.target.value)} aria-label="Active project">
            {projects.length === 0 && <option value="">No project</option>}
            {projects.map(project => <option value={project.id} key={project.id}>{project.name}</option>)}
          </select><ChevronDown size={14} /></label>
          <span className={`api-state ${online ? 'online' : 'offline'}`}><span />{online ? 'API connected' : 'API offline'}</span>
          <button className="avatar-button" title="Local workspace">CV</button>
        </div>
      </header>
      <section className="page-content">
        <div className="page-heading"><div><h1>{title}</h1><p>{PAGE_META[page]?.[1]}</p></div>
          <div className="heading-actions">{page === 'overview' && <span className="date-chip"><Clock3 size={14} /> Live workspace</span>}{page !== 'overview' && <span className="date-chip"><ShieldCheck size={14} /> Research model build</span>}</div>
        </div>
        {renderPage()}
        <footer className="main-footer"><span>CrackVision AI <span className="footer-sep">·</span> Capstone prototype</span><span>Detection <i /> Segmentation <i /> Video pipeline <i /> Damage mapping</span></footer>
      </section>
    </main>
    {toast && <div className={`toast ${toast.kind}`}><span className="toast-icon">{toast.kind === 'error' ? <AlertTriangle size={17} /> : <CheckCircle2 size={17} />}</span><span>{toast.message}</span><button aria-label="Dismiss notification" onClick={() => setToast(null)}><X size={15} /></button></div>}
  </div>;
}

function DashboardPage({ dashboard, health, projects, onNavigate }) {
  const stats = dashboard || {};
  const moduleCounts = stats.by_module || {};
  const modules = [
    { id: 'detection', num: '01', title: 'Crack detection', copy: 'Find and classify visible defects in an image.', icon: ScanLine, tint: 'blue', action: 'Run detection', count: moduleCounts.detection || 0 },
    { id: 'segmentation', num: '02', title: 'Crack segmentation', copy: 'Trace pixel-level crack regions and geometry.', icon: Layers3, tint: 'green', action: 'Open segmentation', count: moduleCounts.segmentation || 0 },
    { id: 'video', num: '03', title: 'Video inspection', copy: 'Scan inspection footage frame by frame.', icon: Video, tint: 'orange', action: 'Analyze video', count: moduleCounts.video || 0 },
    { id: 'mapping', num: '04', title: 'Damage mapping', copy: 'Map crack length, width and topology into ranked inspection regions.', icon: MapPinned, tint: 'violet', action: 'Open damage mapping', count: moduleCounts.mapping || 0 },
    { id: 'vqa', num: '05', title: 'AI Inspector (VQA)', copy: 'Ask visual questions and receive civil engineering structural assessments.', icon: MessageSquareQuote, tint: 'violet', action: 'Launch AI Inspector', count: moduleCounts.vqa || 0 },
  ];
  const counts = Object.entries(stats.class_counts || {}).sort((a,b)=>b[1]-a[1]);
  const maxCount = Math.max(1, ...counts.map(x => x[1]));
  const statusRows = [
    { label: 'Object detector', detail: 'Supplied YOLO weights', ok: !!health?.modules?.detection?.available, meta: health?.modules?.detection?.loaded ? 'Loaded' : 'Load on first run' },
    { label: 'Segmentation model', detail: 'CRACKTO-SFNet V4', ok: !!health?.modules?.segmentation?.available, meta: health?.modules?.segmentation?.loaded ? 'Loaded' : 'Load on first run' },
    { label: 'Video pipeline', detail: 'OpenCV + YOLO', ok: !!health?.modules?.video?.available, meta: 'Frame-by-frame' },
    { label: 'Damage mapping', detail: 'V4 + spatial grid analysis', ok: !!health?.modules?.mapping?.available, meta: 'Shares the V4 checkpoint' },
    { label: 'AI Inspector (VQA)', detail: 'Multimodal VLM + ACI Expert Engine', ok: true, meta: 'Inference Ready' },
  ];
  return <div className="stack-lg">
    <section className="hero-banner">
      <div className="hero-copy"><div className="eyebrow"><Sparkles size={14} /> STRUCTURAL AI WORKSPACE</div><h2>See beneath the surface.<br/><em>Understand every defect.</em></h2><p>One workspace for crack detection, precise segmentation, video inspection, spatial damage mapping, and AI VQA inspection.</p><div className="hero-actions"><button className="button button-primary" onClick={() => onNavigate('detection')}>Start an inspection <ArrowRight size={16} /></button><button className="button button-ghost" onClick={() => onNavigate('history')}>View inspection history</button></div></div>
      <div className="hero-art" aria-hidden="true"><div className="hero-grid"/><div className="wire-cube cube-one"><span/><span/><span/></div><div className="wire-cube cube-two"><span/><span/><span/></div><div className="hero-orbit orbit-one"/><div className="hero-orbit orbit-two"/><div className="hero-crosshair"><Target size={34}/></div><div className="hero-art-label"><span className="status-pip good"/> INFERENCE READY</div></div>
      <div className="hero-footnote"><span><ShieldCheck size={13}/> Research pipeline</span><span>5 inspection modules</span><span>Shared inspection history</span></div>
    </section>
    <section className="stat-grid">
      <StatCard label="Total inspections" value={formatNumber(stats.total_inspections)} sub={`${formatNumber(stats.completed_inspections)} completed`} icon={Activity} accent="blue" trend="PROJECT ACTIVITY" />
      <StatCard label="Defects detected" value={formatNumber(stats.defects_detected)} sub="Across successful runs" icon={Target} accent="orange" trend="MODEL OUTPUT" />
      <StatCard label="Active projects" value={formatNumber(stats.active_projects ?? projects.length)} sub="Inspection sites" icon={Building2} accent="green" trend="FIELD WORKSPACE" />
      <StatCard label="Modules available" value={`${statusRows.filter(s=>s.ok).length}/${statusRows.length}`} sub="Weights and runtime checked" icon={Gauge} accent="violet" trend="SYSTEM STATUS" />
    </section>
    <div className="section-heading"><div><h2>Inspection modules</h2><p>Choose a pipeline to run on a new file.</p></div><button className="text-button" onClick={() => onNavigate('system')}>System details <ArrowRight size={15}/></button></div>
    <section className="module-grid four">
      {modules.map(m => { const Icon=m.icon; return <article className="module-card" key={m.id}>
        <div className="module-card-top"><span className={`module-icon ${m.tint}`}><Icon size={20}/></span><span className="module-number">MODULE {m.num}</span><span className="module-run-count">{m.count} runs</span></div>
        <h3>{m.title}</h3><p>{m.copy}</p>
        <div className="module-card-foot"><span className="module-health"><span className={`status-pip ${health?.modules?.[m.id]?.available ? 'good' : 'warn'}`}/>{health?.modules?.[m.id]?.available ? 'Ready to run' : 'Needs setup'}</span><button className="round-arrow" onClick={() => onNavigate(m.id)} aria-label={m.action}><ArrowRight size={17}/></button></div>
      </article>; })}
    </section>
    <section className="dashboard-bottom-grid">
      <div className="panel recent-panel"><div className="panel-heading"><div><h3>Recent inspections</h3><p>Latest activity across all modules</p></div><button className="icon-button" onClick={() => onNavigate('history')} aria-label="All inspection history"><ArrowRight size={17}/></button></div>
        {stats.recent_inspections?.length ? <div className="recent-list">{stats.recent_inspections.slice(0,5).map(item => <RecentRow key={item.id} item={item} />)}</div> : <EmptyState icon={History} title="No inspections yet" body="Run a detection or segmentation job to see its results here." action="Start with detection" onAction={() => onNavigate('detection')} />}
      </div>
      <div className="panel distribution-panel"><div className="panel-heading"><div><h3>Detected defect classes</h3><p>Aggregated from completed model runs</p></div><span className="subtle-tag">LIVE TOTALS</span></div>
        {counts.length ? <div className="class-bars">{counts.slice(0,5).map(([name,value],i)=><div className="class-bar-row" key={name}><div className="class-label-row"><span>{name}</span><b>{formatNumber(value)}</b></div><div className="bar-track"><span className={`bar-fill bar-${i}`} style={{width:`${Math.max(2,(value/maxCount)*100)}%`}}/></div></div>)}</div> : <div className="empty-chart"><div className="empty-chart-graphic"><BarChart3 size={24}/></div><b>Class summary will appear here</b><span>Run an image or video detector to populate the breakdown.</span></div>}
        <div className="distribution-note"><span className="info-dot">i</span><span>Counts are predictions, not a measure of structural severity.</span></div>
      </div>
    </section>
  </div>;
}

function StatCard({label,value,sub,icon:Icon,accent,trend}) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className={`stat-icon ${accent}`}><Icon size={18}/></span></div><div className="stat-value">{value}</div><div className="stat-sub"><span className="stat-trend"><TrendingUp size={13}/>{trend}</span><span>{sub}</span></div></article>;
}

function RecentRow({item}) {
  const Icon = item.module === 'segmentation' ? Layers3 : item.module === 'video' ? Video : item.module === 'mapping' ? MapPinned : ScanLine;
  return <div className="recent-row"><span className={`recent-module-icon ${item.module}`}><Icon size={16}/></span><div className="recent-row-main"><b>{item.source_filename}</b><span>{item.project_name || 'Unassigned project'} <i/> {formatDate(item.created_at)}</span></div><span className={`status-badge ${item.status}`}>{item.status}</span></div>;
}





function DetectionPage({projectId,notify,onInspectionComplete,onProceedToSegmentation,onStepComplete}) {
  const [file,setFile] = useState(null); const [preview,setPreview] = useState('');
  const [confidence,setConfidence] = useState(0.25); const [iou,setIou] = useState(0.45); const [imageSize,setImageSize] = useState('640');
  const [loading,setLoading] = useState(false); const [inspection,setInspection] = useState(null); const [error,setError] = useState('');
  useEffect(()=>{if(!file){setPreview('');return;} const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url);},[file]);
  const run = async (e) => { e.preventDefault(); if(!file){setError('Choose an image first.');return;} setLoading(true);setError('');setInspection(null);
    const form=new FormData();form.append('file',file);form.append('confidence',String(confidence));form.append('iou',String(iou));form.append('image_size',imageSize);if(projectId)form.append('project_id',String(projectId));
    try {
      const result=await api.postForm('/api/detect',form);
      setInspection(result);
      if (onStepComplete) onStepComplete('detection');
      notify('Detection completed successfully.');
      onInspectionComplete();
    }
    catch(err){setError(err.message);notify(err.message,'error');}finally{setLoading(false);}
  };
  const data=inspection?.result;
  const detections = data?.detections || [];
  const hasCrack = detections.some(d => (d.class_name || '').toLowerCase().includes('crack') && !(d.class_name || '').toLowerCase().includes('fissure'));
  const hasOther = detections.some(d => {
    const name = (d.class_name || '').toLowerCase();
    return name.includes('spalling') || name.includes('rebar') || name.includes('fissure');
  });
  const targetMode = hasOther ? 'damage' : 'crack';
  const isCrackTarget = targetMode === 'crack';

  return <div className="stack-lg"><ModuleHeader eyebrow="MODULE 01 · OBJECT DETECTION" title="Find defects in a photograph" text="Upload one structural image. The image detector returns defect boxes, class labels, and confidence scores. Regional evidence counts box centers; it is not pixel segmentation or damaged-area measurement." icon={ScanLine} tag="YOLO · 640 PX DEFAULT" />
    <div className="workbench-grid"><form className="panel control-panel" onSubmit={run}><div className="panel-title-row"><div><h3>Inspection setup</h3><p>Configure detection for the source image.</p></div><span className="step-badge">01</span></div>
      <UploadBox accept="image/*" file={file} setFile={setFile} label="Drop your structural image" help="JPG, PNG, WEBP or TIFF · max 250 MB" />
      <div className="settings-divider"/><div className="settings-heading"><SlidersHorizontal size={16}/><span>Inference settings</span></div>
      <SettingField label="Confidence threshold" value={`${Math.round(confidence*100)}%`} help="Higher values reduce low-confidence detections."><input className="range-input" type="range" min="0.05" max="0.9" step="0.05" value={confidence} onChange={e=>setConfidence(Number(e.target.value))}/><div className="range-ends"><span>More detections</span><span>More selective</span></div></SettingField>
      <SettingField label="NMS IoU threshold" value={iou.toFixed(2)} help="Controls overlap suppression between boxes."><input className="range-input orange-range" type="range" min="0.1" max="0.8" step="0.05" value={iou} onChange={e=>setIou(Number(e.target.value))}/></SettingField>
      <SettingField label="Inference image size" value={`${imageSize} × ${imageSize}`}><div className="segmented-options">{['320','416','512','640','768','960'].map(v=><button type="button" key={v} className={imageSize===v?'selected':''} onClick={()=>setImageSize(v)}>{v}</button>)}</div></SettingField>
      {error&&<ErrorNotice message={error}/>}
      <button className="button button-primary button-wide" type="submit" disabled={loading||!file}>{loading?<><LoaderCircle size={17} className="spin"/> Running detector…</>:<>Run image detection <ArrowRight size={16}/></>}</button>
      <p className="panel-footnote"><ShieldCheck size={13}/> Uses the dedicated image checkpoint. The video module uses a separate checkpoint.</p>
    </form>
    <div className="panel result-panel"><div className="panel-title-row"><div><h3>Detection result</h3><p>{inspection?`Inspection ${inspection.id.slice(0,8)} · ${formatDate(inspection.completed_at)}`:'Your annotated image and model predictions will appear here.'}</p></div>{inspection&&<span className="status-badge succeeded"><Check size={12}/> Completed</span>}</div>
      {inspection&&data ? <><div className="result-image-frame"><img src={publicUrl(data.annotated_image_url)} alt="YOLO annotated structural image"/><a className="image-download" href={publicUrl(data.annotated_image_url)} download><Download size={14}/> Download annotated</a></div>
        <div className="result-stat-row"><MiniMetric label="Detections" value={formatNumber(data.detection_count)}/><MiniMetric label="Max confidence" value={`${Math.round((data.maximum_confidence||0)*100)}%`}/><MiniMetric label="Classes found" value={Object.keys(data.class_counts||{}).length}/></div>
        <RegionalEvidenceMap evidence={data.regional_evidence}/>
        <div className="result-subheading"><b>Predicted defects</b><span>{data.detections?.length||0} boxes</span></div>
        {data.detections?.length ? <div className="detections-table-wrap"><table className="data-table"><thead><tr><th>Class</th><th>Confidence</th><th>Box (x1, y1, x2, y2)</th></tr></thead><tbody>{data.detections.map((d,i)=><tr key={`${d.class_id}-${i}`}><td><span className="class-pill"><span className={`class-dot class-${d.class_id%4}`}/>{d.class_name}</span></td><td><ConfidenceBar value={d.confidence}/></td><td className="mono-text">{[d.box.x1,d.box.y1,d.box.x2,d.box.y2].map(n=>Math.round(n)).join(', ')}</td></tr>)}</tbody></table></div>:<EmptyState icon={Target} title="No boxes above the threshold" body="Try lowering confidence or using a closer, sharper image."/>}
        {onProceedToSegmentation && detections.length > 0 && (
          <div className="pipeline-next-step-card">
            <div className="pipeline-next-step-info">
              <span className="pipeline-step-badge">NEXT STEP</span>
              <h4>Proceed to {isCrackTarget ? 'CRACKTO-SFNet Segmentation' : 'Multi-Defect Segmentation'}</h4>
              <p>
                {isCrackTarget
                  ? 'Crack detected. Locks segmentation to CRACKTO-SFNet V4 and carries over this inspection image.'
                  : 'The detector supplies the non-crack label; FastSAM proposes a matching object mask. Review it because FastSAM is not trained for these defect classes.'}
              </p>
            </div>
            <button
              type="button"
              className="button button-primary pipeline-next-btn"
              onClick={() => onProceedToSegmentation({ file, mode: targetMode, detectionResult: data })}
            >
              <span>Proceed to {isCrackTarget ? 'Crack' : 'Detector-guided'} Segmentation</span>
              <ArrowRight size={16} />
            </button>
          </div>
        )}
      </> : <><ResultPlaceholder file={file} preview={preview} icon={ScanLine} title="Awaiting image analysis" text="Your source preview and the detector output will be shown side by side after the run." />{!file && <NotebookReference kind="image" />}</>}
    </div></div>
  </div>;
}

function SegmentationPage({projectId,notify,onInspectionComplete,pipeline,onProceedToMapping,onStepComplete}) {
  const isPipelineLocked = Boolean(pipeline?.locked && pipeline?.targetMode);
  const [file,setFile] = useState(pipeline?.file || null); const [preview,setPreview] = useState('');
  const [threshold,setThreshold] = useState(0.45); const [tta,setTta] = useState(false);
  const [loading,setLoading] = useState(false); const [inspection,setInspection] = useState(null); const [error,setError] = useState('');
  const [display,setDisplay] = useState('overlay'); const [mode,setMode] = useState(pipeline?.targetMode || 'crack');

  useEffect(() => {
    if (pipeline?.file) {
      setFile(pipeline.file);
      setMode(pipeline.targetMode || 'crack');
      setInspection(null);
    }
  }, [pipeline]);

  useEffect(()=>{if(!file){setPreview('');return;}const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url);},[file]);
  const run=async e=>{e.preventDefault();if(!file){setError('Choose an image first.');return;}setLoading(true);setError('');setInspection(null);const form=new FormData();form.append('file',file);if(projectId)form.append('project_id',String(projectId));
    const endpoint=mode==='damage'?'/api/segment/damage':'/api/segment';
    if(mode==='crack'){form.append('threshold',String(threshold));form.append('tta',String(tta));}
    try{
      const result=await api.postForm(endpoint,form);
      setInspection(result);
      if (onStepComplete) onStepComplete('segmentation');
      notify('Segmentation completed successfully.');
      onInspectionComplete();
    }catch(err){setError(err.message);notify(err.message,'error');}finally{setLoading(false);}};
  const selectMode=next=>{setMode(next);setInspection(null);setError('');setDisplay('overlay');};
  const data=inspection?.result;const imageUrl=data?(display==='overlay'?data.overlay_url:display==='mask'?data.mask_url:display==='probability'?data.probability_url:data.contours_url):'';
  const damageMetrics=data?.class_metrics||{};
  return <div className="stack-lg"><ModuleHeader eyebrow="MODULE 02 · PIXEL-LEVEL ANALYSIS" title={mode==='crack'?'Trace the crack, pixel by pixel':'Review detector-guided damage masks'} text={mode==='crack'?'CRACKTO-SFNet V4 predicts crack masks. Contour and width statistics stay in pixel units unless a physical calibration is provided.':'YOLO provides non-crack labels and FastSAM proposes matching object masks. These are not masks from a class-trained segmentation model; review overlays before interpreting measurements.'} icon={Layers3} tag={mode==='crack'?'CRACKTO-SFNET V4':'YOLO + FASTSAM PROPOSALS'} />
    <div className="workbench-grid"><form className="panel control-panel" onSubmit={run}><div className="panel-title-row"><div><h3>Segmentation setup</h3><p>Choose the source image and model target.</p></div><span className="step-badge green-step">02</span></div>
      <div className="segmented-options">
        {[['crack','Crack-trained V4'],['damage','Detector-guided other labels']].map(([id,label])=>(
          <button
            type="button"
            key={id}
            className={mode===id?'selected':''}
            disabled={isPipelineLocked && mode !== id}
            title={isPipelineLocked && mode !== id ? 'Locked by detection pipeline' : ''}
            onClick={()=>selectMode(id)}
          >
            {label}
          </button>
        ))}
      </div>
      {isPipelineLocked && (
        <span className="pipeline-locked-notice">
          <ShieldCheck size={12} /> Model locked to {mode === 'crack' ? 'Crack model (CRACKTO-SFNet V4)' : 'Multi-defect (Spalling, Rebar, Fissures)'} from Detection step
        </span>
      )}
      <UploadBox accept="image/*" file={file} setFile={setFile} kind="segment" label="Drop a structural inspection image" help="JPG, PNG, WEBP or TIFF · max 250 MB" />
      {mode==='crack'&&<><div className="settings-divider"/><div className="settings-heading"><SlidersHorizontal size={16}/><span>Mask configuration</span></div>
        <SettingField label="Mask threshold" value={threshold.toFixed(2)} help="Pixel probabilities above this value are treated as crack foreground."><input className="range-input green-range" type="range" min="0.15" max="0.75" step="0.025" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/><div className="range-ends"><span>Capture finer cracks</span><span>Stricter mask</span></div></SettingField>
        <label className="toggle-row"><span className="toggle-copy"><b>Flip test-time augmentation</b><small>Average predictions across four image orientations. Slower, may improve consistency.</small></span><input type="checkbox" checked={tta} onChange={e=>setTta(e.target.checked)}/><span className="toggle-ui"/></label></>}
      {mode==='damage'&&<p className="measurement-note"><AlertTriangle size={15}/><span>FastSAM is class-agnostic. YOLO supplies the label and the mask is selected by overlap with its box. This is not supervised class-specific segmentation; confirm the overlay against the source image.</span></p>}
      {error&&<ErrorNotice message={error}/>}
      <button className="button button-primary button-wide" type="submit" disabled={loading||!file}>{loading?<><LoaderCircle size={17} className="spin"/> Segmenting image…</>:<>Generate {mode==='crack'?'crack':'damage'} mask <ArrowRight size={16}/></>}</button>
      <p className="panel-footnote"><ShieldCheck size={13}/> {mode==='crack'?'Uses the supplied CRACKTO-SFNet V4 crack checkpoint.':'Combines the supplied YOLO detector with FastSAM-s.'}</p>
    </form>
    <div className="panel result-panel"><div className="panel-title-row"><div><h3>Segmentation result</h3><p>{inspection?`${mode==='crack'?`Threshold ${data.threshold?.toFixed(2)}`:'Class-aware mask'} · ${formatDate(inspection.completed_at)}`:'Overlay, mask, confidence, and contours will appear here.'}</p></div>{inspection&&<span className="status-badge succeeded"><Check size={12}/> Completed</span>}</div>
      {data ? <><div className="result-tabs">{[['overlay','Overlay'],['mask',mode==='crack'?'Binary mask':'Class mask'],['probability',mode==='crack'?'Probability':'Model support'],['contours','Contours']].map(([id,label])=><button className={display===id?'active':''} onClick={()=>setDisplay(id)} key={id}>{label}</button>)}</div><div className="result-image-frame segmentation-frame"><img src={publicUrl(imageUrl)} alt={`Segmentation ${display}`}/><a className="image-download" href={publicUrl(imageUrl)} download><Download size={14}/> Download image</a></div>
        {mode==='crack'?<>
          <div className="result-stat-row four-stats"><MiniMetric label="Mask coverage" value={`${data.mask_coverage_percent}%`}/><MiniMetric label="Crack pixels" value={formatNumber(data.crack_pixels)}/><MiniMetric label="Contours" value={formatNumber(data.contour_count)}/><MiniMetric label="Median width*" value={`${data.estimated_median_width_px} px`}/></div>
          <div className="measurement-grid"><div><span>Total contour length</span><b>{formatNumber(data.total_contour_length_px)} px</b></div><div><span>Largest contour</span><b>{formatNumber(data.largest_contour_length_px)} px</b></div><div><span>Largest contour area</span><b>{formatNumber(data.largest_contour_area_px2)} px²</b></div><div><span>Mean mask probability</span><b>{Number(data.mean_probability||0).toFixed(3)}</b></div></div>
        </>:<>
          <div className="result-stat-row four-stats">{Object.entries(damageMetrics).map(([name,metric])=><MiniMetric key={name} label={`${name.replace('_',' ')} area`} value={`${metric.coverage_percent}%`}/>)}</div>
          <div className="measurement-grid">{Object.entries(damageMetrics).map(([name,metric])=><div key={name}><span>{name.replace('_',' ')} · pixels / components</span><b>{formatNumber(metric.pixels)} / {formatNumber(metric.component_count)}</b></div>)}</div>
          <a className="download-link" href={publicUrl(data.class_ids_url)} download><Download size={14}/> Download numeric class-ID mask</a>
        </>}
        <div className="measurement-note"><AlertTriangle size={15}/><span>{data.measurement_note || 'Measurements are in pixel units, not physical dimensions.'}</span></div>
        {onProceedToMapping && (
          <div className="pipeline-next-step-card">
            <div className="pipeline-next-step-info">
              <span className="pipeline-step-badge">NEXT STEP</span>
              <h4>Proceed to Damage Mapping</h4>
              <p>
                {mode === 'crack'
                  ? 'Transfer crack geometry measurements to the 20×20 spatial damage grid.'
                  : 'Transfer multi-defect spalling & rebar segmentation to the 20×20 damage grid.'}
              </p>
            </div>
            <button
              type="button"
              className="button button-primary pipeline-next-btn"
              onClick={() => onProceedToMapping({
                file,
                mode: mode === 'crack' ? 'crack' : 'damage',
                segmentationInspection: inspection,
              })}
            >
              <span>{mode === 'crack' ? 'Proceed to Crack Mapping' : 'Proceed to Non-crack Mapping'}</span>
              <ArrowRight size={16} />
            </button>
          </div>
        )}
      </>:<><ResultPlaceholder file={file} preview={preview} icon={Layers3} title="Awaiting segmentation" text="Choose the crack or multi-class damage model, then generate a pixel mask."/>{!file && <NotebookReference kind="segmentation" />}</>}
    </div></div>
  </div>;
}

function VideoPage({projectId,notify,onInspectionComplete,onStepComplete}) {
  const [file,setFile] = useState(null); const [preview,setPreview] = useState(''); const [confidence,setConfidence] = useState(0.25); const [iou,setIou] = useState(0.45); const [imageSize,setImageSize] = useState('640');
  const [loading,setLoading] = useState(false); const [inspection,setInspection] = useState(null); const [error,setError] = useState('');
  useEffect(()=>{if(!file){setPreview('');return;}const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url);},[file]);
  useEffect(()=>{if(!inspection?.id||!['queued','processing'].includes(inspection.status))return;let stopped=false;const tick=async()=>{try{const next=await api.get(`/api/inspections/${inspection.id}`);if(stopped)return;setInspection(next);if(['succeeded','failed'].includes(next.status)){setLoading(false);if(next.status==='succeeded'){if(onStepComplete)onStepComplete('video');notify('Video inspection completed.');onInspectionComplete();}else{setError(next.error_message||'Video processing failed.');notify(next.error_message||'Video processing failed.','error');}}}catch(err){if(!stopped){setLoading(false);setError(err.message);}}};const timer=window.setInterval(tick,1800);tick();return()=>{stopped=true;window.clearInterval(timer);};},[inspection?.id,inspection?.status,notify,onInspectionComplete,onStepComplete]);
  const run=async e=>{e.preventDefault();if(!file){setError('Choose a video first.');return;}setLoading(true);setError('');setInspection(null);const form=new FormData();form.append('file',file);form.append('confidence',String(confidence));form.append('iou',String(iou));form.append('image_size',imageSize);if(projectId)form.append('project_id',String(projectId));
    try{const job=await api.postForm('/api/video',form);setInspection(job);notify('Video queued for frame-by-frame processing.','success');}catch(err){setLoading(false);setError(err.message);notify(err.message,'error');}};
  const data=inspection?.result||{};const progress=Number(data.progress_percent||0);const active=loading&&inspection&&['queued','processing'].includes(inspection.status);
  return <div className="stack-lg"><ModuleHeader eyebrow="MODULE 03 · TEMPORAL DETECTION" title="Inspect motion, not just a still frame" text="Process every frame with the dedicated video YOLO checkpoint. Export an annotated video, timestamped detections, and per-class regional occurrence evidence." icon={Video} tag="CV2 + YOLO" />
    <div className="workbench-grid"><form className="panel control-panel" onSubmit={run}><div className="panel-title-row"><div><h3>Video configuration</h3><p>Upload an inspection clip to analyze.</p></div><span className="step-badge orange-step">03</span></div>
      <UploadBox accept="video/*,.mp4,.mov,.avi,.mkv,.webm" file={file} setFile={setFile} kind="video" label="Drop your inspection video" help="MP4, MOV, AVI, MKV or WEBM · max 250 MB" />
      <div className="settings-divider"/><div className="settings-heading"><SlidersHorizontal size={16}/><span>Frame inference settings</span></div>
      <SettingField label="Confidence threshold" value={`${Math.round(confidence*100)}%`}><input className="range-input" type="range" min="0.05" max="0.9" step="0.05" value={confidence} onChange={e=>setConfidence(Number(e.target.value))}/></SettingField>
      <SettingField label="NMS IoU threshold" value={iou.toFixed(2)}><input className="range-input orange-range" type="range" min="0.1" max="0.8" step="0.05" value={iou} onChange={e=>setIou(Number(e.target.value))}/></SettingField>
      <SettingField label="Inference image size" value={`${imageSize} px`}><select className="select-control" value={imageSize} onChange={e=>setImageSize(e.target.value)}>{['320','416','512','640','768','960'].map(x=><option key={x}>{x}</option>)}</select></SettingField>
      {error&&<ErrorNotice message={error}/>}
      <button className="button button-primary button-wide" type="submit" disabled={loading||!file}>{active?<><LoaderCircle size={17} className="spin"/> Processing frames…</>:loading?<>Starting job…</>:<>Start video analysis <ArrowRight size={16}/></>}</button>
      <p className="panel-footnote"><Clock3 size={13}/> Long clips run as background jobs. Keep the backend terminal open.</p>
    </form>
    <div className="panel result-panel"><div className="panel-title-row"><div><h3>Video output</h3><p>{inspection?`${inspection.source_filename} · ${inspection.status}`:'Frame timeline, detections, and downloadable artifacts.'}</p></div>{inspection&&<span className={`status-badge ${inspection.status}`}>{inspection.status==='succeeded'?<Check size={12}/>:inspection.status==='failed'?<AlertTriangle size={12}/>:<LoaderCircle size={12} className="spin"/>}{inspection.status}</span>}</div>
      {active ? <div className="job-progress"><div className="job-progress-icon"><LoaderCircle size={24} className="spin"/></div><h3>Analyzing every frame</h3><p>The detector is evaluating each video frame and writing the annotated output.</p><div className="progress-track"><span style={{width:`${progress||3}%`}}/></div><div className="progress-labels"><span>{data.processed_frames?`${formatNumber(data.processed_frames)} frames processed`:'Preparing video reader'}</span><b>{progress?`${progress}%`:'In progress'}</b></div><div className="job-note"><Clock3 size={14}/> Processing time depends on video duration, frame dimensions, and available GPU.</div></div>
      : inspection?.status==='succeeded' ? <><div className="video-output-frame"><video controls playsInline src={publicUrl(data.annotated_video_url)} /><a className="image-download" href={publicUrl(data.annotated_video_url)} download><Download size={14}/> Download annotated video</a></div>
          <div className="result-stat-row"><MiniMetric label="Total detections" value={formatNumber(data.total_detections)}/><MiniMetric label="Frames processed" value={formatNumber(data.processed_frame_count)}/><MiniMetric label="With detections" value={formatNumber(data.frames_with_detections)}/></div>
          <div className="measurement-grid"><div><span>Duration</span><b>{Number(data.duration_seconds||0).toFixed(2)} sec</b></div><div><span>Video FPS</span><b>{Number(data.fps||0).toFixed(2)}</b></div><div><span>Max confidence</span><b>{Math.round((data.maximum_confidence||0)*100)}%</b></div><div><span>Processing time</span><b>{Number(data.processing_time_seconds||0).toFixed(1)} sec</b></div></div>
          <div className="result-subheading"><b>Class detections</b><a className="text-button" href={publicUrl(data.detections_csv_url)} download><Download size={14}/> Export CSV</a></div>
          <div className="class-count-pills">{Object.entries(data.class_counts||{}).map(([name,count])=><span className="count-pill" key={name}><b>{name}</b><strong>{formatNumber(count)}</strong></span>)}</div>
          <RegionalEvidenceMap evidence={data.regional_evidence} video/>
          <div className="download-row"><a href={publicUrl(data.summary_json_url)} download className="download-link"><Database size={15}/> Summary JSON <Download size={13}/></a><a href={publicUrl(data.detections_csv_url)} download className="download-link"><HardDrive size={15}/> Frame detections CSV <Download size={13}/></a></div>
      </> : inspection?.status==='failed' ? <ErrorNotice message={inspection.error_message||error||'The video job failed.'} large/> : <><ResultPlaceholder file={file} preview={preview} icon={Video} title="Awaiting video analysis" text="Upload a clip. CrackVision will annotate it and preserve individual frame detections with timestamps."/>{!file && <NotebookReference kind="video" />}</>}
    </div></div>
  </div>;
}

function RegionalEvidenceMap({evidence,video=false}) {
  const classes=Object.entries(evidence?.classes||{});
  const [selectedClass,setSelectedClass]=useState('');
  const active=classes.find(([name])=>name===selectedClass)||classes[0];
  if(!evidence)return null;
  const counts=active?.[1]?.grid_counts||[];
  const displayCounts=video?active?.[1]?.occurrences_per_100_frames||counts:counts;
  const maxCount=Math.max(0,...displayCounts.flat());
  const perHundred=video?(active?.[1]?.detection_count||0)*100/Math.max(evidence.frames_analyzed||1,1):null;
  return <section className="regional-evidence">
    <div className="result-subheading"><b>Regional detection evidence</b><span>{video?'20 × 20 · accumulated across frames':'20 × 20 image grid'}</span></div>
    <p className="regional-evidence-note">{evidence.limitation}</p>
    {!classes.length?<p className="regional-evidence-empty">No detections to map at this confidence threshold.</p>:<>
      <div className="regional-evidence-tabs" role="tablist" aria-label="Detection class evidence">
        {classes.map(([name,item])=><button type="button" role="tab" aria-selected={active?.[0]===name} className={active?.[0]===name?'selected':''} key={name} onClick={()=>setSelectedClass(name)}>
          {name}<span>{formatNumber(item.detection_count)}</span>
        </button>)}
      </div>
      <div className="regional-evidence-grid" role="img" aria-label={`${active?.[0]} detections grouped by bounding-box center across a 20 by 20 grid`} style={{gridTemplateColumns:`repeat(${evidence.grid_cols||20},1fr)`}}>
        {displayCounts.flatMap((row,rowIndex)=>row.map((value,colIndex)=>{
          const count=counts[rowIndex]?.[colIndex]||0;
          const opacity=value?0.12+0.78*value/maxCount:0;
          const amount=video?`${value} per 100 frames`:`${count} detection${count===1?'':'s'}`;
          return <span key={`${rowIndex}-${colIndex}`} title={`Row ${rowIndex+1}, column ${colIndex+1}: ${amount}`} style={{backgroundColor:value?`rgba(194, 75, 58, ${opacity})`:'#f2f4f6'}}/>;
        }))}
      </div>
      <div className="regional-evidence-legend"><span>Less</span><i/><span>More {video?'per 100 frames':'detections'}</span><b>{formatNumber(active?.[1]?.detection_count)} total{video?` · ${perHundred.toFixed(2)}/100 frames`:''}</b></div>
    </>}
  </section>;
}


function ConfidenceBar({value}) { return <div className="confidence-cell"><div className="confidence-track"><span style={{width:`${Math.max(0,Math.min(100,value*100))}%`}}/></div><b>{Math.round(value*100)}%</b></div>; }


function NotebookReference({kind}) {
  if (kind === 'image') return <details className="notebook-reference"><summary><Sparkles size={14}/> View saved detector output from the attached notebook</summary><img src="/reference/detector_result_1.png" alt="Saved detector validation montage from supplied notebook"/><a className="text-button" href="/reference/detector_result_2.png" target="_blank" rel="noreferrer">Open second validation montage <ArrowRight size={13}/></a><p>These are reference outputs created in the attached notebook, not a prediction for the file selected above.</p></details>;
  if (kind === 'segmentation') return <details className="notebook-reference"><summary><Sparkles size={14}/> Browse saved V4 validation diagnostics</summary><div className="reference-image-grid"><a href="/reference/segmentation/threshold_curve.png" target="_blank" rel="noreferrer"><img src="/reference/segmentation/threshold_curve.png" alt="V4 threshold curve"/><span>Threshold curve</span></a><a href="/reference/segmentation/width_bin_iou.png" target="_blank" rel="noreferrer"><img src="/reference/segmentation/width_bin_iou.png" alt="V4 IoU by crack width"/><span>IoU by crack width</span></a></div><div className="reference-links"><a href="/reference/segmentation/diagnostic_summary.json" download>Diagnostic summary JSON <Download size={13}/></a><a href="/reference/segmentation/threshold_curve.csv" download>Threshold metrics CSV <Download size={13}/></a><a href="/reference/segmentation/v3_v4_global_comparison.csv" download>V3 / V4 comparison <Download size={13}/></a><a href="/reference/segmentation/topology_analysis.csv" download>Topology analysis <Download size={13}/></a><a href="/reference/segmentation/worst_case_contact_sheet.png" target="_blank" rel="noreferrer">Worst-case examples <ArrowRight size={13}/></a><a href="/reference/segmentation/boundary_analysis.csv" download>Boundary analysis <Download size={13}/></a></div><p>These are previously saved validation diagnostics from the attached archive. New images are segmented live using the V4 best checkpoint.</p></details>;
  return <details className="notebook-reference"><summary><Sparkles size={14}/> Play the saved annotated video from your notebook</summary><video controls preload="metadata" src="/reference/video_pipeline/reference_detected_video.mp4"/><div className="reference-links"><a href="/reference/video_pipeline/frame_detections.csv" download>Frame detections CSV <Download size={13}/></a><a href="/reference/video_pipeline/video_summary.json" download>Pipeline summary JSON <Download size={13}/></a></div><p>Saved reference run · 306 frames · 418 detector boxes. Recorded class counts:</p><div className="class-count-pills"><span className="count-pill"><b>crack</b><strong>366</strong></span><span className="count-pill"><b>concrete spalling</b><strong>12</strong></span><span className="count-pill"><b>rebar</b><strong>40</strong></span></div><p>These are detector-box counts, not pixel masks or a segmentation validation set. New uploads are processed separately.</p></details>;
}

function ProjectsPage({projects,notify,onRefresh}) {
  const [formOpen,setFormOpen] = useState(false); const [saving,setSaving]=useState(false); const [error,setError]=useState('');
  const [name,setName]=useState(''); const [location,setLocation]=useState(''); const [type,setType]=useState('Building'); const [description,setDescription]=useState('');
  const create=async e=>{e.preventDefault();setSaving(true);setError('');try{await api.postJson('/api/projects',{name,site_location:location,structure_type:type,description});setName('');setLocation('');setType('Building');setDescription('');setFormOpen(false);notify('Project created.');onRefresh();}catch(err){setError(err.message);notify(err.message,'error');}finally{setSaving(false);}};
  const remove=async project=>{if(!window.confirm(`Delete “${project.name}”? Projects with inspection history cannot be deleted.`))return;try{await api.delete(`/api/projects/${project.id}`);notify('Project deleted.');onRefresh();}catch(err){notify(err.message,'error');}};
  return <div className="stack-lg"><div className="projects-toolbar"><div className="toolbar-summary"><div className="toolbar-icon"><Building2 size={20}/></div><div><b>{projects.length} projects configured</b><span>Each inspection is stored against the active site.</span></div></div><button className="button button-primary" onClick={()=>setFormOpen(v=>!v)}>{formOpen?<X size={16}/>:<Plus size={16}/>} {formOpen?'Close form':'New project'}</button></div>
    {formOpen&&<form className="panel project-form" onSubmit={create}><div className="panel-title-row"><div><h3>Create a project</h3><p>Use a project for each structure, building, or inspection site.</p></div><span className="step-badge">NEW</span></div><div className="form-grid"><label className="input-field"><span>Project name *</span><input required minLength={2} maxLength={120} value={name} onChange={e=>setName(e.target.value)} placeholder="e.g. North Block — Level 2"/></label><label className="input-field"><span>Site / location</span><input value={location} onChange={e=>setLocation(e.target.value)} placeholder="e.g. Campus, Block B"/></label><label className="input-field"><span>Structure type</span><select value={type} onChange={e=>setType(e.target.value)}>{['Building','Bridge','Road / pavement','Wall','Slab','Pillar / column','Dam','Other'].map(x=><option key={x}>{x}</option>)}</select></label><label className="input-field span-two"><span>Description</span><textarea value={description} onChange={e=>setDescription(e.target.value)} rows="3" placeholder="Optional notes about the structure or inspection context."/></label></div>{error&&<ErrorNotice message={error}/>}<div className="form-actions"><button className="button button-primary" disabled={saving}>{saving?<LoaderCircle className="spin" size={16}/>:<Check size={16}/>} Save project</button></div></form>}
    <section className="project-grid">{projects.map((project,i)=><article className="project-card" key={project.id}><div className="project-card-visual"><span className={`project-building building-${i%3}`}><Building2 size={32}/></span><span className="project-type">{project.structure_type}</span><button className="project-more" onClick={()=>remove(project)} aria-label={`Delete ${project.name}`}><MoreHorizontal size={18}/></button></div><div className="project-card-body"><div className="project-title-line"><h3>{project.name}</h3><span className="project-id">CV-{String(project.id).padStart(3,'0')}</span></div><p className="project-location"><Building2 size={14}/>{project.site_location||'Location not specified'}</p>{project.description&&<p className="project-description">{project.description}</p>}<div className="project-card-footer"><span><Activity size={14}/>{project.inspection_count} inspections</span><span>Added {formatDate(project.created_at)}</span></div></div></article>)}
      <button className="project-add-card" onClick={()=>setFormOpen(true)}><span><Plus size={22}/></span><b>Add a new project</b><small>Keep inspections organized by site</small></button>
    </section>
  </div>;
}

function HistoryPage({projects,onRefresh,notify}) {
  const [items,setItems]=useState([]); const [loading,setLoading]=useState(true); const [module,setModule]=useState('all');const [status,setStatus]=useState('all');const [query,setQuery]=useState('');const [active,setActive]=useState(null);
  const load=useCallback(async()=>{setLoading(true);try{const params=new URLSearchParams();if(module!=='all')params.set('module',module);if(status!=='all')params.set('status',status);if(query.trim())params.set('q',query.trim());setItems(await api.get(`/api/inspections?${params.toString()}`));}catch(err){notify(err.message,'error');}finally{setLoading(false);}},[module,status,query,notify]);
  useEffect(()=>{load();},[load]);
  const remove=async item=>{if(!window.confirm(`Delete inspection for “${item.source_filename}”? Its stored artifacts will also be removed.`))return;try{await api.delete(`/api/inspections/${item.id}`);notify('Inspection deleted.');setActive(null);load();onRefresh();}catch(err){notify(err.message,'error');}};
  return <div className="stack-lg"><div className="history-toolbar"><div className="search-box"><Search size={16}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search source filename…"/></div><label className="filter-select"><Filter size={15}/><select value={module} onChange={e=>setModule(e.target.value)}><option value="all">All modules</option><option value="detection">Crack detection</option><option value="segmentation">Segmentation</option><option value="video">Video inspection</option><option value="mapping">Damage mapping</option></select></label><label className="filter-select"><select value={status} onChange={e=>setStatus(e.target.value)}><option value="all">Any status</option><option value="succeeded">Succeeded</option><option value="processing">Processing</option><option value="queued">Queued</option><option value="failed">Failed</option></select></label><button className="button button-secondary" onClick={load}><Activity size={15}/> Refresh</button></div>
    <div className="panel history-panel"><div className="panel-heading"><div><h3>Inspection ledger</h3><p>{items.length} records matching your filters.</p></div><span className="subtle-tag">SAVED RUNS</span></div>
      {loading?<LoadingRows/>:items.length?<div className="history-table-wrap"><table className="data-table history-table"><thead><tr><th>Source file</th><th>Module</th><th>Project</th><th>Created</th><th>Status</th><th>Action</th></tr></thead><tbody>{items.map(item=><tr key={item.id}><td><div className="history-file"><span className={`recent-module-icon ${item.module}`}>{item.module==='video'?<Video size={15}/>:item.module==='segmentation'?<Layers3 size={15}/>:item.module==='mapping'?<MapPinned size={15}/>:<ScanLine size={15}/>}</span><span><b>{item.source_filename}</b><small>#{item.id.slice(0,8)}</small></span></div></td><td><span className={`module-text ${item.module}`}>{moduleLabel(item.module)}</span></td><td>{item.project_name||'—'}</td><td>{formatDate(item.created_at)}</td><td><span className={`status-badge ${item.status}`}>{item.status}</span></td><td><button className="table-action" onClick={()=>setActive(item)}>View <ArrowRight size={13}/></button></td></tr>)}</tbody></table></div>:<EmptyState icon={History} title="No matching inspections" body="Completed and failed runs will be saved here with their generated artifacts."/>}
    </div>
    {active&&<InspectionModal item={active} onClose={()=>setActive(null)} onDelete={()=>remove(active)}/>}
  </div>;
}

function InspectionModal({item,onClose,onDelete}) {
  const [full,setFull]=useState(item);
  useEffect(()=>{api.get(`/api/inspections/${item.id}`).then(setFull).catch(()=>setFull(item));},[item.id]);
  const result=full.result||{};
  return <div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}><section className="modal-card"><header className="modal-header"><div><span className="eyebrow dark-eyebrow">INSPECTION DETAIL</span><h2>{full.source_filename}</h2><p>{full.project_name||'Unassigned project'} · {formatDate(full.created_at)}</p></div><button className="icon-button" onClick={onClose}><X size={18}/></button></header><div className="modal-body"><div className="modal-status-row"><span className={`status-badge ${full.status}`}>{full.status}</span><span className={`module-text ${full.module}`}>{moduleLabel(full.module)}</span></div>{full.error_message&&<ErrorNotice message={full.error_message}/>}<div className="modal-result-grid">{Object.entries(summarizeResult(full.module,result)).map(([k,v])=><div className="modal-result-item" key={k}><span>{k}</span><b>{v}</b></div>)}</div>{result.annotated_image_url&&<img className="modal-result-image" src={publicUrl(result.annotated_image_url)} alt="Inspection result"/>}{result.overlay_url&&<img className="modal-result-image" src={publicUrl(result.overlay_url)} alt="Result overlay"/>}{result.annotated_video_url&&<video className="modal-result-video" controls src={publicUrl(result.annotated_video_url)}/>}<details className="json-details"><summary>Raw result payload</summary><pre>{JSON.stringify(result,null,2)}</pre></details></div><footer className="modal-footer"><button className="button button-danger-ghost" onClick={onDelete}><Trash2 size={15}/> Delete inspection</button><button className="button button-secondary" onClick={onClose}>Close</button></footer></section></div>;
}

function summarizeResult(module,result) {
  if(module==='detection')return {'Detections':result.detection_count??0,'Classes':Object.keys(result.class_counts||{}).length,'Max confidence':`${Math.round((result.maximum_confidence||0)*100)}%`};
  if(module==='segmentation')return {'Mask coverage':`${result.mask_coverage_percent??0}%`,'Contours':result.contour_count??0,'Total contour':`${result.total_contour_length_px??0} px`};
  if(module==='mapping'){
    if(result.kind==='video')return {'Sampled frames':result.processed_frame_count??0,'Frames with cracks':result.frames_with_cracks??0,'Peak index':Number(result.peak_max_priority||0).toFixed(2)};
    return {'Hotspot cells':`${result.hotspot_count??0} / ${result.total_cells??400}`,'Crack area':`${(100*(result.metrics?.crack_area_ratio||0)).toFixed(2)}%`,'Max index':Number(result.max_priority||0).toFixed(2)};
  }
  return {'Detections':result.total_detections??0,'Frames':result.processed_frame_count??0,'Duration':`${Number(result.duration_seconds||0).toFixed(2)} sec`};
}
function moduleLabel(module){return module==='detection'?'Image detection':module==='segmentation'?'Segmentation':module==='mapping'?'Damage mapping':'Video pipeline';}
function LoadingRows(){return <div className="loading-rows"><LoaderCircle className="spin" size={22}/><span>Loading inspection history…</span></div>;}

function SystemPage({health,onRefresh}) {
  const modules=health?.modules||{};const rows=[
    {id:'detection',title:'Crack detection',subtitle:'Separate image-detection YOLO checkpoint',icon:ScanLine,info:modules.detection,detail:modules.detection?.classes?`${Object.keys(modules.detection.classes).length} classes`: 'Lazy loaded on the first inference'},
    {id:'segmentation',title:'Segmentation',subtitle:'CRACKTO-SFNet V4 · matching runtime and best checkpoint',icon:Layers3,info:modules.segmentation,detail:modules.segmentation?.checkpoint?.best_threshold?`Best threshold ${modules.segmentation.checkpoint.best_threshold}`:'Best threshold stored in checkpoint'},
    {id:'damage_segmentation',title:'Detector-guided masks',subtitle:'YOLO labels + class-agnostic FastSAM proposals',icon:Layers3,info:modules.damage_segmentation,detail:'Non-crack masks are proposals, not class-trained segmentation'},
    {id:'video',title:'Video pipeline',subtitle:'OpenCV frame reader + dedicated video YOLO checkpoint',icon:Video,info:modules.video,detail:'Frame timestamp CSV + annotated MP4 + summary JSON'},
    {id:'mapping',title:'Damage mapping',subtitle:'V4 probability maps + skeleton, width proxy, 20×20 grid and priority index',icon:MapPinned,info:modules.mapping,detail:modules.mapping?.normalization_reference_found?'Validation-derived normalization reference found':'Normalization reference missing (embedded fallback used)'},
    {id:'damage_mapping',title:'All-label mapping',subtitle:'Crack V4 masks + detector-guided FastSAM masks',icon:MapPinned,info:modules.damage_mapping,detail:'Image grid is exploratory, not a severity assessment'},
  ];
  return <div className="stack-lg"><div className="system-banner"><div className="system-banner-icon"><Activity size={22}/></div><div><h2>{health?.status==='ok'?'The API is responding':'Backend connection needs attention'}</h2><p>{health?.status==='ok'?'The application can reach the API. Model files are checked separately below.':'Start the FastAPI backend, then use Refresh status to retry the connection.'}</p></div><button className="button button-secondary" onClick={onRefresh}><Activity size={15}/> Refresh status</button></div>
    <div className="panel system-panel"><div className="panel-heading"><div><h3>Module readiness</h3><p>Model files are loaded only when their module is first used.</p></div><span className={`status-badge ${health?.status==='ok'?'succeeded':'failed'}`}>{health?.status||'unknown'}</span></div>
      {rows.map(row=>{const Icon=row.icon;const info=row.info||{};const ok=!!info.available;return <div className="system-module-row" key={row.id}><div className={`system-module-icon ${row.id}`}><Icon size={20}/></div><div className="system-module-main"><b>{row.title}</b><span>{row.subtitle}</span><small>{row.detail}</small>{info.load_error&&<p className="system-error">{info.load_error}</p>}</div><div className="system-module-status"><span className={`status-pip ${ok?'good':'warn'}`}/><b>{ok?'Artifact found':'Unavailable'}</b><small>{info.loaded?'In memory':'Not loaded'}</small></div></div>;})}
    </div>
    <div className="system-two-col"><div className="panel system-detail-card"><div className="panel-heading"><div><h3>Runtime environment</h3><p>Detected from the live backend.</p></div><Command size={18} className="muted"/></div><KeyValue label="API status" value={health?.status||'unknown'}/><KeyValue label="Database" value={health?.database?'Connected':'Unavailable'}/><KeyValue label="PyTorch" value={health?.torch?.version|| (health?.torch?.installed?'Installed':'Not found')}/><KeyValue label="Inference device" value={health?.torch?.device||'—'}/><KeyValue label="CUDA available" value={health?.torch?.cuda_available?'Yes':'No / CPU mode'}/></div>
      <div className="panel system-detail-card"><div className="panel-heading"><div><h3>Model provenance</h3><p>Boundaries of this build</p></div><ShieldCheck size={18} className="muted"/></div><ul className="provenance-list"><li><CheckCircle2 size={15}/><span><b>Image YOLO detector</b> is kept separate from the video-specific checkpoint.</span></li><li><CheckCircle2 size={15}/><span><b>CRACKTO-SFNet V4</b> is loaded from the attached V4 best checkpoint with strict state-dict matching.</span></li><li><CheckCircle2 size={15}/><span><b>Non-crack masks</b> are class-agnostic FastSAM proposals matched to YOLO boxes; this build has no supervised multi-class segmentation checkpoint.</span></li><li><CheckCircle2 size={15}/><span><b>Video inference</b> loads video_detector_best_reconstructed.pt from your attached results archive and records per-frame timestamps and boxes.</span></li><li><CheckCircle2 size={15}/><span><b>Damage mapping</b> uses image-space masks and a 20×20 grid; class-agnostic proposals need visual review.</span></li><li><AlertTriangle size={15}/><span>The priority index is exploratory — not a structural-safety score.</span></li><li><AlertTriangle size={15}/><span>Segmentation dimensions are pixel-based. Real-world sizes need scale calibration.</span></li></ul></div></div>
  </div>;
}
function KeyValue({label,value}){return <div className="key-value"><span>{label}</span><b>{value}</b></div>;}
