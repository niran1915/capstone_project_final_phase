import React, { useState } from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, CloudUpload, FileImage, FileVideo } from 'lucide-react';

// Presentational components shared by every module page (moved out of App.jsx unchanged).

export function EmptyState({icon:Icon,title,body,action,onAction}) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={23}/></div><b>{title}</b><p>{body}</p>{action && <button className="button button-secondary button-small" onClick={onAction}>{action}<ArrowRight size={14}/></button>}</div>;
}

export function ModuleHeader({eyebrow, title, text, icon:Icon, tag}) {
  return <div className="module-hero"><div className="module-hero-icon"><Icon size={23}/></div><div className="module-hero-text"><div className="eyebrow dark-eyebrow">{eyebrow}</div><h2>{title}</h2><p>{text}</p></div>{tag && <span className="subtle-tag module-hero-tag">{tag}</span>}</div>;
}

export function UploadBox({accept, file, setFile, label, help, kind='image'}) {
  const [dragging,setDragging] = useState(false);
  const inputId = `upload-${kind}`;
  const choose = (files) => { const item = files?.[0]; if (item) setFile(item); };
  return <div className={`upload-zone ${dragging?'dragging':''} ${file?'has-file':''}`} onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);choose(e.dataTransfer.files);}}>
    <input id={inputId} type="file" accept={accept} onChange={e=>choose(e.target.files)} />
    {file ? <><div className="upload-file-icon">{kind==='video'?<FileVideo size={23}/>:<FileImage size={23}/>}</div><div className="upload-file-name">{file.name}</div><div className="upload-file-meta">{(file.size/1024/1024).toFixed(2)} MB · selected</div><div className="upload-file-actions"><label htmlFor={inputId} className="text-button">Replace file</label><button type="button" className="text-button danger-text" onClick={()=>setFile(null)}>Remove</button></div></> : <><div className="upload-icon"><CloudUpload size={24}/></div><b>{label || 'Drop a file here, or browse'}</b><p>{help || 'Upload an image to start analyzing.'}</p><label className="button button-secondary button-small" htmlFor={inputId}>Browse files</label></>}
  </div>;
}

export function SettingField({label,value,children,help}) {
  return <div className="setting-field"><div className="setting-label-row"><label>{label}</label><span>{value}</span></div>{children}{help&&<small>{help}</small>}</div>;
}

export function ResultPlaceholder({file,preview,icon:Icon,title,text}) {
  return <div className="result-placeholder">{file&&preview?<div className="source-preview"><img src={preview} alt="Selected source preview"/><span className="source-file-name">{file.name}</span></div>:<div className="placeholder-illustration"><div className="placeholder-grid"/><div className="placeholder-square"><Icon size={36}/></div><span className="placeholder-corner corner-a"/><span className="placeholder-corner corner-b"/></div>}<h3>{title}</h3><p>{text}</p><div className="placeholder-checks"><span><CheckCircle2 size={14}/> Model-backed inference</span><span><CheckCircle2 size={14}/> Saved to inspection history</span></div></div>;
}

export function MiniMetric({label,value}) { return <div className="mini-metric"><span>{label}</span><b>{value}</b></div>; }

export function ErrorNotice({message,large=false}) { return <div className={`error-notice ${large?'large':''}`}><AlertTriangle size={16}/><span>{message}</span></div>; }
