import React, { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, ArrowRight, Check, Cloud, Database, FileArchive, FileCode2, FileJson2, FileSpreadsheet, FileText, FolderOpen, GitBranch, Globe2, Search, X } from 'lucide-react';
import { request } from '../api';

const API_BASE = window.location.port === '5173' ? 'http://localhost:5000' : '';

const SOURCES = {
  Files: [
    { id: 'csv', name: 'Text / CSV', detail: 'CSV, TSV, and delimited text', icon: FileText, kind: 'file', accept: '.csv,.tsv,.txt' },
    { id: 'excel', name: 'Excel workbook', detail: 'Excel 97â€“365 workbooks', icon: FileSpreadsheet, kind: 'file', accept: '.xlsx,.xls' },
    { id: 'json', name: 'JSON', detail: 'JSON arrays, records, and lines', icon: FileJson2, kind: 'file', accept: '.json,.jsonl' },
    { id: 'xml', name: 'XML', detail: 'Tabular XML documents', icon: FileCode2, kind: 'file', accept: '.xml' },
    { id: 'pdf', name: 'PDF tables & text', detail: 'Extract page text into rows', icon: FileText, kind: 'file', accept: '.pdf' },
    { id: 'parquet', name: 'Parquet', detail: 'Columnar analytics files', icon: FileArchive, kind: 'file', accept: '.parquet' },
    { id: 'sharepoint', name: 'SharePoint folders', detail: 'Microsoft OAuth setup required', icon: FolderOpen, kind: 'unavailable' },
  ],
  Databases: [
    { id: 'mysql', name: 'MySQL', detail: 'MySQL and MariaDB', icon: Database, kind: 'database', port: '3306' },
    { id: 'postgresql', name: 'PostgreSQL', detail: 'PostgreSQL-compatible servers', icon: Database, kind: 'database', port: '5432' },
    { id: 'sqlserver', name: 'SQL Server', detail: 'SQL Server via ODBC Driver 18', icon: Database, kind: 'database', port: '1433' },
    { id: 'azuresql', name: 'Azure SQL Database', detail: 'Azure SQL via ODBC Driver 18', icon: Cloud, kind: 'database', port: '1433' },
    { id: 'oracle', name: 'Oracle Database', detail: 'Oracle service name or SID', icon: Database, kind: 'database', port: '1521' },
    { id: 'ibmdb2', name: 'IBM Db2', detail: 'Db2 LUW database', icon: Database, kind: 'database', port: '50000' },
    { id: 'databricks', name: 'Databricks SQL', detail: 'SQL Warehouse and catalog', icon: Cloud, kind: 'database' },
    { id: 'odbc', name: 'SQL Server ODBC', detail: 'SQL Server via installed ODBC driver', icon: Database, kind: 'database' },
  ],
  'Online services': [
    { id: 'odata', name: 'OData feed', detail: 'OData JSON services', icon: Globe2, kind: 'odata' },
    { id: 'github', name: 'GitHub', detail: 'Repositories, issues, pulls, commits', icon: GitBranch, kind: 'github' },
    { id: 'salesforce', name: 'Salesforce', detail: 'OAuth connector not configured', icon: Cloud, kind: 'unavailable' },
    { id: 'dynamics', name: 'Dynamics 365', detail: 'OAuth connector not configured', icon: Cloud, kind: 'unavailable' },
    { id: 'google-analytics', name: 'Google Analytics', detail: 'OAuth connector not configured', icon: Cloud, kind: 'unavailable' },
    { id: 'mailchimp', name: 'Mailchimp', detail: 'OAuth connector not configured', icon: Cloud, kind: 'unavailable' },
    { id: 'exchange', name: 'Microsoft Exchange', detail: 'Microsoft Graph OAuth required', icon: Cloud, kind: 'unavailable' },
  ],
  Other: [
    { id: 'web', name: 'Web page tables', detail: 'Import tables from a public HTTPS page', icon: Globe2, kind: 'web' },
    { id: 'python', name: 'Python / R scripts', detail: 'Script execution is not enabled', icon: FileCode2, kind: 'unavailable' },
    { id: 'oledb', name: 'OLE DB', detail: 'Provider integration not configured', icon: Database, kind: 'unavailable' },
  ]
};

const INITIAL_CONFIG = {
  host: '', port: '', username: '', password: '', database: '', token: '',
  http_path: '', catalog: '', schema: '', query: 'SELECT * FROM table_name', driver: 'ODBC Driver 18 for SQL Server',
  url: '', repository: '', resource: 'issues'
};

export default function DataSourceDialog({ open, onClose, onConnected, onImportStart, onImportEnd }) {
  const [category, setCategory] = useState('Files');
  const [selected, setSelected] = useState(SOURCES.Files[0]);
  const [search, setSearch] = useState('');
  const [config, setConfig] = useState(INITIAL_CONFIG);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = event => { if (event.key === 'Escape' && !busy) onClose(); };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, busy, onClose]);

  const entries = useMemo(() => (SOURCES[category] || []).filter(source =>
    `${source.name} ${source.detail}`.toLowerCase().includes(search.toLowerCase())
  ), [category, search]);

  if (!open) return null;

  const chooseCategory = name => {
    setCategory(name);
    setSearch('');
    setSelected(SOURCES[name][0]);
    setError('');
  };
  const chooseSource = source => {
    setSelected(source);
    setConfig(current => ({ ...INITIAL_CONFIG, port: source.port || '', query: current.query }));
    setError(source.kind === 'unavailable' ? source.detail : '');
  };
  const update = (field, value) => setConfig(current => ({ ...current, [field]: value }));

  const importFile = async event => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setBusy(true); setError(''); onImportStart?.();
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not import this file');
      onConnected(result); onClose();
    } catch (err) { setError(err.message); }
    finally { setBusy(false); onImportEnd?.(); }
  };

  const submit = async event => {
    event.preventDefault();
    setBusy(true); setError(''); onImportStart?.();
    try {
      let endpoint = '/api/connect/database';
      let payload = { provider: selected.id, ...config };
      if (selected.kind === 'odata') endpoint = '/api/connect/odata';
      if (selected.kind === 'github') endpoint = '/api/connect/github';
      if (selected.kind === 'web') endpoint = '/api/connect/web';
      const info = await request(endpoint, { method: 'POST', body: JSON.stringify(payload) });
      onConnected(info); onClose();
    } catch (err) { setError(err.message); }
    finally { setBusy(false); onImportEnd?.(); }
  };

  const SelectedIcon = selected.icon;
  const sqlDriverNote = selected.kind === 'database' && selected.id !== 'databricks'
    ? 'Connect mode: Import snapshot into Vizifyâ€™s local SQLite workspace. DirectQuery is not available yet.'
    : '';

  return <div className="source-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget && !busy) onClose(); }}>
    <section className="source-dialog" role="dialog" aria-modal="true" aria-labelledby="source-dialog-title">
      <header className="source-dialog-header">
        <div><span className="eyebrow">DATA WORKSPACE</span><h2 id="source-dialog-title">Connect your data</h2><p>Choose a source to import into Vizify.</p></div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close" disabled={busy}><X size={19} /></button>
      </header>
      <div className="source-dialog-body">
        <nav className="source-categories" aria-label="Source categories">
          {Object.keys(SOURCES).map(name => <button type="button" key={name} className={category === name ? 'active' : ''} onClick={() => chooseCategory(name)}>{name}<span>{SOURCES[name].length}</span></button>)}
        </nav>
        <div className="source-list-panel">
          <label className="source-search"><Search size={16} /><input value={search} onChange={event => setSearch(event.target.value)} placeholder="Search data sources" /></label>
          <div className="source-cards">
            {entries.map(source => {
              const Icon = source.icon;
              return <button type="button" key={source.id} className={`source-card ${selected.id === source.id ? 'selected' : ''} ${source.kind === 'unavailable' ? 'unavailable' : ''}`} onClick={() => chooseSource(source)}>
                <span className="source-card-icon"><Icon size={19} /></span>
                <span className="source-card-copy"><strong>{source.name}</strong><small>{source.detail}</small></span>
                {source.kind === 'unavailable' ? <span className="source-badge">Setup needed</span> : selected.id === source.id ? <Check size={16} className="source-check" /> : <ArrowRight size={15} className="source-arrow" />}
              </button>;
            })}
            {entries.length === 0 && <div className="source-empty">No sources match your search.</div>}
          </div>
        </div>
        <div className="source-config-panel">
          <div className="source-config-title"><span className="source-card-icon large"><SelectedIcon size={20} /></span><div><span className="eyebrow">SELECTED SOURCE</span><h3>{selected.name}</h3></div></div>

          {selected.kind === 'file' && <div className="source-form-block">
            <p>Select a local or network file. Vizify detects its columns and imports it to the local workspace.</p>
            <label className="file-dropzone"><input type="file" accept={selected.accept} onChange={importFile} disabled={busy} /><FolderOpen size={23} /><strong>{busy ? 'Importingâ€¦' : 'Browse files'}</strong><small>Accepted: {selected.accept.replaceAll('.', '').replaceAll(',', ', ')}</small></label>
          </div>}

          {selected.kind === 'database' && <form className="source-form-block" onSubmit={submit}>
            {selected.id === 'databricks' ? <>
              <label>Workspace hostname<input required value={config.host} onChange={event => update('host', event.target.value)} placeholder="workspace.cloud.databricks.com" /></label>
              <label>Access token<input required type="password" value={config.token} onChange={event => update('token', event.target.value)} placeholder="Databricks personal access token" /></label>
              <label>SQL warehouse HTTP path<input required value={config.http_path} onChange={event => update('http_path', event.target.value)} placeholder="/sql/1.0/warehouses/â€¦" /></label>
              <div className="form-two-col"><label>Catalog<input value={config.catalog} onChange={event => update('catalog', event.target.value)} /></label><label>Schema<input value={config.schema} onChange={event => update('schema', event.target.value)} /></label></div>
            </> : <>
              {selected.id === 'odbc' ? <label>ODBC connection string<input required value={config.host} onChange={event => update('host', event.target.value)} placeholder="DSN=MyWarehouse" /></label> : <>
                <div className="form-two-col"><label>Server / host<input required value={config.host} onChange={event => update('host', event.target.value)} placeholder={selected.id === 'azuresql' ? 'server.database.windows.net' : 'localhost'} /></label><label>Port<input value={config.port} onChange={event => update('port', event.target.value)} placeholder="Default" /></label></div>
                <div className="form-two-col"><label>Database / service<input required value={config.database} onChange={event => update('database', event.target.value)} placeholder="Database name" /></label><label>Username<input value={config.username} onChange={event => update('username', event.target.value)} placeholder="Username" /></label></div>
                <label>Password<input type="password" value={config.password} onChange={event => update('password', event.target.value)} placeholder="Password" /></label>
              </>}
              {['sqlserver', 'azuresql', 'odbc'].includes(selected.id) && <label>ODBC driver<input value={config.driver} onChange={event => update('driver', event.target.value)} /></label>}
            </>}
            <label>SQL query<textarea rows="4" required value={config.query} onChange={event => update('query', event.target.value)} spellCheck="false" /></label>
            <p className="source-mode-note">{sqlDriverNote}</p>
            {error && <p className="source-error">{error}</p>}
            <button className="btn btn-primary source-submit" disabled={busy}>{busy ? 'Connectingâ€¦' : 'Connect and import'}<ArrowRight size={16} /></button>
            <small className="source-privacy-note">Credentials are used for this import and are not saved.</small>
          </form>}

          {selected.kind === 'odata' && <form className="source-form-block" onSubmit={submit}>
            <p>Import an OData JSON collection. Use an HTTPS service endpoint that returns an entity collection.</p>
            <label>Feed URL<input required type="url" value={config.url} onChange={event => update('url', event.target.value)} placeholder="https://service.example.com/odata/Products" /></label>
            <label>Bearer token (optional)<input type="password" value={config.token} onChange={event => update('token', event.target.value)} placeholder="Access token" /></label>
            {error && <p className="source-error">{error}</p>}
            <button className="btn btn-primary source-submit" disabled={busy}>{busy ? 'Connectingâ€¦' : 'Connect and import'}<ArrowRight size={16} /></button>
          </form>}

          {selected.kind === 'github' && <form className="source-form-block" onSubmit={submit}>
            <p>Import up to 100 records from a public repository, or provide a personal access token for private repositories.</p>
            <label>Repository<input required value={config.repository} onChange={event => update('repository', event.target.value)} placeholder="owner/repository" /></label>
            <label>Data to import<select value={config.resource} onChange={event => update('resource', event.target.value)}><option value="issues">Issues</option><option value="pulls">Pull requests</option><option value="commits">Commits</option><option value="releases">Releases</option></select></label>
            <label>Personal access token (optional)<input type="password" value={config.token} onChange={event => update('token', event.target.value)} placeholder="ghp_â€¦" /></label>
            {error && <p className="source-error">{error}</p>}
            <button className="btn btn-primary source-submit" disabled={busy}>{busy ? 'Connectingâ€¦' : 'Connect and import'}<ArrowRight size={16} /></button>
          </form>}

          {selected.kind === 'web' && <form className="source-form-block" onSubmit={submit}>
            <p>Extract HTML tables from a public HTTPS page. Login-only pages are not supported.</p>
            <label>Page URL<input required type="url" value={config.url} onChange={event => update('url', event.target.value)} placeholder="https://example.com/data" /></label>
            {error && <p className="source-error">{error}</p>}
            <button className="btn btn-primary source-submit" disabled={busy}>{busy ? 'Reading pageâ€¦' : 'Import tables'}<ArrowRight size={16} /></button>
          </form>}

          {selected.kind === 'unavailable' && <div className="source-unavailable"><Cloud size={22} /><strong>{selected.name} needs additional setup</strong><p>{selected.detail}. This connector is not enabled in this build.</p></div>}
          <footer className="source-dialog-footer"><span>Imported data is stored in your local SQLite workspace.</span><button type="button" onClick={onClose}><ArrowLeft size={14} /> Back to workspace</button></footer>
        </div>
      </div>
    </section>
  </div>;
}

