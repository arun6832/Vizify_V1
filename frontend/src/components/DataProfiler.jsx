import React, { useState } from 'react';
import { Sparkles, Check, AlertCircle } from 'lucide-react';
import { request } from '../api';

export default function DataProfiler({ dataInfo, setDataInfo }) {
  const [cleaning, setCleaning] = useState(false);
  const [cleanReport, setCleanReport] = useState(null);

  const runMagicClean = async () => {
    setCleaning(true);
    setCleanReport(null);

    try {
      const result = await request('/api/magic-clean', { method: 'POST' });
      setDataInfo(result.info);
      setCleanReport(result.actions);
    } catch (err) {
      alert(`Cleaning failed: ${err.message}`);
    } finally {
      setCleaning(false);
    }
  };

  // Compute metrics
  const totalRows = dataInfo.shape?.[0] || 0;
  const totalCols = dataInfo.shape?.[1] || 0;
  const missingCount = dataInfo.missing_count || 0;
  const missingPct = dataInfo.missing_pct || 0;
  const duplicateCount = dataInfo.duplicate_count || 0;

  // Prepare column statistics list
  const columnsList = dataInfo.columns_health || [];
  const summaryStats = dataInfo.summary_stats || {};
  const statRows = Object.keys(summaryStats);

  return (
    <div className="fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700 }} className="glow-text-primary">
          🧹 Data Profiler & Cleaner
        </h2>
        <p style={{ fontSize: '0.9rem', marginTop: '0.25rem' }}>
          Inspect dataset structures, check missing values, and run automated repairs.
        </p>
      </div>

      {/* Metrics Row */}
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-label">Total Rows</div>
          <div className="metric-value">{totalRows.toLocaleString()}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Index bounds</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Total Columns</div>
          <div className="metric-value">{totalCols}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Features mapped</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Missing Values</div>
          <div className="metric-value" style={{ color: missingCount > 0 ? 'var(--accent-amber)' : 'var(--accent-green)' }}>
            {missingCount.toLocaleString()}
          </div>
          <div className="metric-change negative" style={{ color: missingCount > 0 ? 'var(--accent-amber)' : 'var(--accent-green)' }}>
            {missingPct.toFixed(1)}% cell vacancy
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Duplicate Rows</div>
          <div className="metric-value" style={{ color: duplicateCount > 0 ? 'var(--accent-red)' : 'var(--text-primary)' }}>
            {duplicateCount.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Redundant rows</div>
        </div>
      </div>

      {/* Row 2: Columns breakdown and Clean action */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Column Types & Health */}
        <div className="glass-card" style={{ overflowX: 'auto' }}>
          <h3 style={{ fontSize: '1.15rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>
            📊 Column Health
          </h3>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Column Name</th>
                  <th>Data Type</th>
                  <th>Non-Null</th>
                  <th>Missing %</th>
                </tr>
              </thead>
              <tbody>
                {columnsList.map(col => {
                  const hasMissing = parseFloat(col.missing_pct) > 0;
                  return (
                    <tr key={col.name}>
                      <td style={{ fontWeight: 500 }}>{col.name}</td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-muted)' }}>{col.type}</td>
                      <td>{col.non_null.toLocaleString()}</td>
                      <td style={{ color: hasMissing ? 'var(--accent-amber)' : 'var(--accent-green)', fontWeight: 500 }}>
                        {col.missing_pct}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Magic Data Cleaner panel */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.15rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={20} style={{ color: 'var(--accent-primary)' }} /> Magic Data Cleaner
          </h3>
          <p style={{ fontSize: '0.9rem' }}>
            Run our automated data cleaning engine to format text structures and fix missing fields:
          </p>
          <ul style={{ paddingLeft: '1.25rem', fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <li>Drop any columns containing more than 90% null values automatically.</li>
            <li>Strip standard currency indicators ($) and text commas from numbers.</li>
            <li>Detect temporal patterns and coerce fields to date/time format.</li>
          </ul>

          <button 
            className="btn btn-primary" 
            style={{ width: '100%', marginTop: 'auto' }}
            onClick={runMagicClean}
            disabled={cleaning}
          >
            {cleaning ? (
              <>
                <span className="spinner" style={{ width: '14px', height: '14px', borderWidth: '2px' }}></span>
                Running clean...
              </>
            ) : (
              <>
                <Sparkles size={16} /> Run Magic Clean
              </>
            )}
          </button>

          {cleanReport && (
            <div style={{
              background: 'rgba(16,185,129,0.06)',
              border: '1px solid rgba(16,185,129,0.15)',
              borderRadius: 'var(--radius-md)',
              padding: '0.75rem',
              fontSize: '0.85rem',
              maxHeight: '150px',
              overflowY: 'auto'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--accent-green)', fontWeight: 600, marginBottom: '0.5rem' }}>
                <Check size={16} /> Cleaning Complete!
              </div>
              <ul style={{ paddingLeft: '1rem', listStyleType: 'circle', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                {cleanReport.length === 0 ? (
                  <li style={{ color: 'var(--text-muted)' }}>Dataset is already clean. No modifications needed!</li>
                ) : (
                  cleanReport.map((act, i) => (
                    <li key={i}>{act}</li>
                  ))
                )}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Row 3: Descriptive Statistics */}
      <div className="glass-card" style={{ overflowX: 'auto' }}>
        <h3 style={{ fontSize: '1.15rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>
          📈 Summary Statistics
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Stat Attribute</th>
                {dataInfo.columns?.map(col => (
                  <th key={col} style={{ minWidth: '120px' }}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {statRows.map(row => (
                <tr key={row}>
                  <td style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{row}</td>
                  {dataInfo.columns?.map(col => {
                    const value = summaryStats[row]?.[col];
                    const displayVal = typeof value === 'number' 
                      ? value % 1 === 0 ? value : value.toFixed(3)
                      : value === null || value === undefined ? 'N/A' : String(value);
                    return (
                      <td key={col}>{displayVal}</td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
