import React, { useState } from 'react';
import { 
  Database, 
  Sparkles, 
  Cpu, 
  MessageSquare, 
  Upload, 
  Plus, 
  Filter, 
  ChevronDown, 
  ChevronUp, 
  Eye, 
  EyeOff 
} from 'lucide-react';
import { request } from '../api';

export default function Sidebar({
  activeTab,
  setActiveTab,
  fileLoaded,
  setFileLoaded,
  dataInfo,
  setDataInfo,
  api_key,
  setApiKey,
  selectedModel,
  setSelectedModel,
  onAddChart,
  globalFilters,
  setGlobalFilters,
  chartTypes
}) {
  const [showApiKey, setShowApiKey] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedChartType, setSelectedChartType] = useState('Distribution Plot');

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(
        `${window.location.port === '5173' ? 'http://localhost:5000' : ''}/api/upload`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const info = await response.json();
      setDataInfo(info);
      setFileLoaded(true);
      // Reset filters when a new file is uploaded
      setGlobalFilters({
        categorical: {},
        numeric: {},
        datetime: {}
      });
    } catch (err) {
      alert(`Error uploading file: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const updateNumericFilter = (col, valIdx, value) => {
    setGlobalFilters(prev => {
      const current = prev.numeric[col] || [dataInfo.numeric_ranges[col][0], dataInfo.numeric_ranges[col][1]];
      const next = [...current];
      next[valIdx] = parseFloat(value);
      return {
        ...prev,
        numeric: {
          ...prev.numeric,
          [col]: next
        }
      };
    });
  };

  const updateCategoricalFilter = (col, value) => {
    setGlobalFilters(prev => {
      const current = prev.categorical[col] || [];
      const next = current.includes(value)
        ? current.filter(v => v !== value)
        : [...current, value];
      
      return {
        ...prev,
        categorical: {
          ...prev.categorical,
          [col]: next
        }
      };
    });
  };

  const updateDateFilter = (col, field, value) => {
    setGlobalFilters(prev => {
      const current = prev.datetime[col] || { start: '', end: '' };
      return {
        ...prev,
        datetime: {
          ...prev.datetime,
          [col]: {
            ...current,
            [field]: value
          }
        }
      };
    });
  };

  return (
    <aside className="sidebar">
      {/* Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem' }}>
        <Sparkles size={28} className="glow-text-primary" style={{ color: 'var(--accent-primary)' }} />
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.02em' }} className="glow-text-primary">
            Vizify AI
          </h1>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>v0.5.0</span>
        </div>
      </div>

      {/* Upload & Setup */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginBottom: '1.5rem' }}>
        <div>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Database size={16} /> 1) Data Source
          </label>
          <div style={{ position: 'relative' }}>
            <label className="btn" style={{ width: '100%', justifyContent: 'center', display: 'flex', cursor: 'pointer', gap: '0.5rem' }}>
              <Upload size={16} /> {uploading ? 'Uploading...' : fileLoaded ? 'Change CSV File' : 'Upload CSV File'}
              <input 
                type="file" 
                accept=".csv" 
                onChange={handleFileUpload} 
                style={{ display: 'none' }} 
              />
            </label>
            {fileLoaded && (
              <div style={{ fontSize: '0.75rem', color: 'var(--accent-green)', marginTop: '0.5rem', textAlign: 'center', fontWeight: 500 }}>
                Active: {dataInfo.filename} ({dataInfo.shape[0]} rows)
              </div>
            )}
          </div>
        </div>

        {/* API Key */}
        <div>
          <label>2) Gemini API Key (Optional)</label>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <input
              type={showApiKey ? 'text' : 'password'}
              placeholder="Enter Gemini API Key..."
              value={api_key}
              onChange={(e) => setApiKey(e.target.value)}
              style={{ paddingRight: '2.5rem' }}
            />
            <button
              type="button"
              onClick={() => setShowApiKey(!showApiKey)}
              style={{
                position: 'absolute',
                right: '0.25rem',
                background: 'transparent',
                border: 'none',
                padding: '0.5rem',
                color: 'var(--text-muted)',
                cursor: 'pointer'
              }}
            >
              {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </div>
      </div>

      {/* Navigation */}
      {fileLoaded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flexGrow: 1, overflowY: 'auto' }}>
          <h3 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
            Navigation
          </h3>
          <ul className="nav-list" style={{ marginTop: 0, gap: '0.35rem' }}>
            <li 
              className={`nav-item ${activeTab === 'explorer' ? 'active' : ''}`}
              onClick={() => setActiveTab('explorer')}
            >
              <Database size={18} />
              <span>Data Explorer</span>
            </li>
            <li 
              className={`nav-item ${activeTab === 'profiler' ? 'active' : ''}`}
              onClick={() => setActiveTab('profiler')}
            >
              <Sparkles size={18} />
              <span>Data Profiler</span>
            </li>
            <li 
              className={`nav-item ${activeTab === 'ml' ? 'active' : ''}`}
              onClick={() => setActiveTab('ml')}
            >
              <Cpu size={18} />
              <span>ML Studio</span>
            </li>
            <li 
              className={`nav-item ${activeTab === 'agent' ? 'active' : ''}`}
              onClick={() => setActiveTab('agent')}
            >
              <MessageSquare size={18} />
              <span>AI Data Agent</span>
            </li>
          </ul>

          {/* Add Chart to Dashboard widget (Explorer specific) */}
          {activeTab === 'explorer' && (
            <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)' }}>
              <label>Add Dashboard Widget</label>
              <select 
                value={selectedChartType} 
                onChange={(e) => setSelectedChartType(e.target.value)}
                style={{ marginBottom: '0.75rem' }}
              >
                {chartTypes.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
              <button 
                className="btn btn-primary" 
                style={{ width: '100%' }}
                onClick={() => onAddChart(selectedChartType)}
              >
                <Plus size={16} /> Add Widget
              </button>
            </div>
          )}

          {/* Global Data Filters expandable panel */}
          <div style={{ marginTop: '1.25rem' }}>
            <button 
              className="btn" 
              style={{ width: '100%', justifyContent: 'space-between' }}
              onClick={() => setShowFilters(!showFilters)}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Filter size={16} /> Global Filters
              </span>
              {showFilters ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            
            {showFilters && (
              <div style={{ 
                marginTop: '0.75rem', 
                padding: '0.75rem', 
                background: 'rgba(15,23,42,0.4)', 
                borderRadius: 'var(--radius-md)', 
                border: '1px solid var(--border-color)', 
                display: 'flex', 
                flexDirection: 'column', 
                gap: '1rem',
                maxHeight: '250px',
                overflowY: 'auto',
                fontSize: '0.85rem'
              }}>
                {/* Categorical filters */}
                {dataInfo.categorical_cols && dataInfo.categorical_cols.map(col => {
                  const uniqueVals = dataInfo.categorical_values?.[col] || [];
                  if (uniqueVals.length === 0) return null;
                  return (
                    <div key={col}>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>{col}</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', paddingLeft: '0.25rem' }}>
                        {uniqueVals.map(val => (
                          <label key={val} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', margin: 0 }}>
                            <input 
                              type="checkbox" 
                              checked={globalFilters.categorical[col]?.includes(val) || false}
                              onChange={() => updateCategoricalFilter(col, val)}
                            />
                            <span>{val || '(Blank)'}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  );
                })}

                {/* Numerical filters */}
                {dataInfo.numeric_cols && dataInfo.numeric_cols.slice(0, 6).map(col => {
                  const range = dataInfo.numeric_ranges?.[col];
                  if (!range || range[0] === range[1]) return null;
                  const currentRange = globalFilters.numeric[col] || range;
                  return (
                    <div key={col}>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem', display: 'flex', justifyContent: 'space-between' }}>
                        <span>{col}</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          {currentRange[0]} - {currentRange[1]}
                        </span>
                      </div>
                      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        <input 
                          type="range" 
                          min={range[0]} 
                          max={range[1]} 
                          step={(range[1] - range[0]) / 100}
                          value={currentRange[0]} 
                          onChange={(e) => updateNumericFilter(col, 0, e.target.value)}
                        />
                        <input 
                          type="range" 
                          min={range[0]} 
                          max={range[1]} 
                          step={(range[1] - range[0]) / 100}
                          value={currentRange[1]} 
                          onChange={(e) => updateNumericFilter(col, 1, e.target.value)}
                        />
                      </div>
                    </div>
                  );
                })}

                {/* Datetime filters */}
                {dataInfo.time_cols && dataInfo.time_cols.slice(0, 3).map(col => {
                  const currentRange = globalFilters.datetime[col] || { start: '', end: '' };
                  return (
                    <div key={col}>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>{col} range</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                        <input 
                          type="date" 
                          value={currentRange.start}
                          onChange={(e) => updateDateFilter(col, 'start', e.target.value)}
                          style={{ padding: '0.35rem 0.5rem', fontSize: '0.8rem' }}
                        />
                        <input 
                          type="date" 
                          value={currentRange.end}
                          onChange={(e) => updateDateFilter(col, 'end', e.target.value)}
                          style={{ padding: '0.35rem 0.5rem', fontSize: '0.8rem' }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
