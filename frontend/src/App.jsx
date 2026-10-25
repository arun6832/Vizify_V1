import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DataExplorer from './components/DataExplorer';
import DataProfiler from './components/DataProfiler';
import NLPStudio from './components/NLPStudio';
import MLStudio from './components/MLStudio';
import AIDataAgent from './components/AIDataAgent';
import { request } from './api';

const CHART_TYPES = [
  'Clustered Column',
  'Stacked Column',
  'Clustered Bar',
  'Stacked Bar',
  'Radar (Spider) Chart',
  'Line Chart',
  'Area Chart',
  'Stacked Area Chart',
  'Waterfall Chart',
  'Stock (OHLC) Chart',
  'Pie Chart',
  'Doughnut Chart',
  'Treemap Chart',
  'Sunburst Chart',
  'Scatter Plot',
  'Bubble Chart',
  'Histogram',
  'Box and Whisker',
  'Filled Map',
  'Funnel Chart',
  'D3 Word Cloud',
  'Text-to-Chart (AI)'
];

export default function App() {
  const [activeTab, setActiveTab] = useState('explorer');
  const [fileLoaded, setFileLoaded] = useState(false);
  const [dataInfo, setDataInfo] = useState({});
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('gemini-2.0-flash');
  const [dashboardItems, setDashboardItems] = useState([]);
  const [globalFilters, setGlobalFilters] = useState({
    categorical: {},
    numeric: {},
    datetime: {}
  });
  const [theme, setTheme] = useState(() => localStorage.getItem('vizify-theme') || 'dark');

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('vizify-theme', theme);
  }, [theme]);

  // Restore the SQLite workspace and any configured Gemini key.
  useEffect(() => {
    async function initApp() {
      try {
        const state = await request('/api/init');
        if (state.api_key) {
          setApiKey(state.api_key);
        }
        if (state.file_loaded) {
          setDataInfo(state.info);
          setFileLoaded(true);
        }
      } catch (err) {
        console.error('Failed to initialize app state:', err);
      }
    }
    initApp();
  }, []);

  const handleAddChart = (type) => {
    setDashboardItems(prev => [...prev, { id: Date.now(), type }]);
  };

  const handleRemoveChart = (id) => {
    setDashboardItems(prev => prev.filter(item => item.id !== id));
  };

  return (
    <div className="app-container">
      {/* Sidebar Panel */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        fileLoaded={fileLoaded}
        setFileLoaded={setFileLoaded}
        dataInfo={dataInfo}
        setDataInfo={setDataInfo}
        api_key={apiKey}
        setApiKey={setApiKey}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        onAddChart={handleAddChart}
        globalFilters={globalFilters}
        setGlobalFilters={setGlobalFilters}
        chartTypes={CHART_TYPES}
        theme={theme}
        onToggleTheme={() => setTheme(current => current === 'dark' ? 'light' : 'dark')}
      />

      {/* Main Viewport */}
      <main className="main-content">
        {!fileLoaded ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '60vh',
            textAlign: 'center'
          }} className="fade-in">
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-xl)',
              padding: '3rem 2rem',
              maxWidth: '600px',
              boxShadow: 'var(--shadow-lg)'
            }}>
              <h2 style={{ fontSize: '2rem', marginBottom: '1rem', fontWeight: 700 }} className="glow-text-primary">
                Welcome to Vizify
              </h2>
              <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)', marginBottom: '2rem', lineHeight: 1.6 }}>
                Explore your data, build visualizations, clean text, and train practical models in one workspace.
              </p>
              <div style={{ 
                background: 'rgba(139,92,246,0.06)', 
                border: '1px dashed rgba(139,92,246,0.3)', 
                padding: '1.25rem', 
                borderRadius: 'var(--radius-lg)',
                color: 'var(--text-primary)',
                fontWeight: 500,
                fontSize: '0.95rem'
              }}>
                👈 Open Data Sources to connect a file, database, or online service.
              </div>
            </div>
          </div>
        ) : (
          <>
            {activeTab === 'explorer' && (
              <DataExplorer
                dashboardItems={dashboardItems}
                onRemoveChart={handleRemoveChart}
                dataInfo={dataInfo}
                globalFilters={globalFilters}
                apiKey={apiKey}
                selectedModel={selectedModel}
              />
            )}

            {activeTab === 'profiler' && (
              <DataProfiler
                dataInfo={dataInfo}
                setDataInfo={setDataInfo}
              />
            )}

            {activeTab === 'nlp' && (
              <NLPStudio
                dataInfo={dataInfo}
                onDataUpdated={(newInfo) => setDataInfo(newInfo)}
              />
            )}

            {activeTab === 'ml' && (
              <MLStudio
                dataInfo={dataInfo}
              />
            )}

            {activeTab === 'agent' && (
              <AIDataAgent
                apiKey={apiKey}
                selectedModel={selectedModel}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
