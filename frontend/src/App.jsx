import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DataExplorer from './components/DataExplorer';
import DataProfiler from './components/DataProfiler';
import MLStudio from './components/MLStudio';
import AIDataAgent from './components/AIDataAgent';
import { request } from './api';
import { Database, Sparkles, Cpu, MessageSquare } from 'lucide-react';

const CHART_TYPES = [
  'Distribution Plot',
  'Categorical Plot',
  'Scatter Plot',
  'Correlation Heatmap',
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

  // Load initial server state (pre-loaded dataframe/file, env api key, etc.)
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
                Welcome to Vizify Studio
              </h2>
              <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)', marginBottom: '2rem', lineHeight: 1.6 }}>
                A premium interactive web studio for automated visual analysis, statistical diagnostics, and no-code machine learning models.
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
                👈 Get started by uploading a CSV dataset in the sidebar control panel.
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
