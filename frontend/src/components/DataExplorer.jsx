import React, { useState, useEffect } from 'react';
import { Trash2, MessageSquare, BrainCircuit, Send, Sparkles } from 'lucide-react';
import { request } from '../api';
import Plot from './Plot';

export default function DataExplorer({
  dashboardItems,
  onRemoveChart,
  dataInfo,
  globalFilters,
  apiKey,
  selectedModel
}) {
  const [chartsData, setChartsData] = useState({});
  const [loadingCharts, setLoadingCharts] = useState({});
  const [chartErrors, setChartErrors] = useState({});
  const [aiComments, setAiComments] = useState({});
  const [loadingAi, setLoadingAi] = useState({});
  const [chatHistories, setChatHistories] = useState({});
  const [chatInputs, setChatInputs] = useState({});
  const [loadingChat, setLoadingChat] = useState({});

  // Individual chart settings state (key: itemId)
  const [chartSettings, setChartSettings] = useState({});

  // Set default settings when items are added
  useEffect(() => {
    const nextSettings = { ...chartSettings };
    let changed = false;

    dashboardItems.forEach(item => {
      if (!nextSettings[item.id]) {
        changed = true;
        if (item.type === 'Distribution Plot') {
          nextSettings[item.id] = { column: dataInfo.numeric_cols?.[0] || '' };
        } else if (item.type === 'Categorical Plot') {
          nextSettings[item.id] = { column: dataInfo.categorical_cols?.[0] || '' };
        } else if (item.type === 'Scatter Plot') {
          nextSettings[item.id] = {
            xAxis: dataInfo.numeric_cols?.[0] || '',
            yAxis: dataInfo.numeric_cols?.[1] || dataInfo.numeric_cols?.[0] || '',
            color: '(none)'
          };
        } else if (item.type === 'Text-to-Chart (AI)') {
          nextSettings[item.id] = { prompt: '' };
        }
      }
    });

    if (changed) {
      setChartSettings(nextSettings);
    }
  }, [dashboardItems, dataInfo]);

  // Load/update charts when settings or filters change
  useEffect(() => {
    dashboardItems.forEach(item => {
      const settings = chartSettings[item.id];
      // Skip if settings are not loaded yet
      if (item.type !== 'Correlation Heatmap' && !settings) return;

      loadChart(item.id, item.type, settings);
    });
  }, [dashboardItems, chartSettings, globalFilters]);

  const loadChart = async (itemId, type, settings) => {
    setLoadingCharts(prev => ({ ...prev, [itemId]: true }));
    setChartErrors(prev => ({ ...prev, [itemId]: null }));

    try {
      const chartJson = await request('/api/get-chart', {
        method: 'POST',
        body: JSON.stringify({
          type,
          settings: settings || {},
          filters: globalFilters
        }),
      });

      setChartsData(prev => ({ ...prev, [itemId]: chartJson }));
    } catch (err) {
      setChartErrors(prev => ({ ...prev, [itemId]: err.message }));
    } finally {
      setLoadingCharts(prev => ({ ...prev, [itemId]: false }));
    }
  };

  const generateAiInterpretation = async (itemId, type) => {
    setLoadingAi(prev => ({ ...prev, [itemId]: true }));
    try {
      const result = await request('/api/chat-chart', {
        method: 'POST',
        body: JSON.stringify({
          type,
          settings: chartSettings[itemId] || {},
          filters: globalFilters,
          question: "Generate a brief 2-3 sentence interpretation of this chart summarizing the key insights.",
          apiKey,
          model: selectedModel
        }),
      });
      setAiComments(prev => ({ ...prev, [itemId]: result.response }));
    } catch (err) {
      alert(`AI Error: ${err.message}`);
    } finally {
      setLoadingAi(prev => ({ ...prev, [itemId]: false }));
    }
  };

  const handleChatSend = async (itemId, type) => {
    const input = chatInputs[itemId]?.trim();
    if (!input) return;

    // Clear input
    setChatInputs(prev => ({ ...prev, [itemId]: '' }));

    // Add user message
    const history = chatHistories[itemId] || [];
    const updatedHistory = [...history, { role: 'user', content: input }];
    setChatHistories(prev => ({ ...prev, [itemId]: updatedHistory }));

    setLoadingChat(prev => ({ ...prev, [itemId]: true }));
    try {
      const result = await request('/api/chat-chart', {
        method: 'POST',
        body: JSON.stringify({
          type,
          settings: chartSettings[itemId] || {},
          filters: globalFilters,
          question: input,
          history: updatedHistory,
          apiKey,
          model: selectedModel
        }),
      });

      setChatHistories(prev => ({
        ...prev,
        [itemId]: [...updatedHistory, { role: 'assistant', content: result.response }]
      }));
    } catch (err) {
      setChatHistories(prev => ({
        ...prev,
        [itemId]: [...updatedHistory, { role: 'assistant', content: `Error: ${err.message}` }]
      }));
    } finally {
      setLoadingChat(prev => ({ ...prev, [itemId]: false }));
    }
  };

  const handleSettingChange = (itemId, key, val) => {
    setChartSettings(prev => ({
      ...prev,
      [itemId]: {
        ...prev[itemId],
        [key]: val
      }
    }));
  };

  const triggerExportPdf = () => {
    const apiBase = window.location.port === '5173' ? 'http://localhost:5000' : '';
    const itemsParam = encodeURIComponent(JSON.stringify(
      dashboardItems.map(item => ({
        type: item.type,
        settings: chartSettings[item.id] || {}
      }))
    ));
    const filtersParam = encodeURIComponent(JSON.stringify(globalFilters));
    window.location.href = `${apiBase}/api/export-pdf?items=${itemsParam}&filters=${filtersParam}`;
  };

  const triggerExportCsv = () => {
    const apiBase = window.location.port === '5173' ? 'http://localhost:5000' : '';
    const filtersParam = encodeURIComponent(JSON.stringify(globalFilters));
    window.location.href = `${apiBase}/api/export-csv?filters=${filtersParam}`;
  };

  return (
    <div className="fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700 }} className="glow-text-primary">
            📊 Interactive Dashboard
          </h2>
          <p style={{ fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Build custom widgets, slice variables, and generate conversational charts.
          </p>
        </div>
        {dashboardItems.length > 0 && (
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="btn" onClick={triggerExportCsv}>
              Export Filtered CSV
            </button>
            <button className="btn btn-primary" onClick={triggerExportPdf}>
              Download PDF Report
            </button>
          </div>
        )}
      </div>

      {dashboardItems.length === 0 ? (
        <div style={{
          background: 'var(--bg-card)',
          border: '1px dashed var(--border-color)',
          borderRadius: 'var(--radius-xl)',
          padding: '4rem 2rem',
          textAlign: 'center',
          marginTop: '2rem'
        }}>
          <Sparkles size={48} style={{ color: 'var(--accent-primary)', marginBottom: '1rem', opacity: 0.8 }} />
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Your Dashboard is Empty</h3>
          <p style={{ maxWidth: '400px', margin: '0 auto', fontSize: '0.95rem' }}>
            Add new widgets from the sidebar controls to explore distributions, correlations, and build custom plots.
          </p>
        </div>
      ) : (
        <div className="dashboard-grid">
          {dashboardItems.map(item => {
            const settings = chartSettings[item.id] || {};
            const chartData = chartsData[item.id];
            const error = chartErrors[item.id];
            const loading = loadingCharts[item.id];
            const comment = aiComments[item.id];
            const chatHistory = chatHistories[item.id] || [];

            return (
              <div 
                key={item.id} 
                className="glass-card fade-in"
                style={{ 
                  gridColumn: item.type === 'Correlation Heatmap' || item.type === 'Scatter Plot' ? '1 / -1' : 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1rem'
                }}
              >
                {/* Card Header */}
                <div className="card-header" style={{ marginBottom: 0 }}>
                  <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.1rem' }}>
                    {item.type}
                  </h4>
                  <button 
                    onClick={() => onRemoveChart(item.id)}
                    className="btn btn-danger"
                    style={{ padding: '0.4rem', border: 'none', background: 'transparent' }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>

                {/* Chart Settings Controls */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', background: 'rgba(255,255,255,0.01)', padding: '0.75rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                  {item.type === 'Distribution Plot' && (
                    <div>
                      <label>Numeric Column</label>
                      <select 
                        value={settings.column || ''}
                        onChange={(e) => handleSettingChange(item.id, 'column', e.target.value)}
                      >
                        {dataInfo.numeric_cols?.map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  {item.type === 'Categorical Plot' && (
                    <div>
                      <label>Categorical Column</label>
                      <select 
                        value={settings.column || ''}
                        onChange={(e) => handleSettingChange(item.id, 'column', e.target.value)}
                      >
                        {dataInfo.categorical_cols?.map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  {item.type === 'Scatter Plot' && (
                    <>
                      <div>
                        <label>X Axis (Numeric)</label>
                        <select 
                          value={settings.xAxis || ''}
                          onChange={(e) => handleSettingChange(item.id, 'xAxis', e.target.value)}
                        >
                          {dataInfo.numeric_cols?.map(col => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label>Y Axis (Numeric)</label>
                        <select 
                          value={settings.yAxis || ''}
                          onChange={(e) => handleSettingChange(item.id, 'yAxis', e.target.value)}
                        >
                          {dataInfo.numeric_cols?.map(col => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label>Color Coding (Category)</label>
                        <select 
                          value={settings.color || '(none)'}
                          onChange={(e) => handleSettingChange(item.id, 'color', e.target.value)}
                        >
                          <option value="(none)">(none)</option>
                          {dataInfo.categorical_cols?.map(col => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </div>
                    </>
                  )}

                  {item.type === 'Text-to-Chart (AI)' && (
                    <div style={{ gridColumn: '1 / -1', display: 'flex', gap: '0.5rem' }}>
                      <input 
                        type="text" 
                        placeholder="e.g. Plot correlation between age and fare as a box plot..."
                        value={settings.prompt || ''}
                        onChange={(e) => handleSettingChange(item.id, 'prompt', e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') loadChart(item.id, item.type, settings);
                        }}
                      />
                      <button 
                        className="btn btn-primary"
                        onClick={() => loadChart(item.id, item.type, settings)}
                        disabled={loading}
                      >
                        Generate
                      </button>
                    </div>
                  )}

                  {item.type === 'Correlation Heatmap' && (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                      Renders correlations across all numerical attributes. Sliced automatically by global filters.
                    </div>
                  )}
                </div>

                {/* Plot Panel */}
                <div style={{ position: 'relative', minHeight: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {loading && (
                    <div style={{ position: 'absolute', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                      <div className="spinner"></div>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Calculating data...</span>
                    </div>
                  )}
                  {error && (
                    <div style={{ color: 'var(--accent-red)', fontSize: '0.9rem', textAlign: 'center', padding: '1rem' }}>
                      ⚠️ {error}
                    </div>
                  )}
                  {!loading && !error && chartData && (
                    <Plot 
                      data={chartData.data} 
                      layout={chartData.layout} 
                      style={{ height: item.type === 'Correlation Heatmap' || item.type === 'Scatter Plot' ? '450px' : '350px' }}
                    />
                  )}
                </div>

                {/* AI Interpretations Accordion */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button 
                      className="btn" 
                      onClick={() => generateAiInterpretation(item.id, item.type)}
                      disabled={loadingAi[item.id] || !apiKey}
                      style={{ flexGrow: 1, justifyContent: 'center', fontSize: '0.85rem' }}
                    >
                      <BrainCircuit size={16} /> 
                      {loadingAi[item.id] ? 'Analyzing chart...' : 'Get AI Interpretation'}
                    </button>
                  </div>

                  {comment && (
                    <div style={{ background: 'rgba(139,92,246,0.06)', border: '1px solid rgba(139,92,246,0.15)', borderRadius: 'var(--radius-md)', padding: '0.85rem', fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                      <strong>🤖 AI Insights:</strong> {comment}
                    </div>
                  )}
                </div>

                {/* Chat Panel Expandable */}
                <details style={{ width: '100%' }}>
                  <summary style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.85rem', userSelect: 'none', padding: '0.5rem 0' }}>
                    <MessageSquare size={14} /> Chat about this chart
                  </summary>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
                    <div style={{
                      maxHeight: '200px',
                      overflowY: 'auto',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-md)',
                      padding: '0.75rem',
                      background: 'rgba(15,23,42,0.3)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.5rem'
                    }}>
                      {chatHistory.length === 0 && (
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic', textAlign: 'center', margin: 'auto' }}>
                          Ask specific queries (e.g. "What does the outlier in the top right mean?")
                        </span>
                      )}
                      {chatHistory.map((msg, idx) => (
                        <div 
                          key={idx} 
                          style={{
                            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                            background: msg.role === 'user' ? 'rgba(139,92,246,0.15)' : 'rgba(30,41,59,0.7)',
                            border: '1px solid ' + (msg.role === 'user' ? 'rgba(139,92,246,0.3)' : 'var(--border-color)'),
                            padding: '0.5rem 0.75rem',
                            borderRadius: 'var(--radius-md)',
                            fontSize: '0.85rem',
                            maxWidth: '85%',
                            wordBreak: 'break-word'
                          }}
                        >
                          {msg.content}
                        </div>
                      ))}
                      {loadingChat[item.id] && (
                        <div style={{ alignSelf: 'flex-start', display: 'flex', gap: '0.35rem', padding: '0.5rem 0.75rem', background: 'rgba(30,41,59,0.7)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)' }}>
                          <span className="spinner" style={{ width: '12px', height: '12px', borderWidth: '2px' }}></span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Thinking...</span>
                        </div>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <input 
                        type="text" 
                        placeholder={apiKey ? "Ask a question..." : "Enter API Key in sidebar to chat"}
                        value={chatInputs[item.id] || ''}
                        disabled={!apiKey}
                        onChange={(e) => setChatInputs(prev => ({ ...prev, [item.id]: e.target.value }))}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleChatSend(item.id, item.type);
                        }}
                        style={{ fontSize: '0.85rem', padding: '0.5rem 0.75rem' }}
                      />
                      <button 
                        className="btn btn-primary"
                        style={{ padding: '0.5rem' }}
                        disabled={!apiKey || loadingChat[item.id]}
                        onClick={() => handleChatSend(item.id, item.type)}
                      >
                        <Send size={14} />
                      </button>
                    </div>
                  </div>
                </details>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
