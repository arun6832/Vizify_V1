import React, { useState, useEffect } from 'react';
import { request } from '../api';
import D3WordCloud from './d3/D3WordCloud';
import D3ForceGraph from './d3/D3ForceGraph';
import D3TreeMap from './d3/D3TreeMap';
import D3BarChart from './d3/D3BarChart';
import { Type, Sparkles, Filter, Smile, Network, Scissors, Check, RefreshCw, Database } from 'lucide-react';

export default function NLPStudio({ dataInfo, onDataUpdated }) {
  const [textColumns, setTextColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cleaningLoading, setCleaningLoading] = useState(false);
  const [activeView, setActiveView] = useState('cloud');
  const [message, setMessage] = useState('');

  // Cleaning options state
  const [cleanOptions, setCleanOptions] = useState({
    lower: true,
    remove_punct: true,
    remove_stopwords: true,
    remove_digits: false,
    remove_urls: true,
    remove_html: true,
    normalize_space: true,
    custom_regex: '',
    custom_replace: ''
  });

  // Fetch available text columns
  useEffect(() => {
    async function loadTextCols() {
      try {
        const res = await request('/api/nlp/text-columns');
        if (res.text_columns && res.text_columns.length > 0) {
          setTextColumns(res.text_columns);
          setSelectedColumn(res.text_columns[0]);
        } else if (dataInfo && dataInfo.categorical_cols && dataInfo.categorical_cols.length > 0) {
          setTextColumns(dataInfo.categorical_cols);
          setSelectedColumn(dataInfo.categorical_cols[0]);
        }
      } catch (err) {
        console.error('Failed to load text columns:', err);
      }
    }
    loadTextCols();
  }, [dataInfo]);

  // Run analysis when column changes
  useEffect(() => {
    if (!selectedColumn) return;
    runAnalysis(selectedColumn);
  }, [selectedColumn]);

  const runAnalysis = async (colName) => {
    setLoading(true);
    setMessage('');
    try {
      const res = await request('/api/nlp/analyze', { method: 'POST', body: JSON.stringify({ column: colName }) });
      if (res.error) {
        setMessage(res.error);
        setAnalysisData(null);
      } else {
        setAnalysisData(res);
      }
    } catch (err) {
      setMessage(`Failed to analyze text column: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyClean = async () => {
    if (!selectedColumn) return;
    setCleaningLoading(true);
    setMessage('');
    try {
      const res = await request('/api/nlp/clean', {
        method: 'POST',
        body: JSON.stringify({
          column: selectedColumn,
          options: cleanOptions
        })
      });
      if (res.error) {
        setMessage(res.error);
      } else {
        setMessage(res.message || 'Text cleaned successfully!');
        if (onDataUpdated && res.info) {
          onDataUpdated(res.info);
        }
        // Refresh analysis
        await runAnalysis(selectedColumn);
      }
    } catch (err) {
      setMessage(`Error applying text cleaning: ${err.message}`);
    } finally {
      setCleaningLoading(false);
    }
  };

  return (
    <div className="fade-in" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Header Bar */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-color)',
        borderRadius: 'var(--radius-xl)',
        padding: '1.25rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
        boxShadow: 'var(--shadow-md)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)',
            borderRadius: '10px',
            padding: '0.5rem',
            display: 'flex',
            color: '#fff'
          }}>
            <Type size={24} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 700, margin: 0 }}>NLP & Text Analytics</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
              Text analysis, sentiment diagnostics, D3 charts & text cleaning pipeline
            </p>
          </div>
        </div>

        {/* Column Select */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Select Text Column:</span>
          <select
            value={selectedColumn}
            onChange={(e) => setSelectedColumn(e.target.value)}
            style={{
              background: 'var(--bg-card-hover)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              padding: '0.5rem 1rem',
              borderRadius: 'var(--radius-md)',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            {textColumns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <button
            onClick={() => runAnalysis(selectedColumn)}
            className="btn btn-secondary"
            disabled={loading}
            style={{ padding: '0.5rem 0.85rem' }}
          >
            <RefreshCw size={16} className={loading ? 'spin' : ''} />
          </button>
        </div>
      </div>

      {message && (
        <div style={{
          background: message.includes('successfully') ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          border: `1px solid ${message.includes('successfully') ? '#10b981' : '#ef4444'}`,
          borderRadius: 'var(--radius-md)',
          padding: '0.75rem 1rem',
          color: 'var(--text-primary)',
          fontSize: '0.9rem',
          fontWeight: 500
        }}>
          {message}
        </div>
      )}

      {/* Metrics Bar */}
      {analysisData && analysisData.metrics && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '1rem'
        }}>
          <div className="stat-card">
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Total Rows Analyzed</span>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {analysisData.metrics.total_rows?.toLocaleString()}
            </div>
          </div>
          <div className="stat-card">
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Avg Word Count</span>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#8b5cf6' }}>
              {analysisData.metrics.avg_word_count}
            </div>
          </div>
          <div className="stat-card">
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Total Words</span>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#ec4899' }}>
              {analysisData.metrics.total_words?.toLocaleString()}
            </div>
          </div>
          <div className="stat-card">
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Vocabulary Richness</span>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>
              {(analysisData.metrics.vocabulary_richness * 100).toFixed(1)}%
            </div>
          </div>
          <div className="stat-card">
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Sentiment Index</span>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: analysisData.avg_sentiment_score >= 0 ? '#10b981' : '#ef4444' }}>
              {analysisData.avg_sentiment_score > 0 ? `+${analysisData.avg_sentiment_score}` : analysisData.avg_sentiment_score}
            </div>
          </div>
        </div>
      )}

      {/* Main Workspace Layout: Charts on Left, Text Cleaner on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem' }}>

        {/* Left Column: D3 Interactive Visualizations */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-xl)',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem'
        }}>
          {/* View Selector Buttons */}
          <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
            <button
              onClick={() => setActiveView('cloud')}
              className={`btn ${activeView === 'cloud' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.85rem' }}
            >
              <Sparkles size={16} /> D3 Word Cloud
            </button>
            <button
              onClick={() => setActiveView('network')}
              className={`btn ${activeView === 'network' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.85rem' }}
            >
              <Network size={16} /> D3 Co-Occurrence Net
            </button>
            <button
              onClick={() => setActiveView('treemap')}
              className={`btn ${activeView === 'treemap' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.85rem' }}
            >
              <Database size={16} /> D3 Treemap
            </button>
            <button
              onClick={() => setActiveView('ngrams')}
              className={`btn ${activeView === 'ngrams' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.85rem' }}
            >
              <Filter size={16} /> N-Grams
            </button>
          </div>

          {/* D3 Chart Render Area */}
          {loading ? (
            <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
              Analyzing text column...
            </div>
          ) : analysisData ? (
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              {activeView === 'cloud' && (
                <D3WordCloud data={analysisData.wordcloud} width={620} height={380} />
              )}
              {activeView === 'network' && (
                <D3ForceGraph data={analysisData.network} width={620} height={380} />
              )}
              {activeView === 'treemap' && (
                <D3TreeMap data={analysisData.wordcloud.slice(0, 25)} width={620} height={380} />
              )}
              {activeView === 'ngrams' && (
                <div style={{ width: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <D3BarChart data={analysisData.bigrams} title="Top Bigrams (2 Words)" width={300} height={350} />
                  <D3BarChart data={analysisData.trigrams} title="Top Trigrams (3 Words)" width={300} height={350} />
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
              Select a valid text column to view NLP diagnostics.
            </div>
          )}

          {/* Sentiment Summary Card */}
          {analysisData && analysisData.sentiment && (
            <div style={{
              background: 'rgba(15, 23, 42, 0.5)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              padding: '1rem 1.25rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-around'
            }}>
              <div style={{ textAlign: 'center' }}>
                <span style={{ color: '#10b981', fontWeight: 700, fontSize: '1.2rem' }}>{analysisData.sentiment.positive}</span>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Positive Sentiments</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <span style={{ color: '#94a3b8', fontWeight: 700, fontSize: '1.2rem' }}>{analysisData.sentiment.neutral}</span>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Neutral Sentiments</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <span style={{ color: '#ef4444', fontWeight: 700, fontSize: '1.2rem' }}>{analysisData.sentiment.negative}</span>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Negative Sentiments</div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Text Cleaning Pipeline Toolbox */}
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-xl)',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
            <Scissors size={18} color="#ec4899" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>Text Cleaning Toolbox</h3>
          </div>

          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0 }}>
            Configure transformations to sanitize '{selectedColumn}':
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', fontSize: '0.85rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanOptions.lower}
                onChange={e => setCleanOptions({ ...cleanOptions, lower: e.target.checked })}
              />
              Lowercase Text
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanOptions.remove_punct}
                onChange={e => setCleanOptions({ ...cleanOptions, remove_punct: e.target.checked })}
              />
              Remove Punctuation & Special Chars
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanOptions.remove_stopwords}
                onChange={e => setCleanOptions({ ...cleanOptions, remove_stopwords: e.target.checked })}
              />
              Strip English Stopwords
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanOptions.remove_digits}
                onChange={e => setCleanOptions({ ...cleanOptions, remove_digits: e.target.checked })}
              />
              Strip Numeric Digits
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanOptions.remove_urls}
                onChange={e => setCleanOptions({ ...cleanOptions, remove_urls: e.target.checked })}
              />
              Strip URLs (http/https)
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanOptions.remove_html}
                onChange={e => setCleanOptions({ ...cleanOptions, remove_html: e.target.checked })}
              />
              Strip HTML Tags
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={cleanOptions.normalize_space}
                onChange={e => setCleanOptions({ ...cleanOptions, normalize_space: e.target.checked })}
              />
              Normalize Whitespace & Newlines
            </label>

            {/* Custom Regex Pattern */}
            <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Custom Regex Replacement:</span>
              <input
                type="text"
                placeholder="Regex Pattern (e.g. \b[0-9]{3}\b)"
                value={cleanOptions.custom_regex}
                onChange={e => setCleanOptions({ ...cleanOptions, custom_regex: e.target.value })}
                style={{
                  background: 'var(--bg-card-hover)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.4rem 0.6rem',
                  color: 'var(--text-primary)',
                  fontSize: '0.8rem'
                }}
              />
              <input
                type="text"
                placeholder="Replace with..."
                value={cleanOptions.custom_replace}
                onChange={e => setCleanOptions({ ...cleanOptions, custom_replace: e.target.value })}
                style={{
                  background: 'var(--bg-card-hover)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.4rem 0.6rem',
                  color: 'var(--text-primary)',
                  fontSize: '0.8rem'
                }}
              />
            </div>
          </div>

          <button
            onClick={handleApplyClean}
            disabled={cleaningLoading || !selectedColumn}
            className="btn btn-primary"
            style={{ width: '100%', marginTop: '0.5rem', display: 'flex', justifyContent: 'center', gap: '0.5rem' }}
          >
            {cleaningLoading ? 'Cleaning Dataset...' : <><Check size={16} /> Apply Text Cleaning</>}
          </button>

          {/* Extracted Entities List */}
          {analysisData && analysisData.entities && (
            <div style={{
              marginTop: '1rem',
              borderTop: '1px solid var(--border-color)',
              paddingTop: '0.75rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem'
            }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-secondary)' }}>Detected Pattern Entities:</span>
              {analysisData.entities.emails?.length > 0 && (
                <div style={{ fontSize: '0.75rem', color: '#3b82f6' }}>
                  Emails ({analysisData.entities.emails.length}): {analysisData.entities.emails.slice(0, 3).join(', ')}
                </div>
              )}
              {analysisData.entities.urls?.length > 0 && (
                <div style={{ fontSize: '0.75rem', color: '#10b981' }}>
                  URLs ({analysisData.entities.urls.length}): {analysisData.entities.urls.slice(0, 2).join(', ')}
                </div>
              )}
            </div>
          )}

          {analysisData?.tfidf?.length > 0 && (
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-secondary)' }}>Top TF-IDF terms</span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.5rem' }}>
                {analysisData.tfidf.slice(0, 12).map(term => (
                  <span key={term.text} title={`Mean TF-IDF: ${term.value}`} style={{ fontSize: '0.72rem', padding: '0.2rem 0.45rem', borderRadius: '999px', background: 'rgba(139,92,246,0.14)', color: '#c4b5fd' }}>{term.text}</span>
                ))}
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
