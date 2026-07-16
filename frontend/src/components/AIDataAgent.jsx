import React, { useState, useRef, useEffect } from 'react';
import { Send, Terminal, Code, AlertCircle } from 'lucide-react';
import { request } from '../api';
import Plot from './Plot';

export default function AIDataAgent({ apiKey, selectedModel }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    const prompt = input.trim();
    if (!prompt || loading) return;

    setInput('');
    // Add user message
    const userMsg = { role: 'user', content: prompt };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const response = await request('/api/agent-chat', {
        method: 'POST',
        body: JSON.stringify({
          question: prompt,
          history: [...messages, userMsg],
          apiKey,
          model: selectedModel
        })
      });

      // Add assistant response
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: response.content,
          code: response.code,
          stdout: response.stdout,
          error: response.error,
          fig: response.fig // Plotly json data if available
        }
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${err.message}`,
          error: err.message
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 8rem)' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700 }} className="glow-text-green">
          💬 AI Data Agent
        </h2>
        <p style={{ fontSize: '0.9rem', marginTop: '0.25rem' }}>
          Conversational data analyst. Ask questions in natural language, and the agent will write and run Python code to analyze your data.
        </p>
      </div>

      {!apiKey && (
        <div style={{
          background: 'rgba(239,68,68,0.06)',
          border: '1px solid rgba(239,68,68,0.15)',
          borderRadius: 'var(--radius-lg)',
          padding: '1rem',
          marginBottom: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          color: '#fca5a5',
          fontSize: '0.9rem'
        }}>
          <AlertCircle size={20} />
          <span>Please configure your Gemini API Key in the sidebar controls to activate the AI Data Agent.</span>
        </div>
      )}

      {/* Chat Area */}
      <div className="chat-container" style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', height: 'auto' }}>
        <div className="chat-messages" style={{ flexGrow: 1, padding: '1.5rem' }}>
          {messages.length === 0 && (
            <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-muted)', maxWidth: '400px' }}>
              <Terminal size={36} style={{ margin: '0 auto 0.75rem auto', opacity: 0.6 }} />
              <h4 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>Start Conversing</h4>
              <p style={{ fontSize: '0.85rem' }}>
                Ask me to calculate stats, plot metrics, or search trends (e.g. "Draw a scatter plot of age vs fare colored by sex" or "What is the average fare?").
              </p>
            </div>
          )}

          {messages.map((msg, idx) => {
            const isUser = msg.role === 'user';
            return (
              <div 
                key={idx} 
                className={`chat-bubble ${isUser ? 'user' : 'assistant'}`}
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '0.75rem',
                  padding: '1rem 1.25rem'
                }}
              >
                {/* Text explanation */}
                <div style={{ whiteSpace: 'pre-line', fontSize: '0.95rem' }}>
                  {msg.content}
                </div>

                {/* Inline Plotly Figure */}
                {msg.fig && (
                  <div style={{ background: 'rgba(0,0,0,0.15)', padding: '0.5rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', marginTop: '0.5rem' }}>
                    <Plot 
                      data={msg.fig.data} 
                      layout={msg.fig.layout} 
                      style={{ height: '350px' }} 
                    />
                  </div>
                )}

                {/* Execution Errors */}
                {msg.error && (
                  <div style={{ 
                    background: 'rgba(239,68,68,0.08)', 
                    border: '1px solid rgba(239,68,68,0.2)', 
                    color: '#fecaca', 
                    borderRadius: 'var(--radius-md)', 
                    padding: '0.75rem', 
                    fontSize: '0.85rem',
                    fontFamily: 'monospace'
                  }}>
                    ⚠️ Execution Error: {msg.error}
                  </div>
                )}

                {/* Console prints */}
                {msg.stdout && msg.stdout.trim() && (
                  <details style={{ width: '100%' }}>
                    <summary style={{ cursor: 'pointer', fontSize: '0.75rem', color: 'var(--text-muted)', userSelect: 'none', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                      <Terminal size={12} /> View Terminal Stdout
                    </summary>
                    <pre style={{
                      marginTop: '0.35rem',
                      background: 'rgba(7,10,18,0.8)',
                      padding: '0.6rem',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-color)',
                      fontSize: '0.8rem',
                      fontFamily: 'monospace',
                      overflowX: 'auto',
                      color: '#a7f3d0'
                    }}>
                      {msg.stdout}
                    </pre>
                  </details>
                )}

                {/* Generated Python Code */}
                {msg.code && (
                  <details style={{ width: '100%' }}>
                    <summary style={{ cursor: 'pointer', fontSize: '0.75rem', color: 'var(--text-muted)', userSelect: 'none', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                      <Code size={12} /> Show Python Source
                    </summary>
                    <pre style={{
                      marginTop: '0.35rem',
                      background: 'rgba(7,10,18,0.8)',
                      padding: '0.6rem',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-color)',
                      fontSize: '0.8rem',
                      fontFamily: 'monospace',
                      overflowX: 'auto',
                      color: '#c084fc'
                    }}>
                      {msg.code}
                    </pre>
                  </details>
                )}
              </div>
            );
          })}

          {loading && (
            <div className="chat-bubble assistant" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', padding: '0.85rem 1.1rem' }}>
              <span className="spinner" style={{ width: '14px', height: '14px', borderWidth: '2px' }}></span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>AI Data Agent is thinking & executing analysis...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input box */}
        <form onSubmit={handleSend} className="chat-input-area">
          <input
            type="text"
            placeholder={apiKey ? "Ask a question about the dataset..." : "Enter Gemini API Key in sidebar to ask questions"}
            value={input}
            disabled={!apiKey || loading}
            onChange={(e) => setInput(e.target.value)}
            style={{ flexGrow: 1 }}
          />
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={!apiKey || loading || !input.trim()}
          >
            <Send size={16} /> Send
          </button>
        </form>
      </div>
    </div>
  );
}
