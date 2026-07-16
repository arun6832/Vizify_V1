import React, { useState, useEffect } from 'react';
import { Cpu, Play, Download, HelpCircle, RefreshCw, BarChart2 } from 'lucide-react';
import { request, getExportModelUrl } from '../api';
import Plot from './Plot';

export default function MLStudio({ dataInfo }) {
  const [step, setStep] = useState(1);
  const [problemType, setProblemType] = useState('Regression'); // 'Regression' or 'Classification'
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [missingStrategy, setMissingStrategy] = useState('Fill with mean (numeric) / mode (categorical)');
  const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);
  const [testSize, setTestSize] = useState(20);
  const [randomSeed, setRandomSeed] = useState(42);
  const [scaleFeatures, setScaleFeatures] = useState(true);
  const [tuneHyperparameters, setTuneHyperparameters] = useState(false);

  // Training results state
  const [training, setTraining] = useState(false);
  const [results, setResults] = useState(null);
  const [bestModel, setBestModel] = useState('');
  const [trainedModelsList, setTrainedModelsList] = useState([]);
  const [modelToExport, setModelToExport] = useState('');
  
  // What-if simulator state
  const [simulatorModel, setSimulatorModel] = useState('');
  const [simulatorInputs, setSimulatorInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  const keyPrefix = 'ml_studio_page';
  const isRegression = problemType === 'Regression';

  // Get available algorithms based on problem type
  const algorithmsList = isRegression
    ? ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Machine", "K-Nearest Neighbors"]
    : ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine", "K-Nearest Neighbors"];

  // Set default selection when target column changes
  useEffect(() => {
    if (dataInfo.numeric_cols?.length > 0) {
      setTargetColumn(dataInfo.numeric_cols[0]);
    }
  }, [dataInfo]);

  // Set default algorithms list
  useEffect(() => {
    setSelectedAlgorithms([algorithmsList[0]]);
  }, [problemType]);

  const handleFeatureToggle = (feat) => {
    setSelectedFeatures(prev =>
      prev.includes(feat) ? prev.filter(f => f !== feat) : [...prev, feat]
    );
  };

  const handleSelectAllFeatures = () => {
    const allFeats = (dataInfo.columns || []).filter(c => c !== targetColumn);
    setSelectedFeatures(allFeats);
  };

  const handleClearAllFeatures = () => {
    setSelectedFeatures([]);
  };

  const triggerTraining = async () => {
    setTraining(true);
    setResults(null);
    setPredictionResult(null);

    try {
      const response = await request('/api/train-ml', {
        method: 'POST',
        body: JSON.stringify({
          keyPrefix,
          problemType,
          features: selectedFeatures,
          target: targetColumn,
          missingStrategy,
          algorithms: selectedAlgorithms,
          testSize: testSize / 100,
          seed: randomSeed,
          scale: scaleFeatures,
          tune: tuneHyperparameters
        })
      });

      setResults(response.results);
      setBestModel(response.best_model);
      const models = Object.keys(response.results);
      setTrainedModelsList(models);
      setModelToExport(models[0]);
      setSimulatorModel(response.best_model);

      // Initialize simulator inputs with feature averages/min values
      const initialInputs = {};
      response.features_list.forEach(feat => {
        // Find if feature corresponds to a column name in health
        const colHealth = dataInfo.columns_health?.find(c => c.name === feat);
        const isNumeric = colHealth ? colHealth.type.includes('int') || colHealth.type.includes('float') : true;

        if (isNumeric) {
          const stats = dataInfo.summary_stats;
          initialInputs[feat] = stats?.mean?.[feat] !== undefined ? stats.mean[feat] : 0;
        } else {
          initialInputs[feat] = "0";
        }
      });
      setSimulatorInputs(initialInputs);
      
      // Go to results step (or display in layout)
      setStep(4);
    } catch (err) {
      alert(`Training Error: ${err.message}`);
    } finally {
      setTraining(false);
    }
  };

  // Run What-If Prediction
  const runSimulatorPrediction = async (inputs = simulatorInputs, modelName = simulatorModel) => {
    if (!modelName || Object.keys(inputs).length === 0) return;
    setPredicting(true);

    try {
      const response = await request('/api/predict-live', {
        method: 'POST',
        body: JSON.stringify({
          keyPrefix,
          modelName,
          inputs
        })
      });
      setPredictionResult(response);
    } catch (err) {
      console.error(err);
    } finally {
      setPredicting(false);
    }
  };

  // Run prediction when simulator values change
  useEffect(() => {
    if (results) {
      runSimulatorPrediction(simulatorInputs, simulatorModel);
    }
  }, [simulatorInputs, simulatorModel]);

  const handleSimulatorInputChange = (feat, val) => {
    setSimulatorInputs(prev => ({
      ...prev,
      [feat]: parseFloat(val)
    }));
  };

  return (
    <div className="fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700 }} className="glow-text-primary">
          🤖 No-Code Machine Learning Studio
        </h2>
        <p style={{ fontSize: '0.9rem', marginTop: '0.25rem' }}>
          Train machine learning models in seconds, compare evaluation metrics, and explore predictive sliders.
        </p>
      </div>

      {/* Stepper Wizard Navigation */}
      <div className="stepper" style={{ maxWidth: '600px', margin: '0 auto 2.5rem auto' }}>
        <div 
          className={`step-node ${step === 1 ? 'active' : step > 1 ? 'completed' : ''}`}
          onClick={() => setStep(1)}
          style={{ cursor: 'pointer' }}
        >
          1
        </div>
        <div 
          className={`step-node ${step === 2 ? 'active' : step > 2 ? 'completed' : ''}`}
          onClick={() => fileLoaded && selectedFeatures.length > 0 && setStep(2)}
          style={{ cursor: selectedFeatures.length > 0 ? 'pointer' : 'not-allowed' }}
        >
          2
        </div>
        <div 
          className={`step-node ${step === 3 ? 'active' : step > 3 ? 'completed' : ''}`}
          onClick={() => fileLoaded && selectedFeatures.length > 0 && setStep(3)}
          style={{ cursor: selectedFeatures.length > 0 ? 'pointer' : 'not-allowed' }}
        >
          3
        </div>
        <div 
          className={`step-node ${step === 4 ? 'active' : ''}`}
          onClick={() => results && setStep(4)}
          style={{ cursor: results ? 'pointer' : 'not-allowed' }}
        >
          4
        </div>
      </div>

      {/* Step Contents */}
      {step === 1 && (
        <div className="glass-card fade-in" style={{ maxWidth: '700px', margin: '0 auto' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            Step 1: Choose ML Goal
          </h3>
          
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.75rem' }}>Problem Type</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div 
                className="glass-card" 
                style={{ 
                  borderColor: problemType === 'Regression' ? 'var(--accent-primary)' : 'var(--border-color)', 
                  cursor: 'pointer',
                  background: problemType === 'Regression' ? 'rgba(139,92,246,0.05)' : 'var(--bg-card)'
                }}
                onClick={() => setProblemType('Regression')}
              >
                <div style={{ fontWeight: 600, fontSize: '1rem', color: problemType === 'Regression' ? 'var(--accent-primary)' : 'var(--text-primary)' }}>
                  Regression
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Predict a continuous numerical field (e.g. Sales, Price, Age).
                </div>
              </div>

              <div 
                className="glass-card" 
                style={{ 
                  borderColor: problemType === 'Classification' ? 'var(--accent-primary)' : 'var(--border-color)', 
                  cursor: 'pointer',
                  background: problemType === 'Classification' ? 'rgba(139,92,246,0.05)' : 'var(--bg-card)'
                }}
                onClick={() => setProblemType('Classification')}
              >
                <div style={{ fontWeight: 600, fontSize: '1rem', color: problemType === 'Classification' ? 'var(--accent-primary)' : 'var(--text-primary)' }}>
                  Classification
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Predict a discrete category class (e.g. Survived vs Died, Default vs Active).
                </div>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button className="btn btn-primary" onClick={() => setStep(2)}>
              Next Step
            </button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="glass-card fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            Step 2: Select Variables
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '2rem', marginBottom: '1.5rem' }}>
            {/* Features multiselect */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <label style={{ fontWeight: 600, margin: 0 }}>Input Features (X)</label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button className="btn" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }} onClick={handleSelectAllFeatures}>All</button>
                  <button className="btn" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }} onClick={handleClearAllFeatures}>Clear</button>
                </div>
              </div>
              <div style={{ 
                maxHeight: '220px', 
                overflowY: 'auto', 
                border: '1px solid var(--border-color)', 
                borderRadius: 'var(--radius-md)', 
                padding: '0.75rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.5rem',
                background: 'rgba(15,23,42,0.3)'
              }}>
                {(dataInfo.columns || []).map(col => {
                  const isTarget = col === targetColumn;
                  return (
                    <label 
                      key={col} 
                      style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '0.5rem', 
                        cursor: isTarget ? 'not-allowed' : 'pointer', 
                        margin: 0,
                        opacity: isTarget ? 0.4 : 1,
                        background: selectedFeatures.includes(col) ? 'rgba(139,92,246,0.1)' : 'transparent',
                        padding: '0.25rem 0.5rem',
                        borderRadius: 'var(--radius-sm)'
                      }}
                    >
                      <input 
                        type="checkbox" 
                        disabled={isTarget}
                        checked={selectedFeatures.includes(col)}
                        onChange={() => handleFeatureToggle(col)}
                      />
                      <span>{col}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Target select */}
            <div>
              <label style={{ fontWeight: 600, marginBottom: '0.75rem' }}>Predict Target (Y)</label>
              <select 
                value={targetColumn} 
                onChange={(e) => {
                  const newTarget = e.target.value;
                  setTargetColumn(newTarget);
                  // Remove from features if selected
                  setSelectedFeatures(prev => prev.filter(f => f !== newTarget));
                }}
                style={{ marginBottom: '1rem' }}
              >
                {/* For regression, suggest numeric fields; for classification, show all */}
                {isRegression
                  ? dataInfo.numeric_cols?.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))
                  : dataInfo.columns?.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))
                }
              </select>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.02)', padding: '0.75rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                {isRegression ? (
                  <span>📊 <strong>Suggestion</strong>: A target variable representing quantitative amounts or scores.</span>
                ) : (
                  <span>🏷️ <strong>Suggestion</strong>: A categorical text column or bin representing status or groups.</span>
                )}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button className="btn" onClick={() => setStep(1)}>
              Back
            </button>
            <button 
              className="btn btn-primary" 
              onClick={() => setStep(3)}
              disabled={selectedFeatures.length === 0}
            >
              Next Step
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="glass-card fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            Step 3: Preprocessing & Algorithms
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '1.5rem' }}>
            {/* Left side: parameters */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div>
                <label style={{ fontWeight: 600 }}>Missing Values Strategy</label>
                <select value={missingStrategy} onChange={(e) => setMissingStrategy(e.target.value)}>
                  <option value="Drop rows with missing values">Drop rows with missing values</option>
                  <option value="Fill with mean (numeric) / mode (categorical)">Fill with mean (numeric) / mode (categorical)</option>
                  <option value="Fill with median (numeric) / mode (categorical)">Fill with median (numeric) / mode (categorical)</option>
                  <option value="KNN Imputation (numeric only)">KNN Imputation (numeric only)</option>
                </select>
              </div>

              <div>
                <label style={{ fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}>
                  <span>Test Split Size</span>
                  <span style={{ color: 'var(--accent-primary)' }}>{testSize}%</span>
                </label>
                <input 
                  type="range" 
                  min="10" 
                  max="40" 
                  value={testSize} 
                  onChange={(e) => setTestSize(parseInt(e.target.value))}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={{ fontWeight: 600 }}>Random Seed</label>
                  <input 
                    type="text" 
                    value={randomSeed} 
                    onChange={(e) => setRandomSeed(e.target.value.replace(/\D/g, ''))} 
                  />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '0.5rem', paddingTop: '1.25rem' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0, cursor: 'pointer' }}>
                    <input 
                      type="checkbox" 
                      checked={scaleFeatures} 
                      onChange={() => setScaleFeatures(!scaleFeatures)} 
                    />
                    <span>Scale Features</span>
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0, cursor: 'pointer' }}>
                    <input 
                      type="checkbox" 
                      checked={tuneHyperparameters} 
                      onChange={() => setTuneHyperparameters(!tuneHyperparameters)} 
                    />
                    <span>Hyperparameters</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Right side: algorithms */}
            <div>
              <label style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Select Algorithms</label>
              <div style={{ 
                border: '1px solid var(--border-color)', 
                borderRadius: 'var(--radius-md)', 
                padding: '0.75rem',
                background: 'rgba(15,23,42,0.3)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.5rem'
              }}>
                {algorithmsList.map(alg => (
                  <label key={alg} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', margin: 0 }}>
                    <input 
                      type="checkbox" 
                      checked={selectedAlgorithms.includes(alg)}
                      onChange={() => {
                        setSelectedAlgorithms(prev => 
                          prev.includes(alg) 
                            ? prev.filter(a => a !== alg) 
                            : [...prev, alg]
                        );
                      }}
                    />
                    <span>{alg}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <button className="btn" onClick={() => setStep(2)} disabled={training}>
              Back
            </button>
            <button 
              className="btn btn-primary" 
              onClick={triggerTraining}
              disabled={training || selectedAlgorithms.length === 0}
            >
              {training ? (
                <>
                  <span className="spinner" style={{ width: '14px', height: '14px', borderWidth: '2px' }}></span>
                  Training Models...
                </>
              ) : (
                <>
                  <Play size={14} /> Train Models
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Results Display */}
      {step === 4 && results && (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          {/* Summary Metric Table */}
          <div className="glass-card">
            <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              🏆 Model Performance Summary
            </h3>
            
            <div style={{ background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.15)', borderRadius: 'var(--radius-lg)', padding: '1rem', marginBottom: '1.25rem', fontSize: '0.95rem' }}>
              👑 Best Performing Model: <strong>{bestModel}</strong> (Evaluation metric optimized on test partition)
            </div>

            <table>
              <thead>
                <tr>
                  <th>Model Name</th>
                  {isRegression ? (
                    <>
                      <th>RMSE (Error)</th>
                      <th>R² Score</th>
                    </>
                  ) : (
                    <th>Accuracy</th>
                  )}
                  <th>Training Duration</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(results).map(name => {
                  const isBest = name === bestModel;
                  return (
                    <tr key={name} style={{ background: isBest ? 'rgba(255,255,255,0.02)' : 'transparent' }}>
                      <td style={{ fontWeight: 600, color: isBest ? 'var(--accent-primary)' : 'var(--text-primary)' }}>
                        {name} {isBest && '⭐'}
                      </td>
                      {isRegression ? (
                        <>
                          <td>{results[name].RMSE.toFixed(4)}</td>
                          <td>{results[name].R2.toFixed(4)}</td>
                        </>
                      ) : (
                        <td>{(results[name].Accuracy * 100).toFixed(2)}%</td>
                      )}
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                        {results[name]['Training Time (s)'].toFixed(2)}s
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Charts Row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem' }}>
            {/* Performance Comparison Bar Plot */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1.15rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart2 size={18} /> Model Comparison
              </h3>
              <Plot 
                data={[{
                  x: Object.keys(results),
                  y: Object.keys(results).map(k => isRegression ? results[k].R2 : results[k].Accuracy),
                  type: 'bar',
                  marker: {
                    color: Object.keys(results).map(k => k === bestModel ? '#8b5cf6' : '#3b82f6')
                  }
                }]}
                layout={{
                  title: isRegression ? 'Model R² Comparison (Higher is Better)' : 'Model Accuracy Comparison (Higher is Better)',
                  yaxis: { title: isRegression ? 'R² Score' : 'Accuracy' },
                  margin: { t: 40, r: 20, b: 60, l: 50 }
                }}
              />
            </div>

            {/* Evaluation Details Plot */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1.15rem', marginBottom: '1rem' }}>
                {isRegression ? `Predictions vs Actual - ${bestModel}` : `Confusion Matrix - ${bestModel}`}
              </h3>
              {isRegression ? (
                <Plot 
                  data={[
                    {
                      x: Array.from(results[bestModel].actual),
                      y: Array.from(results[bestModel].predictions),
                      mode: 'markers',
                      type: 'scatter',
                      name: 'Predictions',
                      marker: { color: '#8b5cf6', opacity: 0.6 }
                    },
                    {
                      x: [Math.min(...results[bestModel].actual), Math.max(...results[bestModel].actual)],
                      y: [Math.min(...results[bestModel].actual), Math.max(...results[bestModel].actual)],
                      mode: 'lines',
                      type: 'scatter',
                      name: 'Ideal Line',
                      line: { color: '#ef4444', dash: 'dash' }
                    }
                  ]}
                  layout={{
                    xaxis: { title: 'Actual Values' },
                    yaxis: { title: 'Predicted Values' },
                    margin: { t: 40, r: 20, b: 40, l: 50 }
                  }}
                />
              ) : (
                // Confusion matrix rendering via plotly heatmap
                <Plot 
                  data={[{
                    z: results[bestModel].confusion_matrix,
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    text: results[bestModel].confusion_matrix.map(row => row.map(val => String(val))),
                    texttemplate: "%{text}",
                    showscale: true
                  }]}
                  layout={{
                    xaxis: { title: 'Predicted Classes' },
                    yaxis: { title: 'Actual Classes' },
                    margin: { t: 40, r: 20, b: 40, l: 50 }
                  }}
                />
              )}
            </div>
          </div>

          {/* Feature Importance best model */}
          {results[bestModel].feature_importances && (
            <div className="glass-card">
              <h3 style={{ fontSize: '1.15rem', marginBottom: '1rem' }}>
                🔥 Top Feature Importances ({bestModel})
              </h3>
              <Plot 
                data={[{
                  y: results[bestModel].feature_importances.map(f => f.feature),
                  x: results[bestModel].feature_importances.map(f => f.importance),
                  type: 'bar',
                  orientation: 'h',
                  marker: { color: '#8b5cf6' }
                }]}
                layout={{
                  title: 'Feature Importance Distribution',
                  xaxis: { title: 'Importance weight' },
                  margin: { t: 40, r: 20, b: 40, l: 120 }
                }}
              />
            </div>
          )}

          {/* Model download */}
          <div className="glass-card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem', alignItems: 'start' }}>
            <div>
              <h3 style={{ fontSize: '1.15rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Download size={18} /> Export Model Package
              </h3>
              <p style={{ fontSize: '0.85rem', marginBottom: '1rem' }}>
                Save your trained model, feature scaler, and mapping classes as a `.pkl` package for standalone deployment.
              </p>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <select 
                  value={modelToExport} 
                  onChange={(e) => setModelToExport(e.target.value)}
                  style={{ flexGrow: 1 }}
                >
                  {trainedModelsList.map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
                <a 
                  href={getExportModelUrl(keyPrefix, modelToExport)}
                  className="btn btn-primary"
                  style={{ textDecoration: 'none' }}
                >
                  Download Model
                </a>
              </div>
            </div>

            <details>
              <summary style={{ cursor: 'pointer', color: 'var(--accent-secondary)', fontSize: '0.85rem', userSelect: 'none', fontWeight: 600 }}>
                📖 How to load this model in Python
              </summary>
              <div style={{
                marginTop: '0.5rem',
                background: 'rgba(7,10,18,0.7)',
                padding: '0.75rem',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                fontSize: '0.75rem',
                fontFamily: 'monospace',
                overflowX: 'auto',
                whiteSpace: 'pre'
              }}>
{`import pickle
import pandas as pd

# 1. Load the model package
with open('vizify_${modelToExport.toLowerCase().replace(/\s+/g, '_')}_model.pkl', 'rb') as f:
    package = pickle.load(f)

model = package['model']
scaler = package['scaler']
features = package['features']

# 2. Make predictions on a dataframe
# inputs_df = pd.DataFrame([your_data])
if scaler:
    X_scaled = scaler.transform(inputs_df[features])
    preds = model.predict(X_scaled)
else:
    preds = model.predict(inputs_df[features])

print(preds)`}
              </div>
            </details>
          </div>

          {/* Live Simulator Playground */}
          <div className="glass-card">
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              🎮 What-If Scenario Simulator
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              Modify inputs in real-time to watch predictions shift instantly on the model.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
              {/* Sliders panel */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxHeight: '400px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <label style={{ margin: 0, fontWeight: 600 }}>Active Model:</label>
                  <select 
                    value={simulatorModel} 
                    onChange={(e) => setSimulatorModel(e.target.value)}
                    style={{ padding: '0.35rem 0.5rem', fontSize: '0.85rem', width: 'auto' }}
                  >
                    {trainedModelsList.map(name => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                </div>

                {Object.keys(simulatorInputs).map(feat => {
                  // Find range if it exists in dataInfo
                  const range = dataInfo.numeric_ranges?.[feat];
                  const val = simulatorInputs[feat];

                  if (range) {
                    return (
                      <div key={feat} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                          <span style={{ fontWeight: 500 }}>{feat}</span>
                          <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>
                            {typeof val === 'number' ? val.toFixed(2) : val}
                          </span>
                        </div>
                        <input 
                          type="range" 
                          min={range[0]} 
                          max={range[1]} 
                          step={(range[1] - range[0]) / 100}
                          value={val}
                          onChange={(e) => handleSimulatorInputChange(feat, e.target.value)}
                        />
                      </div>
                    );
                  } else {
                    return (
                      <div key={feat}>
                        <label>{feat}</label>
                        <input 
                          type="text" 
                          value={val} 
                          onChange={(e) => handleSimulatorInputChange(feat, e.target.value)} 
                        />
                      </div>
                    );
                  }
                })}
              </div>

              {/* Prediction result card */}
              <div style={{ 
                background: 'rgba(255,255,255,0.02)', 
                border: '1px solid var(--border-color)', 
                borderRadius: 'var(--radius-xl)', 
                padding: '2rem', 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600 }}>
                  Real-time Prediction
                </div>
                
                {predicting ? (
                  <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                    <div className="spinner"></div>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Running inference...</span>
                  </div>
                ) : predictionResult ? (
                  <div style={{ marginTop: '1.5rem' }}>
                    <div style={{ 
                      fontSize: '3rem', 
                      fontFamily: 'var(--font-title)', 
                      fontWeight: 800,
                      color: isRegression ? 'var(--accent-secondary)' : 'var(--accent-green)',
                      textShadow: isRegression ? '0 0 40px var(--glow-blue)' : '0 0 40px rgba(16,185,129,0.2)'
                    }}>
                      {isRegression 
                        ? predictionResult.prediction.toFixed(4)
                        : predictionResult.prediction_label || predictionResult.prediction
                      }
                    </div>
                    {predictionResult.confidence !== undefined && (
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.5rem', fontWeight: 500 }}>
                        Confidence Probability: {(predictionResult.confidence * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-muted)', marginTop: '1.5rem', fontSize: '0.9rem', fontStyle: 'italic' }}>
                    Modify sliders to calculate predicted output
                  </div>
                )}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-start', marginTop: '1rem' }}>
            <button className="btn" onClick={() => setStep(3)}>
              Back to Training Configuration
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
