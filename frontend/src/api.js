const API_BASE = window.location.port === '5173' ? 'http://localhost:5000' : '';

export async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.error || `HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

export function getExportModelUrl(keyPrefix, modelName) {
  return `${API_BASE}/api/export-model?key_prefix=${encodeURIComponent(keyPrefix)}&model_name=${encodeURIComponent(modelName)}`;
}
