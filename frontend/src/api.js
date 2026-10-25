const API_BASE = window.location.port === '5173' ? 'http://localhost:5000' : '';

export async function request(endpoint, options = {}, data = null) {
  let fetchOptions = {};
  if (typeof options === 'string') {
    fetchOptions = {
      method: options,
      body: data ? JSON.stringify(data) : undefined
    };
  } else {
    fetchOptions = options;
  }

  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...fetchOptions,
    headers: {
      'Content-Type': 'application/json',
      ...fetchOptions.headers,
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
