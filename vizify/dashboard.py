import os
import sys
import io
import time
import socket
import re
import webbrowser
import pickle
import json
import traceback
import ipaddress
import socket
from urllib.parse import urlparse, urljoin
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

# Optional AI imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Optional ML imports
try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import KNNImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional PDF imports
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# Vizify Engine Imports
try:
    from vizify.engine import SQLiteDataEngine, NLPProcessor
    from vizify.config import get_api_key
except ImportError:
    from engine import SQLiteDataEngine, NLPProcessor
    from config import get_api_key

SQLITE_ENGINE = SQLiteDataEngine()

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for development cross-origin requests

# Global state
ACTIVE_DATA = {
    'df_raw': None,
    'df_cleaned': None,
    'filename': None,
}

def restore_active_data():
    """Restore lightweight dataset state; data rows remain in SQLite."""
    if SQLITE_ENGINE.has_dataset():
        ACTIVE_DATA['filename'] = SQLITE_ENGINE.filename

def get_active_dataframe():
    """Load rows from SQLite on demand instead of retaining the dataset globally."""
    if SQLITE_ENGINE.has_dataset():
        return SQLITE_ENGINE.get_dataframe(limit=None)
    return ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']

ACTIVE_MODELS = {}  # Stores trained models: { key_prefix: { model_name: { 'model': obj, 'scaler': obj, ... } } }


# --- Helper: Dynamic Free Port Finder ---
def find_free_port(start_port=5000):
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            port += 1

# --- Helper: Datetime coercer ---
def coerce_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == 'object':
            try:
                # Try parsing if most values look like dates
                parsed = pd.to_datetime(out[c], errors='raise')
                if parsed.notna().mean() > 0.8:
                    out[c] = parsed
            except Exception:
                pass
    return out

# --- Helper: Filter applier ---
def apply_filters(df, filters):
    if not filters:
        return df
    
    out = df.copy()
    
    # Categorical filters
    categorical = filters.get('categorical', {})
    for col, vals in categorical.items():
        if vals and col in out.columns:
            out = out[out[col].isin(vals)]
            
    # Numeric filters
    numeric = filters.get('numeric', {})
    for col, r in numeric.items():
        if r and col in out.columns and len(r) == 2:
            out = out[(out[col] >= r[0]) & (out[col] <= r[1])]
            
    # Datetime filters
    datetime_filters = filters.get('datetime', {})
    for col, r in datetime_filters.items():
        if r and col in out.columns:
            start = r.get('start')
            end = r.get('end')
            if start:
                out = out[pd.to_datetime(out[col]) >= pd.to_datetime(start)]
            if end:
                out = out[pd.to_datetime(out[col]) <= (pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
                
    return out

# --- Helper: Column Metadata Summarizer ---
def get_dataset_metadata(df, filename):
    df_coerced = coerce_datetime_cols(df)
    
    numeric_cols = df_coerced.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_coerced.select_dtypes(include=['object', 'category']).columns.tolist()
    time_cols = df_coerced.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    
    # Ranges for numeric
    numeric_ranges = {}
    for col in numeric_cols:
        col_min = df_coerced[col].min()
        col_max = df_coerced[col].max()
        numeric_ranges[col] = [
            float(col_min) if np.isfinite(col_min) else 0.0,
            float(col_max) if np.isfinite(col_max) else 0.0
        ]
        
    # Unique values for categories (cap at 20)
    categorical_values = {}
    for col in categorical_cols:
        categorical_values[col] = df_coerced[col].dropna().unique().tolist()[:20]

    # Health list
    columns_health = []
    for col in df_coerced.columns:
        null_count = int(df_coerced[col].isnull().sum())
        null_pct = (null_count / len(df_coerced) * 100) if len(df_coerced) > 0 else 0
        columns_health.append({
            'name': col,
            'type': str(df_coerced[col].dtype),
            'non_null': len(df_coerced) - null_count,
            'missing_pct': f"{null_pct:.1f}%"
        })

    # Summary stats
    summary_stats = df_coerced.describe(include='all').T.fillna('N/A').to_dict()

    return {
        'filename': filename,
        'shape': df_coerced.shape,
        'columns': df_coerced.columns.tolist(),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'time_cols': time_cols,
        'numeric_ranges': numeric_ranges,
        'categorical_values': categorical_values,
        'missing_count': int(df_coerced.isnull().sum().sum()),
        'missing_pct': float(df_coerced.isnull().sum().sum() / df_coerced.size * 100) if df_coerced.size > 0 else 0.0,
        'duplicate_count': int(df_coerced.duplicated().sum()),
        'columns_health': columns_health,
        'summary_stats': summary_stats
    }


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/init', methods=['GET'])
def api_init():
    restore_active_data()
    file_loaded = SQLITE_ENGINE.has_dataset() or ACTIVE_DATA['df_raw'] is not None
    info = {}
    if file_loaded:
        info = SQLITE_ENGINE.get_metadata() if SQLITE_ENGINE.has_dataset() else get_dataset_metadata(ACTIVE_DATA['df_raw'], ACTIVE_DATA['filename'])
        
    return jsonify({
        'api_key': get_api_key(),
        'file_loaded': file_loaded,
        'info': info
    })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        sanitized_filename = os.path.basename(file.filename)
        extension = os.path.splitext(sanitized_filename)[1].lower()
        if extension == '.csv':
            info = SQLITE_ENGINE.ingest_csv(file, filename=sanitized_filename)
        elif extension in ('.xlsx', '.xls', '.tsv', '.txt', '.json', '.jsonl', '.xml', '.pdf', '.parquet'):
            df = read_uploaded_dataframe(file, extension)
            info = SQLITE_ENGINE.ingest_dataframe(df, filename=sanitized_filename)
        else:
            return jsonify({'error': 'Supported files: CSV, TSV, TXT, Excel, JSON, XML, PDF, and Parquet'}), 400

        ACTIVE_DATA['filename'] = sanitized_filename
        
        return jsonify(info)
    except Exception as e:
        if isinstance(e, ImportError):
            return jsonify({'error': 'This file format needs an optional reader. Install it with `pip install vizify[files]`, then restart Vizify.'}), 400
        return jsonify({'error': str(e)}), 500

def read_uploaded_dataframe(file, extension):
    """Read supported local file formats into a table-shaped dataframe."""
    if extension in ('.xlsx', '.xls'):
        return pd.read_excel(file)
    if extension == '.tsv':
        return pd.read_csv(file, sep='\t')
    if extension == '.txt':
        content = file.read().decode('utf-8-sig', errors='replace')
        return pd.DataFrame({'text': content.splitlines()})
    if extension == '.jsonl':
        return pd.read_json(file, lines=True)
    if extension == '.json':
        try:
            return pd.read_json(file)
        except ValueError:
            file.seek(0)
            value = json.load(file)
            return pd.json_normalize(value if isinstance(value, list) else value.get('data', value))
    if extension == '.xml':
        return pd.read_xml(file)
    if extension == '.parquet':
        return pd.read_parquet(file)
    if extension == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF import needs pypdf. Install it with `pip install vizify[files]`.") from exc
        reader = PdfReader(file)
        return pd.DataFrame([{'page': i + 1, 'text': page.extract_text() or ''} for i, page in enumerate(reader.pages)])
    raise ValueError(f"Unsupported file type: {extension}")

def connector_install_hint(provider):
    extras = {
        'mysql': 'mysql', 'postgresql': 'postgres', 'sqlserver': 'sqlserver',
        'azuresql': 'sqlserver', 'oracle': 'oracle', 'ibmdb2': 'db2',
        'databricks': 'databricks', 'odbc': 'sqlserver'
    }
    extra = extras.get(provider, 'connections')
    return f"The {provider} driver is missing. Install it with `pip install 'vizify[{extra}]'`, then restart Vizify."

def require_public_https_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use a public HTTPS URL without embedded credentials.')
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise ValueError('The HTTPS host could not be resolved.') from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError('The URL must resolve only to public internet addresses.')
    return parsed

def get_public_response(url, headers=None, max_redirects=4):
    import requests
    current = url
    for _ in range(max_redirects + 1):
        require_public_https_url(current)
        response = requests.get(current, headers=headers or {}, timeout=(5, 30), allow_redirects=False)
        if response.is_redirect:
            location = response.headers.get('Location')
            if not location:
                raise ValueError('The source returned an invalid redirect.')
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        return response
    raise ValueError('The source redirected too many times.')

@app.route('/api/connect/database', methods=['POST'])
def api_connect_database():
    """Connect to MySQL or Databricks, import a read-only query into local SQLite."""
    body = request.json or {}
    provider = (body.get('provider') or '').lower()
    query = (body.get('query') or '').strip()
    if not query or not query.lower().startswith(('select ', 'with ')):
        return jsonify({'error': 'Provide a SELECT query to import data'}), 400
    try:
        from sqlalchemy.engine import URL
        source_names = {
            'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'sqlserver': 'SQL Server',
            'azuresql': 'Azure SQL', 'oracle': 'Oracle', 'ibmdb2': 'IBM Db2', 'odbc': 'ODBC'
        }
        if provider in source_names:
            drivers = {
                'mysql': 'mysql+pymysql', 'postgresql': 'postgresql+psycopg',
                'sqlserver': 'mssql+pyodbc', 'azuresql': 'mssql+pyodbc',
                'oracle': 'oracle+oracledb', 'ibmdb2': 'db2+ibm_db'
            }
            if provider == 'odbc':
                url = URL.create('mssql+pyodbc', query={'odbc_connect': body.get('host', '')})
            else:
                query_options = {}
                if provider in ('sqlserver', 'azuresql'):
                    query_options = {'driver': body.get('driver') or 'ODBC Driver 18 for SQL Server', 'TrustServerCertificate': 'yes'}
                url = URL.create(drivers[provider], username=body.get('username') or None,
                    password=body.get('password') or None, host=body.get('host'),
                    port=int(body.get('port')) if body.get('port') else None,
                    database=body.get('database'), query=query_options or None)
            source_name = f"{source_names[provider]}: {body.get('database') or body.get('host')}"
        elif provider == 'databricks':
            hostname = (body.get('host') or '').replace('https://', '').strip('/')
            url = URL.create('databricks', username='token', password=body.get('token'),
                host=hostname, query={
                    'http_path': body.get('http_path', ''),
                    'catalog': body.get('catalog', ''),
                    'schema': body.get('schema', '')
                })
            source_name = f"Databricks: {body.get('catalog') or hostname}"
        else:
            return jsonify({'error': 'Choose a supported database connector.'}), 400
        info = SQLITE_ENGINE.ingest_sqlalchemy(str(url), query, filename=source_name)
        ACTIVE_DATA.update({'df_raw': None, 'df_cleaned': None, 'filename': source_name})
        return jsonify(info)
    except ModuleNotFoundError:
        return jsonify({'error': connector_install_hint(provider)}), 400
    except ImportError:
        return jsonify({'error': connector_install_hint(provider)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/connect/odata', methods=['POST'])
def api_connect_odata():
    body = request.json or {}
    try:
        headers = {'Authorization': f"Bearer {body['token']}"} if body.get('token') else {}
        response = get_public_response(body.get('url', ''), headers=headers)
        payload = response.json()
        rows = payload.get('value', payload.get('results', payload)) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            return jsonify({'error': 'The OData endpoint returned no tabular records.'}), 400
        info = SQLITE_ENGINE.ingest_dataframe(pd.json_normalize(rows), filename='OData import')
        ACTIVE_DATA.update({'df_raw': None, 'df_cleaned': None, 'filename': 'OData import'})
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f"OData import failed: {e}"}), 400

@app.route('/api/connect/github', methods=['POST'])
def api_connect_github():
    body = request.json or {}
    repo = (body.get('repository') or '').strip()
    resource = body.get('resource', 'issues')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) or resource not in {'issues', 'pulls', 'commits', 'releases'}:
        return jsonify({'error': 'Enter a repository as owner/name and choose a supported GitHub resource.'}), 400
    try:
        headers = {'Accept': 'application/vnd.github+json'}
        if body.get('token'):
            headers['Authorization'] = f"Bearer {body['token']}"
        response = get_public_response(f'https://api.github.com/repos/{repo}/{resource}?per_page=100', headers=headers)
        rows = response.json()
        if not isinstance(rows, list) or not rows:
            return jsonify({'error': 'GitHub returned no records for that resource.'}), 400
        info = SQLITE_ENGINE.ingest_dataframe(pd.json_normalize(rows, sep='_'), filename=f"GitHub: {repo} / {resource}")
        ACTIVE_DATA.update({'df_raw': None, 'df_cleaned': None, 'filename': f"GitHub: {repo} / {resource}"})
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f"GitHub import failed: {e}"}), 400

@app.route('/api/connect/web', methods=['POST'])
def api_connect_web():
    body = request.json or {}
    try:
        response = get_public_response(body.get('url', ''), headers={'User-Agent': 'Vizify/0.7 data-import'})
        from io import StringIO
        tables = pd.read_html(StringIO(response.text))
        if not tables:
            return jsonify({'error': 'No HTML tables were found on this page.'}), 400
        info = SQLITE_ENGINE.ingest_dataframe(tables[0], filename=f"Web table: {urlparse(body['url']).hostname}")
        ACTIVE_DATA.update({'df_raw': None, 'df_cleaned': None, 'filename': info.get('filename')})
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f"Web table import failed: {e}"}), 400

# --- SQLite Paginated Data Route ---
@app.route('/api/data/page', methods=['POST'])
def api_data_page():
    body = request.json or {}
    page = body.get('page', 1)
    limit = body.get('limit', 50)
    filters = body.get('filters', {})
    result = SQLITE_ENGINE.get_paginated_data(page=page, limit=limit, filters=filters)
    return jsonify(result)

# --- NLP & Text Analysis Routes ---
@app.route('/api/nlp/text-columns', methods=['GET'])
def api_nlp_columns():
    df = get_active_dataframe()
    if df is None:
        return jsonify({'text_columns': []})
    text_cols = NLPProcessor.detect_text_columns(df)
    return jsonify({'text_columns': text_cols})

@app.route('/api/nlp/analyze', methods=['POST'])
def api_nlp_analyze():
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    body = request.json or {}
    col = body.get('column')
    if not col:
        text_cols = NLPProcessor.detect_text_columns(df)
        if not text_cols:
            return jsonify({'error': 'No text columns found in dataset'}), 400
        col = text_cols[0]
        
    result = NLPProcessor.analyze_text(df, col)
    return jsonify(result)

@app.route('/api/nlp/clean', methods=['POST'])
def api_nlp_clean():
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    body = request.json or {}
    col = body.get('column')
    options = body.get('options', {})
    
    if not col or col not in df.columns:
        return jsonify({'error': f"Invalid column '{col}'"}), 400
        
    df_temp = df.copy()
    cleaned_series = NLPProcessor.clean_text_series(df_temp[col], options)
    df_temp[col] = cleaned_series
    
    ACTIVE_DATA['df_cleaned'] = df_temp
    SQLITE_ENGINE.update_table_data(df_temp)
    info = SQLITE_ENGINE.get_metadata()
    
    return jsonify({
        'message': f"Text cleaning applied to '{col}' successfully",
        'info': info,
        'sample_cleaned': cleaned_series.head(10).tolist()
    })


@app.route('/api/magic-clean', methods=['POST'])
def api_magic_clean():
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    try:
        df_temp = df.copy()
        cleaned_actions = []
        
        for col in df_temp.columns:
            # 1. Drop columns with >90% nulls
            if df_temp[col].isnull().sum() / len(df_temp) > 0.9:
                df_temp = df_temp.drop(col, axis=1)
                cleaned_actions.append(f"Dropped column '{col}' (>90% missing)")
            # 2. Convert text containing currency or formatting to numeric
            elif df_temp[col].dtype == 'object':
                try:
                    cleaned_col = df_temp[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                    converted = pd.to_numeric(cleaned_col)
                    df_temp[col] = converted
                    cleaned_actions.append(f"Converted '{col}' to numeric format")
                except Exception:
                    pass
                    
        ACTIVE_DATA['df_cleaned'] = df_temp
        SQLITE_ENGINE.update_table_data(df_temp)
        info = get_dataset_metadata(df_temp, ACTIVE_DATA['filename'])
        
        return jsonify({
            'info': info,
            'actions': cleaned_actions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-chart', methods=['POST'])
def api_get_chart():
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    body = request.json or {}
    chart_type = (body.get('type') or '').strip()
    settings = body.get('settings', {})
    filters = body.get('filters', {})
    
    df_filtered = apply_filters(df, filters)
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df_filtered.columns.tolist()
    
    c_lower = chart_type.lower()

    # NLP visualizations use frequencies and co-occurrences from the selected text,
    # rather than placeholder values derived from column names.
    if 'word cloud' in c_lower or 'force' in c_lower:
        text_columns = NLPProcessor.detect_text_columns(df_filtered)
        col = settings.get('column')
        if col not in text_columns:
            col = text_columns[0] if text_columns else None
        if not col:
            return jsonify({'error': 'This visualization needs a text column with usable values'}), 400
        analysis = NLPProcessor.analyze_text(df_filtered, col)
        return jsonify({'d3': True, 'type': chart_type, 'column': col,
                        'wordcloud': analysis.get('wordcloud', []),
                        'network': analysis.get('network', {'nodes': [], 'links': []})})

    try:
        fig = None
        if 'distribution' in c_lower or 'histo' in c_lower:
            col = settings.get('column')
            if not col or col not in df_filtered.columns:
                col = num_cols[0] if num_cols else all_cols[0]
            fig = px.histogram(df_filtered, x=col, title=f"Histogram / Distribution of {col}", marginal="box", template="plotly_dark")
            
        elif 'count' in c_lower or 'categorical' in c_lower:
            col = settings.get('column')
            if not col or col not in df_filtered.columns:
                col = cat_cols[0] if cat_cols else all_cols[0]
            counts = df_filtered[col].value_counts(dropna=False).reset_index().head(25)
            counts.columns = [col, 'count']
            fig = px.bar(counts, x=col, y='count', title=f"Frequency Count of {col}", template="plotly_dark", color='count', color_continuous_scale='Viridis')
            
        elif 'pie' in c_lower or 'donut' in c_lower:
            col = settings.get('column')
            if not col or col not in df_filtered.columns:
                col = cat_cols[0] if cat_cols else all_cols[0]
            counts = df_filtered[col].value_counts(dropna=False).reset_index().head(15)
            counts.columns = [col, 'count']
            fig = px.pie(counts, names=col, values='count', hole=0.4, title=f"Excel Donut Breakdown of {col}", template="plotly_dark")
            
        elif 'line' in c_lower:
            x_col = settings.get('xAxis') or (all_cols[0] if all_cols else '')
            y_col = settings.get('yAxis') or (num_cols[0] if num_cols else all_cols[0])
            color_col = settings.get('color', '(none)')
            kw = {'color': color_col} if color_col != '(none)' and color_col in df_filtered.columns else {}
            fig = px.line(df_filtered.head(500), x=x_col, y=y_col, title=f"Excel Line Trend: {y_col} vs {x_col}", template="plotly_dark", **kw)

        elif 'area' in c_lower:
            x_col = settings.get('xAxis') or (all_cols[0] if all_cols else '')
            y_col = settings.get('yAxis') or (num_cols[0] if num_cols else all_cols[0])
            color_col = settings.get('color', '(none)')
            kw = {'color': color_col} if color_col != '(none)' and color_col in df_filtered.columns else {}
            fig = px.area(df_filtered.head(500), x=x_col, y=y_col, title=f"Excel Area Chart: {y_col} vs {x_col}", template="plotly_dark", **kw)

        elif 'stacked' in c_lower:
            x_col = settings.get('xAxis') or (cat_cols[0] if cat_cols else all_cols[0])
            y_col = settings.get('yAxis') or (num_cols[0] if num_cols else all_cols[0])
            color_col = settings.get('color') or (cat_cols[1] if len(cat_cols) > 1 else '(none)')
            kw = {'color': color_col} if color_col != '(none)' and color_col in df_filtered.columns else {}
            fig = px.bar(df_filtered.head(300), x=x_col, y=y_col, title=f"Excel Stacked Bar: {y_col} by {x_col}", barmode="stack", template="plotly_dark", **kw)

        elif 'box' in c_lower:
            y_col = settings.get('yAxis') or (num_cols[0] if num_cols else all_cols[0])
            x_col = settings.get('xAxis', '(none)')
            kw = {'x': x_col} if x_col != '(none)' and x_col in df_filtered.columns else {}
            fig = px.box(df_filtered, y=y_col, points="outliers", title=f"Excel Box Plot / Outlier Diagnostics ({y_col})", template="plotly_dark", **kw)
            
        elif 'scatter' in c_lower:
            x_col = settings.get('xAxis')
            y_col = settings.get('yAxis')
            color_col = settings.get('color', '(none)')
            if not x_col or x_col not in df_filtered.columns:
                x_col = num_cols[0] if num_cols else all_cols[0]
            if not y_col or y_col not in df_filtered.columns:
                y_col = num_cols[1] if len(num_cols) > 1 else x_col
                
            color_kw = {}
            if color_col != '(none)' and color_col in df_filtered.columns:
                color_kw['color'] = color_col
                
            fig = px.scatter(df_filtered, x=x_col, y=y_col, title=f"{y_col} vs. {x_col}", template="plotly_dark", **color_kw)
            
        elif 'correlation' in c_lower or 'heatmap' in c_lower:
            if len(num_cols) < 2:
                return jsonify({'error': 'Correlation heatmap requires at least 2 numerical columns'}), 400
            corr = df_filtered[num_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix", template="plotly_dark")
            
        else:
            col = all_cols[0] if all_cols else ''
            fig = px.bar(df_filtered.head(100), x=col, title=f"Chart for {col}", template="plotly_dark")
            
        fig_dict = json.loads(pio.to_json(fig))

        # Format d3Data payload for frontend D3 chart suite
        d3_data = []
        try:
            if any(k in c_lower for k in ['pie', 'donut', 'count', 'categorical', 'histo', 'distribution', 'bar', 'column']):
                col = settings.get('column') or (all_cols[0] if all_cols else '')
                counts = df_filtered[col].value_counts(dropna=False).head(15)
                d3_data = [{'label': str(k), 'value': int(v)} for k, v in counts.items()]
            elif 'funnel' in c_lower:
                col = settings.get('column') or (cat_cols[0] if cat_cols else all_cols[0])
                counts = df_filtered[col].value_counts().head(10)
                d3_data = [{'stage': str(k), 'value': int(v)} for k, v in counts.items()]
            elif 'radar' in c_lower or 'spider' in c_lower:
                cols_to_use = num_cols[:6] if len(num_cols) >= 3 else all_cols[:6]
                means = df_filtered[cols_to_use].mean(numeric_only=True)
                d3_data = [{'label': str(k), 'value': float(v)} for k, v in means.items()]
            else:
                x_col = settings.get('xAxis') or (all_cols[0] if all_cols else '')
                y_col = settings.get('yAxis') or (num_cols[0] if num_cols else all_cols[0])
                sample = df_filtered[[x_col, y_col]].dropna().head(20)
                d3_data = [{'label': str(r[x_col]), 'value': float(r[y_col]) if str(r[y_col]).replace('.','',1).replace('-','',1).isdigit() else 10} for _, r in sample.iterrows()]
        except Exception:
            d3_data = []

        fig_dict['d3Data'] = d3_data
        return jsonify(fig_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/chat-chart', methods=['POST'])
def api_chat_chart():
    if not GEMINI_AVAILABLE:
        return jsonify({'error': 'google-genai is not installed'}), 500
        
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    body = request.json or {}
    api_key = body.get('apiKey')
    model_name = body.get('model', 'gemini-2.0-flash')
    question = body.get('question')
    filters = body.get('filters', {})
    chart_type = body.get('type')
    settings = body.get('settings', {})
    history = body.get('history', [])
    
    if not api_key:
        return jsonify({'error': 'Gemini API Key is required'}), 400
        
    try:
        df_filtered = apply_filters(df, filters)
        df_sample = df_filtered.head(100) # Give sample to fit context limits
        
        # Build prompt
        chat_context = f"""
        You are a data analyst. The user has generated a chart of type '{chart_type}' with settings: {settings}.
        The filtered dataset has {len(df_filtered)} rows in total.
        
        Data Sample (Top 100 rows):
        {df_sample.to_string()}
        
        User Question: {question}
        
        Goal: Provide a deep, insightful answer.
        1. Do NOT just read the numbers back.
        2. Offer PLAUSIBLE HYPOTHESES or business explanations for the trends.
        3. Keep response concise (under 4 sentences).
        """
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=chat_context
        )
        
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent-chat', methods=['POST'])
def api_agent_chat():
    if not GEMINI_AVAILABLE:
        return jsonify({'error': 'google-genai is not installed'}), 500
        
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    body = request.json or {}
    api_key = body.get('apiKey')
    model_name = body.get('model', 'gemini-2.0-flash')
    question = body.get('question')
    
    if not api_key:
        return jsonify({'error': 'Gemini API Key is required'}), 400
        
    try:
        schema = df.dtypes.to_string()
        sample_data = df.head(3).to_string()
        
        sys_prompt = f"""You are a senior data analyst and expert Python programmer.
You have access to a pandas DataFrame named 'df' loaded in memory.
DataFrame columns and datatypes:
{schema}

DataFrame sample (first 3 rows):
{sample_data}

User Request: {question}

Your Task:
Generate valid Python code that will perform the requested analysis or visualization.
Rules:
1. The DataFrame 'df' is already defined. Do NOT re-read or create a mock DataFrame.
2. If the user wants to calculate something (e.g., mean, groups, correlation), calculate it and print() the results, OR assign the result to the variable 'result'.
3. If the user wants a chart, generate a Plotly Express figure and assign it to the variable 'fig' (e.g. `fig = px.bar(...)`). Choose a professional dark layout.
4. Output ONLY valid, executable Python code. Do NOT wrap it in markdown code blocks like ```python. Do NOT include comments or explanation text in your code.
5. If the request cannot be answered with code, print an appropriate message.
"""
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=sys_prompt
        )
        
        code_to_run = response.text.replace("```python", "").replace("```", "").strip()
        
        # Execute code in captured env
        captured_stdout = ""
        error_msg = None
        fig = None
        result = None
        
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        local_vars = {
            'df': df,
            'pd': pd,
            'np': np,
            'px': px,
            'result': None,
            'fig': None
        }
        
        try:
            exec(code_to_run, globals(), local_vars)
            result = local_vars.get('result')
            fig = local_vars.get('fig')
        except Exception as e:
            error_msg = str(e)
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            captured_stdout = redirected_output.getvalue()
            
        fig_json = None
        if fig is not None:
            fig_json = json.loads(pio.to_json(fig))
            
        # Summarize results
        summary_text = ""
        if error_msg:
            summary_text = f"An error occurred while executing the code: `{error_msg}`"
        else:
            summary_prompt = f"""You are a data analyst summarizing execution results.
User question: "{question}"
Generated Python Code:
{code_to_run}
Execution prints/stdout:
{captured_stdout}
Returned variable result:
{result}
Chart generated: {'Yes' if fig is not None else 'No'}

Write a concise, professional, data-analyst explanation of these results to answer the user's question. Focus on key insights. Do not describe the code itself, explain the findings.
"""
            summary_response = client.models.generate_content(
                model=model_name,
                contents=summary_prompt
            )
            summary_text = summary_response.text
            
        return jsonify({
            'content': summary_text,
            'code': code_to_run,
            'stdout': captured_stdout,
            'error': error_msg,
            'fig': fig_json
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-ml', methods=['POST'])
def api_train_ml():
    if not SKLEARN_AVAILABLE:
        return jsonify({'error': 'scikit-learn is not installed'}), 500
        
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    body = request.json or {}
    key_prefix = body.get('keyPrefix', 'ml_studio_page')
    problem_type = body.get('problemType')
    selected_features = body.get('features')
    target_column = body.get('target')
    handle_missing = body.get('missingStrategy')
    selected_algorithms = body.get('algorithms')
    test_size = body.get('testSize', 0.2)
    random_state = int(body.get('seed', 42))
    scale_features = body.get('scale', True)
    tune_hyperparameters = body.get('tune', False)
    
    is_regression = problem_type == 'Regression'
    
    try:
        X = df[selected_features].copy()
        y = df[target_column].copy()
        
        # Missing values preprocessing
        if handle_missing == "Drop rows with missing values":
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
        elif "mean" in handle_missing:
            for col in X.select_dtypes(include=[np.number]).columns:
                X[col] = X[col].fillna(X[col].mean())
            for col in X.select_dtypes(include=['object', 'category']).columns:
                if not X[col].empty:
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown')
            if y.dtype in ['int64', 'float64']:
                y = y.fillna(y.mean())
            else:
                y = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else 'Unknown')
        elif "median" in handle_missing:
            for col in X.select_dtypes(include=[np.number]).columns:
                X[col] = X[col].fillna(X[col].median())
            for col in X.select_dtypes(include=['object', 'category']).columns:
                if not X[col].empty:
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown')
            if y.dtype in ['int64', 'float64']:
                y = y.fillna(y.median())
            else:
                y = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else 'Unknown')
        elif "KNN" in handle_missing:
            num_cols_X = X.select_dtypes(include=[np.number]).columns
            if len(num_cols_X) > 0:
                imputer = KNNImputer(n_neighbors=5)
                X[num_cols_X] = imputer.fit_transform(X[num_cols_X])
            for col in X.select_dtypes(include=['object', 'category']).columns:
                if not X[col].empty:
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown')
            if y.dtype in ['int64', 'float64']:
                y = y.fillna(y.median())
            else:
                y = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else 'Unknown')
                
        # Categorical encoder
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
        label_encoder = None
        if not is_regression and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            
        # Algorithms Map
        if is_regression:
            algorithms_map = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Support Vector Machine": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor()
            }
        else:
            algorithms_map = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Support Vector Machine": SVC(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
            
        results = {}
        trained_models = {}
        
        for name in selected_algorithms:
            model = algorithms_map[name]
            start_time = time.time()
            
            if tune_hyperparameters:
                param_grid = {}
                if "Random Forest" in name:
                    param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
                elif "Decision Tree" in name:
                    param_grid = {'max_depth': [None, 10, 20]}
                elif "Support Vector Machine" in name:
                    param_grid = {'C': [0.1, 1, 10]}
                elif "K-Nearest Neighbors" in name:
                    param_grid = {'n_neighbors': [3, 5, 7]}
                
                if param_grid:
                    search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                    search.fit(X_train_scaled, y_train)
                    model = search.best_estimator_
                else:
                    model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled, y_train)
                
            y_pred = model.predict(X_test_scaled)
            training_time = time.time() - start_time
            
            # Record results
            if is_regression:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                results[name] = {
                    'RMSE': float(rmse),
                    'R2': float(r2),
                    'Training Time (s)': float(training_time),
                    'actual': y_test.tolist(),
                    'predictions': y_pred.tolist()
                }
            else:
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                results[name] = {
                    'Accuracy': float(accuracy),
                    'Training Time (s)': float(training_time),
                    'confusion_matrix': cm.tolist(),
                    'actual': y_test.tolist(),
                    'predictions': y_pred.tolist()
                }
                
            # Feature importance
            feat_imp = []
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_df = pd.DataFrame({'feature': list(X.columns), 'importance': importances})
                feat_df = feat_df.sort_values('importance', ascending=True).tail(10)
                feat_imp = feat_df.to_dict(orient='records')
            elif hasattr(model, 'coef_'):
                coefs = model.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
                feat_df = pd.DataFrame({'feature': list(X.columns), 'importance': np.abs(coefs)})
                feat_df = feat_df.sort_values('importance', ascending=True).tail(10)
                feat_imp = feat_df.to_dict(orient='records')
                
            if feat_imp:
                results[name]['feature_importances'] = feat_imp
                
            trained_models[name] = {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'features': list(X.columns),
                'target': target_column,
                'problem_type': problem_type
            }
            
        # Store in state
        ACTIVE_MODELS[key_prefix] = trained_models
        
        # Calculate best model
        if is_regression:
            best_model = min(results.keys(), key=lambda k: results[k]['RMSE'])
        else:
            best_model = max(results.keys(), key=lambda k: results[k]['Accuracy'])
            
        return jsonify({
            'results': results,
            'best_model': best_model,
            'features_list': list(X.columns)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-live', methods=['POST'])
def api_predict_live():
    body = request.json or {}
    key_prefix = body.get('keyPrefix', 'ml_studio_page')
    model_name = body.get('modelName')
    inputs = body.get('inputs', {})
    
    models_dict = ACTIVE_MODELS.get(key_prefix, {})
    model_data = models_dict.get(model_name)
    
    if not model_data:
        return jsonify({'error': f"Model '{model_name}' is not trained"}), 400
        
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        le = model_data['label_encoder']
        is_regression = model_data['problem_type'] == 'Regression'
        
        # Build raw DataFrame
        input_data = {}
        for feat in features:
            val = inputs.get(feat, 0.0)
            input_data[feat] = [float(val)]
            
        input_df = pd.DataFrame(input_data)
        
        if scaler:
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
        else:
            pred = model.predict(input_df)[0]
            
        response = {}
        
        if not is_regression and le:
            # Classification with Encoder
            pred_label = le.inverse_transform([int(pred)])[0]
            response['prediction'] = int(pred)
            response['prediction_label'] = str(pred_label)
            
            # Probability if supported
            if hasattr(model, 'predict_proba'):
                if scaler:
                    probs = model.predict_proba(input_scaled)[0]
                else:
                    probs = model.predict_proba(input_df)[0]
                response['confidence'] = float(max(probs))
        else:
            response['prediction'] = float(pred)
            
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-model', methods=['GET'])
def api_export_model():
    key_prefix = request.args.get('key_prefix', 'ml_studio_page')
    model_name = request.args.get('model_name')
    
    models_dict = ACTIVE_MODELS.get(key_prefix, {})
    model_data = models_dict.get(model_name)
    
    if not model_data:
        return jsonify({'error': f"Model '{model_name}' is not found"}), 404
        
    try:
        model_package = {
            'model': model_data['model'],
            'scaler': model_data['scaler'],
            'label_encoder': model_data['label_encoder'],
            'features': model_data['features'],
            'target': model_data['target'],
            'problem_type': model_data['problem_type'],
            'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        model_buffer = io.BytesIO()
        pickle.dump(model_package, model_buffer)
        model_buffer.seek(0)
        
        filename = f"vizify_{model_name.lower().replace(' ', '_')}_model.pkl"
        return send_file(
            model_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/octet-stream"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-pdf', methods=['GET', 'POST'])
def api_export_pdf():
    if not REPORTLAB_AVAILABLE:
        return jsonify({'error': 'reportlab is not installed'}), 500
        
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    if request.method == 'POST':
        body = request.json or {}
        items = body.get('items', [])
        filters = body.get('filters', {})
    else:
        items_str = request.args.get('items', '[]')
        filters_str = request.args.get('filters', '{}')
        try:
            items = json.loads(items_str)
            filters = json.loads(filters_str)
        except Exception:
            items = []
            filters = {}
    
    try:
        df_filtered = apply_filters(df, filters)
        
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        y_pos = height - 50
        
        # Document Header
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, y_pos, "Vizify Studio PDF Report")
        y_pos -= 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y_pos, f"Exported on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        y_pos -= 15
        c.drawString(50, y_pos, f"Source File: {ACTIVE_DATA['filename']} ({len(df_filtered)} filtered rows)")
        y_pos -= 30
        
        for item in items:
            chart_type = item['type']
            settings = item.get('settings', {})
            
            # Skip ML generator logic or similar complex ones
            if chart_type == 'ML Model Training':
                continue
                
            c_lower = chart_type.lower()
            fig = None
            if 'distribution' in c_lower or 'histo' in c_lower:
                col = settings.get('column') or (df_filtered.columns[0] if len(df_filtered.columns) > 0 else None)
                if col and col in df_filtered.columns:
                    fig = px.histogram(df_filtered, x=col, title=f"Histogram of {col}", marginal="box")
            elif 'count' in c_lower or 'categorical' in c_lower:
                col = settings.get('column') or (df_filtered.columns[0] if len(df_filtered.columns) > 0 else None)
                if col and col in df_filtered.columns:
                    counts = df_filtered[col].value_counts(dropna=False).reset_index().head(25)
                    counts.columns = [col, 'count']
                    fig = px.bar(counts, x=col, y='count', title=f"Frequency Counts of {col}")
            elif 'pie' in c_lower or 'donut' in c_lower:
                col = settings.get('column') or (df_filtered.columns[0] if len(df_filtered.columns) > 0 else None)
                if col and col in df_filtered.columns:
                    counts = df_filtered[col].value_counts(dropna=False).reset_index().head(15)
                    counts.columns = [col, 'count']
                    fig = px.pie(counts, names=col, values='count', hole=0.4, title=f"Donut Breakdown of {col}")
            elif 'line' in c_lower:
                x_col = settings.get('xAxis') or df_filtered.columns[0]
                y_col = settings.get('yAxis') or df_filtered.columns[0]
                fig = px.line(df_filtered.head(500), x=x_col, y=y_col, title=f"Line Trend: {y_col} vs {x_col}")
            elif 'area' in c_lower:
                x_col = settings.get('xAxis') or df_filtered.columns[0]
                y_col = settings.get('yAxis') or df_filtered.columns[0]
                fig = px.area(df_filtered.head(500), x=x_col, y=y_col, title=f"Area Trend: {y_col} vs {x_col}")
            elif 'stacked' in c_lower:
                x_col = settings.get('xAxis') or df_filtered.columns[0]
                y_col = settings.get('yAxis') or df_filtered.columns[0]
                fig = px.bar(df_filtered.head(300), x=x_col, y=y_col, barmode="stack", title=f"Stacked Bar: {y_col} by {x_col}")
            elif 'box' in c_lower:
                y_col = settings.get('yAxis') or df_filtered.columns[0]
                fig = px.box(df_filtered, y=y_col, points="outliers", title=f"Box Plot of {y_col}")
            elif 'scatter' in c_lower:
                x_col = settings.get('xAxis') or df_filtered.columns[0]
                y_col = settings.get('yAxis') or df_filtered.columns[0]
                color_col = settings.get('color', '(none)')
                color_kw = {'color': color_col} if color_col != '(none)' and color_col in df_filtered.columns else {}
                fig = px.scatter(df_filtered, x=x_col, y=y_col, title=f"{y_col} vs. {x_col}", **color_kw)
            elif 'correlation' in c_lower or 'heatmap' in c_lower:
                num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) >= 2:
                    corr = df_filtered[num_cols].corr(numeric_only=True)
                    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
                    
            if fig is None:
                continue
                
            if y_pos < 320:
                c.showPage()
                y_pos = height - 50
                
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, chart_type)
            y_pos -= 280
            
            # Render chart to static png and import to canvas
            try:
                img_data = fig.to_image(format="png", width=600, height=250, scale=2)
                img_reader = ImageReader(io.BytesIO(img_data))
                c.drawImage(img_reader, 50, y_pos, width=500, height=250, preserveAspectRatio=True, anchor='n')
                y_pos -= 20
            except Exception as chart_err:
                c.setFont("Helvetica", 10)
                c.drawString(50, y_pos + 10, f"Error rendering plot: {chart_err}")
                
        c.save()
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"vizify_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-csv', methods=['GET', 'POST'])
def api_export_csv():
    df = get_active_dataframe()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    if request.method == 'POST':
        body = request.json or {}
        filters = body.get('filters', {})
    else:
        filters_str = request.args.get('filters', '{}')
        try:
            filters = json.loads(filters_str)
        except Exception:
            filters = {}
    
    try:
        df_filtered = apply_filters(df, filters)
        csv_buffer = io.StringIO()
        df_filtered.to_csv(csv_buffer, index=False)
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            as_attachment=True,
            download_name="filtered_data.csv",
            mimetype="text/csv"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# ENTRYPOINT RUNNER
# ==============================================================================

def start_server(file_path=None, df=None, port=None):
    # Preload data if specified
    if df is not None:
        SQLITE_ENGINE.ingest_dataframe(df, filename="active_dataframe")
        ACTIVE_DATA['df_raw'] = None
        ACTIVE_DATA['filename'] = "active_dataframe"
        print("[Vizify] Preloaded DataFrame into SQLite workspace.")
    elif file_path is not None:
        try:
            filename = os.path.basename(file_path)
            if os.path.splitext(filename)[1].lower() == '.csv':
                SQLITE_ENGINE.ingest_csv(file_path, filename=filename)
            else:
                SQLITE_ENGINE.ingest_excel(file_path, filename=filename)
            ACTIVE_DATA['df_raw'] = None
            ACTIVE_DATA['filename'] = filename
            print(f"[Vizify] Preloaded {file_path} into server.")
        except Exception as e:
            print(f"[Vizify] Failed to preload dataset: {e}")
            
    bind_port = port if port is not None else find_free_port(5000)
    
    # Start web browser in a slight delay
    url = f"http://127.0.0.1:{bind_port}/"
    print(f"[Vizify] Launching Vizify Dashboard on {url}")
    
    # Simple browser trigger
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[Vizify] Failed to open web browser: {e}")
        
    # Disable console logging for Flask in production mode to keep CLI clean
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host='127.0.0.1', port=bind_port, debug=False)

if __name__ == '__main__':
    start_server()
