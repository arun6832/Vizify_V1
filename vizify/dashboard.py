import os
import sys
import io
import time
import socket
import webbrowser
import pickle
import json
import traceback
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


app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for development cross-origin requests

# Global state
ACTIVE_DATA = {
    'df_raw': None,
    'df_cleaned': None,
    'filename': None,
}

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
    file_loaded = ACTIVE_DATA['df_raw'] is not None
    info = {}
    if file_loaded:
        df = ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']
        info = get_dataset_metadata(df, ACTIVE_DATA['filename'])
        
    return jsonify({
        'api_key': os.getenv('GEMINI_API_KEY', ''),
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
        df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        ACTIVE_DATA['df_raw'] = df
        ACTIVE_DATA['df_cleaned'] = None
        
        # Sanitize filename to only show base name
        sanitized_filename = os.path.basename(file.filename)
        ACTIVE_DATA['filename'] = sanitized_filename
        
        info = get_dataset_metadata(df, sanitized_filename)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/magic-clean', methods=['POST'])
def api_magic_clean():
    df = ACTIVE_DATA['df_raw']
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
        info = get_dataset_metadata(df_temp, ACTIVE_DATA['filename'])
        
        return jsonify({
            'info': info,
            'actions': cleaned_actions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-chart', methods=['POST'])
def api_get_chart():
    df = ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
        
    body = request.json or {}
    chart_type = body.get('type')
    settings = body.get('settings', {})
    filters = body.get('filters', {})
    
    df_filtered = apply_filters(df, filters)
    
    try:
        fig = None
        if chart_type == 'Distribution Plot':
            col = settings.get('column')
            if not col or col not in df_filtered.columns:
                col = df_filtered.select_dtypes(include=[np.number]).columns.tolist()[0]
            fig = px.histogram(df_filtered, x=col, title=f"Distribution of {col}", marginal="box")
            
        elif chart_type == 'Categorical Plot':
            col = settings.get('column')
            if not col or col not in df_filtered.columns:
                col = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()[0]
            counts = df_filtered[col].value_counts(dropna=False).reset_index().head(25)
            counts.columns = [col, 'count']
            fig = px.bar(counts, x=col, y='count', title=f"Counts of {col}")
            
        elif chart_type == 'Scatter Plot':
            x_col = settings.get('xAxis')
            y_col = settings.get('yAxis')
            color_col = settings.get('color', '(none)')
            
            num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            if not x_col or x_col not in df_filtered.columns:
                x_col = num_cols[0]
            if not y_col or y_col not in df_filtered.columns:
                y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
                
            color_kw = {}
            if color_col != '(none)' and color_col in df_filtered.columns:
                color_kw['color'] = color_col
                
            fig = px.scatter(df_filtered, x=x_col, y=y_col, title=f"{y_col} vs. {x_col}", **color_kw)
            
        elif chart_type == 'Correlation Heatmap':
            num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                return jsonify({'error': 'Correlation heatmap requires at least 2 numerical columns'}), 400
            corr = df_filtered[num_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
            
        else:
            return jsonify({'error': f"Unknown chart type: {chart_type}"}), 400
            
        fig_dict = json.loads(pio.to_json(fig))
        return jsonify(fig_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-chart', methods=['POST'])
def api_chat_chart():
    if not GEMINI_AVAILABLE:
        return jsonify({'error': 'google-genai is not installed'}), 500
        
    df = ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']
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
        
    df = ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']
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
        
    df = ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']
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
        
    df = ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']
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
                
            fig = None
            if chart_type == 'Distribution Plot':
                col = settings.get('column')
                if col in df_filtered.columns:
                    fig = px.histogram(df_filtered, x=col, title=f"Distribution of {col}", marginal="box")
            elif chart_type == 'Categorical Plot':
                col = settings.get('column')
                if col in df_filtered.columns:
                    counts = df_filtered[col].value_counts(dropna=False).reset_index().head(25)
                    counts.columns = [col, 'count']
                    fig = px.bar(counts, x=col, y='count', title=f"Counts of {col}")
            elif chart_type == 'Scatter Plot':
                x_col = settings.get('xAxis')
                y_col = settings.get('yAxis')
                color_col = settings.get('color', '(none)')
                color_kw = {'color': color_col} if color_col != '(none)' and color_col in df_filtered.columns else {}
                if x_col in df_filtered.columns and y_col in df_filtered.columns:
                    fig = px.scatter(df_filtered, x=x_col, y=y_col, title=f"{y_col} vs. {x_col}", **color_kw)
            elif chart_type == 'Correlation Heatmap':
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
    df = ACTIVE_DATA['df_cleaned'] if ACTIVE_DATA['df_cleaned'] is not None else ACTIVE_DATA['df_raw']
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
        ACTIVE_DATA['df_raw'] = df
        ACTIVE_DATA['filename'] = "active_dataframe"
        print("[Vizify] Preloaded Pandas DataFrame into server.")
    elif file_path is not None:
        try:
            ACTIVE_DATA['df_raw'] = pd.read_csv(file_path)
            ACTIVE_DATA['filename'] = os.path.basename(file_path)
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
