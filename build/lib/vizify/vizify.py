import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import tempfile
import getpass
import time
from itertools import combinations
import sys
import subprocess
import json
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import pickle

# --- AI and PDF Generation Imports ---
try:
    from google import genai
    from google.genai import types
    from google.api_core.exceptions import GoogleAPICallError
    from fpdf import FPDF
    PDF_LIBS_AVAILABLE = True
except ImportError:
    class FPDF:
        pass
    PDF_LIBS_AVAILABLE = False

# Optional dependencies
try:
    import streamlit as st
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Optional: Pivot Table
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# Optional: Plotly click events
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False
    
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- ML Imports ---
try:
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Configuration ---
warnings.filterwarnings("ignore")


# ==============================================================================
# SECTION 1: CORE AI AND PDF GENERATION LOGIC
# ==============================================================================

def interpret_graph_with_gemini(image_path: str, prompt: str = "Briefly describe this data visualization in 2-3 sentences, summarizing the key insight."):
    """Uses Gemini 1.5 Pro to generate a description for a given chart image."""
    try:
        # Use selected model from session state if available, otherwise default
        model_name = st.session_state.get('selected_model', 'gemini-2.0-flash')
        
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) 
        with open(image_path, "rb") as f:
             image_bytes = f.read()
        
        # Simple retry logic for 429
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")]
                )
                return response.text
            except Exception as e:
                # Check for 429 in error message (since we might catch a generic exception)
                if "429" in str(e) and attempt < max_retries:
                    time.sleep(10) # Wait 10s before retry
                    continue
                raise e

    except GoogleAPICallError as e:
        return f"[API Error] {e}"
    except Exception as e:
        return f"[Error] An unexpected error occurred: {e}"

class PDF(FPDF):
    """Custom PDF class to handle headers, footers, and chapter styling."""
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Vizify', 0, False, 'R')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, 'Created Using Vizify', 0, False, 'C')
        self.cell(0, 10, f'Page {self.page_no()}', 0, False, 'R')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 24)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 20, title, 0, 1, 'C', True)
        self.ln(10)

    def plot_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def interpretation_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, "Interpretation: " + text)
        self.ln(5)

class Vizify:
    """The engine for generating static PDF reports."""
    def __init__(self, file_path, output_prefix="Plots_Report", api_key=None, sample_size=None):
        try:
            self.data = pd.read_csv(file_path, encoding="utf-8", on_bad_lines='skip')
        except FileNotFoundError:
            print(f"❌ Error: The file at {file_path} was not found.")
            raise
        self.data.attrs['name'] = os.path.basename(file_path)
        
        if sample_size and len(self.data) > sample_size:
            print(f"ℹ️ Dataset is large. Using a random sample of {sample_size} rows for plotting.")
            self.data = self.data.sample(n=sample_size, random_state=42)

        self.pdf_filename = f"{output_prefix}.pdf"
        self.html_filename = f"{output_prefix}.html"
        self.num_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
        self.time_cols = self._find_time_cols()
        self.pdf = PDF('P', 'mm', 'A4')
        self.pdf.set_auto_page_break(auto=True, margin=20)

        self.use_llm = False
        if api_key:
            if not PDF_LIBS_AVAILABLE:
                 print("⚠️ Required PDF libraries not found. AI features depend on them.")
                 return
            try:
                # With new SDK, client is initialized when needed, but we can check if we can init it
                client = genai.Client(api_key=api_key)
                self.use_llm = True
                print("✅ Gemini API key configured successfully. Interpretations are ENABLED.")
            except Exception as e:
                print(f"⚠️ Failed to configure Gemini API key: {e}. Interpretations will be DISABLED.")
        else:
            print("ℹ️ No API key provided. Interpretations are DISABLED.")

    def launch_dashboard(self=None, file_path=None, df=None, port=None):
        """Launches the interactive Vizify Studio dashboard."""
        print("[Vizify] Launching Vizify Studio Dashboard...")
        try:
            from .dashboard import start_server
        except (ImportError, ValueError):
            from dashboard import start_server

        if isinstance(self, Vizify):
            start_server(df=self.data, port=port)
        else:
            actual_df = df
            actual_file = file_path
            actual_port = port
            if isinstance(self, str):
                actual_file = self
            elif isinstance(self, pd.DataFrame):
                actual_df = self
            elif isinstance(self, int):
                actual_port = self
            start_server(file_path=actual_file, df=actual_df, port=actual_port)


    def _find_time_cols(self):
        time_cols = []
        for col in self.data.select_dtypes(include=["object", "datetime64"]).columns:
            try:
                pd.to_datetime(self.data[col], errors='raise', infer_datetime_format=True)
                self.data[col] = pd.to_datetime(self.data[col])
                time_cols.append(col)
            except (ValueError, TypeError):
                continue
        return time_cols

    def _add_plot_to_pdf(self, fig, title):
        self.pdf.add_page()
        self.pdf.plot_title(title)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_path = tmpfile.name
        tmpfile.close()
        try:
            fig.savefig(temp_path, format='png', dpi=300, bbox_inches='tight')
            self.pdf.image(temp_path, x=10, w=self.pdf.w - 20)
            self.pdf.ln(5)
            if self.use_llm:
                print(f"Interpreting: {title}...")
                description = interpret_graph_with_gemini(temp_path)
                self.pdf.interpretation_text(description)
        finally:
            os.remove(temp_path)
            plt.close(fig)

    def basic_statistics(self):
        self.pdf.add_page()
        self.pdf.chapter_title("Basic Statistics")
        stats = self.data.describe(include='all').T.fillna("N/A")
        stats = stats.applymap(lambda x: (str(x)[:25] + '...') if isinstance(x, str) and len(str(x)) > 25 else x)
        fig, ax = plt.subplots(figsize=(8, len(stats) * 0.4))
        ax.axis('tight'); ax.axis('off')
        table = ax.table(cellText=stats.values, colLabels=stats.columns, rowLabels=stats.index, loc="center", cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(8)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_path = tmpfile.name
        tmpfile.close()
        try:
            fig.savefig(temp_path, format='png', dpi=300, bbox_inches='tight')
            self.pdf.image(temp_path, x=10, w=self.pdf.w - 20)
        finally:
            os.remove(temp_path)
            plt.close(fig)

    def distribution_plots(self):
        if not self.num_cols: return
        self.pdf.add_page()
        self.pdf.chapter_title("Univariate Analysis: Numerical")
        for col in self.num_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(self.data[col].dropna(), kde=True, ax=ax, color="skyblue", bins=30)
            plt.tight_layout()
            self._add_plot_to_pdf(fig, f"Distribution of {col}")

    def categorical_plots(self):
        if not self.cat_cols: return
        self.pdf.add_page()
        self.pdf.chapter_title("Univariate Analysis: Categorical")
        for col in self.cat_cols:
            if 1 < self.data[col].nunique() < 30:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=self.data[col].astype(str), ax=ax, palette="pastel", order=self.data[col].value_counts().index)
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center', fontsize=8, color='gray', xytext=(5, 0), textcoords='offset points')
                plt.tight_layout()
                self._add_plot_to_pdf(fig, f"Count of {col}")

    def pie_charts(self):
        if not self.cat_cols: return
        self.pdf.add_page()
        self.pdf.chapter_title("Proportional Analysis: Pie Charts")
        for col in self.cat_cols:
            if 1 < self.data[col].nunique() < 10:
                fig, ax = plt.subplots(figsize=(8, 8))
                counts = self.data[col].value_counts()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"), wedgeprops={"edgecolor": "white"})
                ax.axis('equal')
                self._add_plot_to_pdf(fig, f"Pie Chart for {col}")

    def correlation_heatmap(self):
        if len(self.num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = self.data[self.num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="viridis", linewidths=0.5, ax=ax, annot_kws={"size": 8})
            plt.tight_layout()
            self._add_plot_to_pdf(fig, "Correlation Heatmap")

    def scatter_plots(self):
        if len(self.num_cols) < 2: return
        print("Generating Scatter Plots...")
        self.pdf.add_page()
        self.pdf.chapter_title("Bivariate Analysis: Scatter Plots")
        for col1, col2 in combinations(self.num_cols, 2):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=self.data, x=col1, y=col2, ax=ax, alpha=0.6)
            plt.tight_layout()
            self._add_plot_to_pdf(fig, f"Scatter Plot: {col1} vs {col2}")

    def outlier_detection_plots(self):
        if not self.num_cols: return
        print("Generating Outlier Detection Plots...")
        self.pdf.add_page()
        self.pdf.chapter_title("Outlier Detection Analysis")
        for col in self.num_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=self.data.index, y=self.data[col], ax=ax, label='Normal')
            if not outliers.empty:
                sns.scatterplot(x=outliers.index, y=outliers[col], color='red', ax=ax, label='Outlier', s=100)
            plt.title(f"Outlier Detection in {col}")
            plt.legend()
            plt.tight_layout()
            self._add_plot_to_pdf(fig, f"Outliers in {col}")

    def line_charts(self):
        if not self.time_cols or not self.num_cols: return
        print("Generating Line Charts...")
        self.pdf.add_page()
        self.pdf.chapter_title("Time-Series Line Charts")
        time_col = self.time_cols[0]
        for num_col in self.num_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            time_series_data = self.data.sort_values(by=time_col)
            sns.lineplot(data=time_series_data, x=time_col, y=num_col, ax=ax)
            plt.title(f"Trend of {num_col} over Time")
            plt.xticks(rotation=45)
            plt.tight_layout()
            self._add_plot_to_pdf(fig, f"Line Chart for {num_col}")

    def generate_html_report(self):
        html_content = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Vizify Report</title>
        <style>body{{font-family:sans-serif;margin:0;background-color:#f4f4f9;text-align:center;}} .container{{padding:20px;}} .pdf-embed{{border:1px solid #ddd;width:90%;height:80vh;max-width:1000px;}}</style>
        </head><body><div class="container"><h1>Data Visualization Report</h1>
        <p>The full PDF report is embedded below. <a href="{self.pdf_filename}">Download it here</a>.</p>
        <embed src="{self.pdf_filename}" type="application/pdf" class="pdf-embed"></div></body></html>
        """
        with open(self.html_filename, "w", encoding="utf-8") as html_file:
            html_file.write(html_content)
        print(f"✅ HTML report ready: {self.html_filename}")

# ==============================================================================
# SECTION 2: INTERACTIVE STREAMLIT DASHBOARD LOGIC (WITH ML INTEGRATION)
# ==============================================================================

def run_dashboard():
    """Contains the logic for the advanced, customizable Streamlit dashboard."""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit is not installed. Please run 'pip install streamlit plotly streamlit-option-menu'.")
        return

    # --- Page Configuration ---
    st.set_page_config(
        page_title="Vizify Interactive Dashboard",
        page_icon="📊",
        layout="wide"
    )

    # --- Helpers ---
    def coerce_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == 'object':
                try:
                    parsed = pd.to_datetime(out[c], errors='raise', infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        out[c] = parsed
                except Exception:
                    pass
        return out

    # --- AI Interpretation Function ---
    def get_ai_interpretation(df_sample, plot_title, api_key):
        if not api_key or not GEMINI_AVAILABLE:
            st.warning("Enter your Gemini API key in the sidebar to enable AI insights (google-generativeai required).")
            return
        try:
            client = genai.Client(api_key=api_key)
            # Use selected model from session state
            model_name = st.session_state.get('selected_model', 'gemini-2.0-flash')
            
            prompt = f"""
            You are a data analyst. A user has generated a plot titled '{plot_title}'.
            Based on the following sample of the data, provide a brief, insightful interpretation in 2-3 sentences.

            Data Sample:
            {df_sample.to_string()}
            """
            with st.spinner(f"🧠 Analyzing with {model_name}..."):
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                st.info(response.text)
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ Rate limit exceeded. Please wait a moment or switch models in the sidebar.")
            st.error(f"Error generating interpretation: {e}")

    # --- Local Chat Helper ---
    def render_local_chat(key_prefix, df_sample, plot_title):
        """Renders a localized chat interface for a specific chart."""
        with st.expander("💬 Chat about this chart", expanded=False):
            # Unique history for this specific chart item
            history_key = f"{key_prefix}_chat_history"
            if history_key not in st.session_state:
                st.session_state[history_key] = []
            
            # Display history
            for msg in st.session_state[history_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Input Area
            # Using text_input + button because chat_input is pinned to bottom and doesn't work well in multiple instances
            col1, col2 = st.columns([0.85, 0.15])
            user_input = col1.text_input("Ask a question specific to this chart...", key=f"{key_prefix}_input")
            send_clicked = col2.button("Send", key=f"{key_prefix}_send")
            
            if send_clicked and user_input:
                if not st.session_state.api_key:
                    st.warning("Please enter API Key in sidebar first.")
                    return

                # Add user message
                st.session_state[history_key].append({"role": "user", "content": user_input})
                st.rerun() # Rerun to show the user message immediately

            # Logic to process the last user message if it hasn't been answered (simple state check)
            # This is a bit tricky with rerun. Better flow:
            # If we just reran and the last message is user, generate answer.
            if st.session_state[history_key] and st.session_state[history_key][-1]["role"] == "user":
                 last_user_msg = st.session_state[history_key][-1]["content"]
                 with st.spinner("Thinking..."):
                    try:
                        model_name = st.session_state.get('selected_model', 'gemini-2.0-flash')
                        client = genai.Client(api_key=st.session_state.api_key)
                        
                        prompt = f"""
                        You are an expert data science consultant. The user is asking about a specific chart.
                        
                        Chart Title: '{plot_title}'
                        Data Sample (subset used for this chart):
                        {df_sample.to_string()}
                        
                        User Question: {last_user_msg}
                        
                        Your Goal: Provide a deep, insightful answer.
                        1.  Do NOT just read the numbers back to the user (they can see the chart).
                        2.  Offer PLAUSIBLE HYPOTHESES or "out-of-the-box" reasons for the trends (e.g., economic factors, human behavior, business context).
                        3.  Mention what other data would help confirm these hypotheses.
                        4.  Be professional but conversational.
                        
                        Example of good reasoning: "Education is often seen as an essential investment, whereas home improvement is discretionary and often postponed during economic uncertainty."
                        """
                        
                        # Retry logic
                        try:
                            response = client.models.generate_content(model=model_name, contents=prompt)
                            reply = response.text
                        except Exception as e:
                            if "429" in str(e):
                                time.sleep(5)
                                response = client.models.generate_content(model=model_name, contents=prompt)
                                reply = response.text
                            else:
                                raise e # Re-raise other errors

                        st.session_state[history_key].append({"role": "assistant", "content": reply})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ======================================================================
    # GLOBAL SLICERS (apply to entire dashboard)
    # ======================================================================
    def render_global_slicers(df):
        df = df.copy()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        dt_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()

        with st.sidebar.expander("🌐 Global Data Filters", expanded=False):
            for col in cat_cols:
                vals = df[col].dropna().unique().tolist()
                default_vals = st.multiselect(f"{col}", vals, default=vals)
                if default_vals:
                    df = df[df[col].isin(default_vals)]

            for col in num_cols[:6]:
                col_min, col_max = float(df[col].min()), float(df[col].max())
                if np.isfinite(col_min) and np.isfinite(col_max) and col_min != col_max:
                    r = st.slider(f"{col} range", col_min, col_max, (col_min, col_max))
                    df = df[(df[col] >= r[0]) & (df[col] <= r[1])]

            for col in dt_cols[:3]:
                min_d, max_d = df[col].min(), df[col].max()
                if pd.notna(min_d) and pd.notna(max_d) and min_d != max_d:
                    r = st.date_input(f"{col} range", (min_d.date(), max_d.date()))
                    if isinstance(r, (list, tuple)) and len(r) == 2:
                        start, end = pd.to_datetime(r[0]), pd.to_datetime(r[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        df = df[(df[col] >= start) & (df[col] <= end)]
        return df

    # ======================================================================
    # PLOTTING FUNCTIONS (for displaying in Streamlit)
    # ======================================================================
    def render_distribution_plot(df, key_prefix):
        st.markdown("#### Distribution Plot (Histogram)")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not num_cols:
            st.warning("No numerical columns available for this plot.")
            return

        selected_col = st.selectbox("Select a numerical column:", num_cols, key=f"{key_prefix}_num_col")
        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}", marginal="box")
        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(df[[selected_col]].head(), fig.layout.title.text, st.session_state.api_key)
        
        # Local Chat
        render_local_chat(key_prefix, df[[selected_col]].head(20), fig.layout.title.text)

    def render_categorical_plot(df, key_prefix):
        st.markdown("#### Categorical Plot (Bar Chart)")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            st.warning("No categorical columns available for this plot.")
            return

        selected_col = st.selectbox("Select a categorical column:", cat_cols, key=f"{key_prefix}_cat_col")
        counts = df[selected_col].value_counts(dropna=False).reset_index().head(25)
        counts.columns = [selected_col, 'count']
        fig = px.bar(counts, x=selected_col, y='count', title=f"Counts of {selected_col}")
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(counts, fig.layout.title.text, st.session_state.api_key)

        # Local Chat
        render_local_chat(key_prefix, counts, fig.layout.title.text)

    def render_scatter_plot(df, key_prefix):
        st.markdown("#### Scatter Plot")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) < 2:
            st.warning("At least two numerical columns are needed for a scatter plot.")
            return

        c1, c2, c3 = st.columns([1, 1, 1])
        x_axis = c1.selectbox("Select X-axis:", num_cols, key=f"{key_prefix}_x_axis")
        y_axis = c2.selectbox("Select Y-axis:", num_cols, index=min(1, len(num_cols)-1), key=f"{key_prefix}_y_axis")
        color = c3.selectbox("Color (optional):", ["(none)"] + df.columns.tolist(), key=f"{key_prefix}_color")
        color_kw = {} if color == "(none)" else {"color": color}

        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs. {x_axis}", **color_kw)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(df[[x_axis, y_axis]].head(), fig.layout.title.text, st.session_state.api_key)

        # Local Chat
        render_local_chat(key_prefix, df[[x_axis, y_axis]].head(20), fig.layout.title.text)

    def render_heatmap(df, key_prefix):
        st.markdown("#### Correlation Heatmap")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) < 2:
            st.warning("At least two numerical columns are needed for a heatmap.")
            return

        corr = df[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(corr, fig.layout.title.text, st.session_state.api_key)

        # Local Chat
        render_local_chat(key_prefix, corr, fig.layout.title.text)

    def render_text_to_chart(df, key_prefix):
        st.markdown("#### 🗣️ Text-to-Chart (Generative AI Builder)")
        user_prompt = st.text_area("Describe the chart you want to build...", key=f"{key_prefix}_prompt")
        
        if st.button("✨ Generate Chart", key=f"{key_prefix}_gen_btn", type="primary"):
            if not st.session_state.api_key:
                st.warning("Please enter your Gemini API Key in the sidebar.")
                return
            
            with st.spinner("AI is building your chart..."):
                try:
                    client = genai.Client(api_key=st.session_state.api_key)
                    model_name = st.session_state.selected_model
                    
                    schema = df.dtypes.to_string()
                    sys_prompt = f"""You are a python Plotly Express expert.
Dataframe 'df' is available with these columns and types:
{schema}

User request: {user_prompt}

Return ONLY valid python code that creates a Plotly Express figure assigned to the variable 'fig'.
Example:
fig = px.scatter(df, x='A', y='B', color='C')
Do NOT include markdown formatting, backticks, or comments. ONLY code.
"""
                    response = client.models.generate_content(model=model_name, contents=sys_prompt)
                    code_to_run = response.text.replace("```python", "").replace("```", "").strip()
                    
                    local_vars = {'df': df, 'px': px, 'pd': pd, 'np': np}
                    exec(code_to_run, globals(), local_vars)
                    fig = local_vars.get('fig')
                    if fig:
                        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("Show Generated Code"):
                            st.code(code_to_run, language='python')
                    else:
                        st.error("Failed to generate 'fig' variable.")
                except Exception as e:
                    st.error(f"Error building chart: {e}")

    # ======================================================================
    # NEW: ML TRAINING FUNCTION
    # ======================================================================
    def render_ml_training(df, key_prefix):
        """ML Model Training Interface"""
        if not SKLEARN_AVAILABLE:
            st.error("❌ Machine Learning features require scikit-learn. Please install it with: pip install scikit-learn")
            return
            
        st.markdown("#### 🤖 Train Machine Learning Models")
        
        # Initialize session state for ML
        if 'ml_models' not in st.session_state:
            st.session_state.ml_models = {}
        if 'ml_results' not in st.session_state:
            st.session_state.ml_results = {}
        
        # Step 1: Problem Type Selection
        st.subheader("Step 1: Choose Problem Type")
        problem_type = st.radio(
            "What type of ML problem do you want to solve?",
            ["Regression (Predict Numbers)", "Classification (Predict Categories)"],
            key=f"{key_prefix}_problem_type"
        )
        
        is_regression = problem_type.startswith("Regression")
        
        # Step 2: Feature and Target Selection
        st.subheader("Step 2: Select Features and Target")
        
        # Get appropriate columns based on problem type
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = numeric_cols + categorical_cols
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature selection
            st.write("**Features (Input Variables)**")
            selected_features = st.multiselect(
                "Select features to use for prediction:",
                options=all_cols,
                key=f"{key_prefix}_features"
            )
        
        with col2:
            # Target selection based on problem type
            st.write("**Target (What to Predict)**")
            if is_regression:
                target_options = numeric_cols
                helper_text = "Select a numerical column to predict"
            else:
                target_options = all_cols
                helper_text = "Select a column with categories to predict"
            
            target_column = st.selectbox(
                helper_text,
                options=target_options,
                key=f"{key_prefix}_target"
            )
        
        if not selected_features or not target_column:
            st.warning("Please select features and target variable to continue.")
            return
        
        # Data preprocessing preview
        st.subheader("Step 3: Data Preprocessing")
        
        # Handle missing values
        handle_missing = st.selectbox(
            "How to handle missing values?",
            ["Drop rows with missing values", "Fill with mean (numeric) / mode (categorical)", "Fill with median (numeric) / mode (categorical)", "KNN Imputation (numeric only)"],
            key=f"{key_prefix}_missing"
        )
        
        # Prepare data
        try:
            X = df[selected_features].copy()
            y = df[target_column].copy()
            
            # Handle missing values
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
            
            # Encode categorical variables
            categorical_features = [col for col in selected_features if col in categorical_cols]
            if categorical_features:
                X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
            
            # Encode target if classification
            label_encoder = None
            if not is_regression and y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            st.success(f"✅ Data prepared: {X.shape[0]} rows, {X.shape[1]} features")
            
            # Show data preview
            with st.expander("Preview Processed Data"):
                preview_df = pd.concat([X.head(), pd.Series(y[:5], name=target_column)], axis=1)
                st.dataframe(preview_df)
            
        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            return
        
        # Step 4: Model Selection
        st.subheader("Step 4: Choose ML Algorithm")
        
        if is_regression:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Support Vector Machine": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor()
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Support Vector Machine": SVC(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
        
        selected_models = st.multiselect(
            "Select algorithms to train and compare:",
            options=list(models.keys()),
            default=[list(models.keys())[0]],
            key=f"{key_prefix}_models"
        )
        
        # Step 5: Training Configuration
        st.subheader("Step 5: Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, key=f"{key_prefix}_test_size") / 100
        with col2:
            random_state = st.number_input("Random Seed", value=42, key=f"{key_prefix}_seed")
        with col3:
            scale_features = st.checkbox("Scale Features", value=True, key=f"{key_prefix}_scale")
            tune_hyperparameters = st.checkbox("Enable Hyperparameter Tuning", value=False, help="Uses GridSearchCV to find better parameters (Takes longer)", key=f"{key_prefix}_tune")
        
        # Step 6: Train Models
        if st.button("🚀 Train Models", key=f"{key_prefix}_train_btn", type="primary"):
            try:
                with st.spinner("Training models... This may take a moment."):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=int(random_state)
                    )
                    
                    # Scale features if requested
                    scaler = None
                    if scale_features:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test
                    
                    # Train selected models
                    results = {}
                    trained_models = {}
                    
                    for model_name in selected_models:
                        start_time = time.time()
                        
                        # Train model
                        model = models[model_name]
                        if tune_hyperparameters:
                            param_grid = {}
                            if "Random Forest" in model_name:
                                param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
                            elif "Decision Tree" in model_name:
                                param_grid = {'max_depth': [None, 10, 20, 30]}
                            elif "Support Vector Machine" in model_name:
                                param_grid = {'C': [0.1, 1, 10]}
                            elif "K-Nearest Neighbors" in model_name:
                                param_grid = {'n_neighbors': [3, 5, 7, 9]}
                            
                            if param_grid:
                                search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                                search.fit(X_train_scaled, y_train)
                                model = search.best_estimator_
                            else:
                                model.fit(X_train_scaled, y_train)
                        else:
                            model.fit(X_train_scaled, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        training_time = time.time() - start_time
                        
                        if is_regression:
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            results[model_name] = {
                                'MSE': mse,
                                'RMSE': np.sqrt(mse),
                                'R²': r2,
                                'Training Time (s)': training_time,
                                'predictions': y_pred,
                                'actual': y_test
                            }
                        else:
                            accuracy = accuracy_score(y_test, y_pred)
                            results[model_name] = {
                                'Accuracy': accuracy,
                                'Training Time (s)': training_time,
                                'predictions': y_pred,
                                'actual': y_test,
                                'classification_report': classification_report(y_test, y_pred, output_dict=True)
                            }
                        
                        trained_models[model_name] = {
                            'model': model,
                            'scaler': scaler,
                            'label_encoder': label_encoder,
                            'features': list(X.columns),
                            'target': target_column,
                            'problem_type': problem_type
                        }
                    
                    # Store results in session state
                    st.session_state.ml_results[key_prefix] = results
                    st.session_state.ml_models[key_prefix] = trained_models
                    
                    st.success("✅ Model training completed!")
            
            except Exception as e:
                st.error(f"Error during training: {e}")
                return
        
        # Step 7: Display Results
        if key_prefix in st.session_state.ml_results:
            st.subheader("Step 7: Model Performance")
            
            results = st.session_state.ml_results[key_prefix]
            
            # Create comparison table
            if is_regression:
                comparison_data = []
                for model_name, metrics in results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'RMSE': f"{metrics['RMSE']:.4f}",
                        'R²': f"{metrics['R²']:.4f}",
                        'Training Time (s)': f"{metrics['Training Time (s)']:.2f}"
                    })
            else:
                comparison_data = []
                for model_name, metrics in results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics['Accuracy']:.4f}",
                        'Training Time (s)': f"{metrics['Training Time (s)']:.2f}"
                    })
            
            # Display comparison table
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best model identification
            if is_regression:
                best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
                st.success(f"🏆 Best Model: **{best_model}** (Lowest RMSE: {results[best_model]['RMSE']:.4f})")
            else:
                best_model = max(results.keys(), key=lambda x: results[x]['Accuracy'])
                st.success(f"🏆 Best Model: **{best_model}** (Highest Accuracy: {results[best_model]['Accuracy']:.4f})")
            
            # Visualization
            st.subheader("Model Performance Visualization")
            
            if len(results) > 1:
                # Comparison chart
                if is_regression:
                    fig = px.bar(
                        comparison_df, 
                        x='Model', 
                        y='R²',
                        title="Model Comparison (Higher is Better)",
                        color='R²',
                        color_continuous_scale='viridis'
                    )
                else:
                    fig = px.bar(
                        comparison_df, 
                        x='Model', 
                        y='Accuracy',
                        title="Model Comparison (Higher is Better)",
                        color='Accuracy',
                        color_continuous_scale='viridis'
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction vs Actual for best model
            best_results = results[best_model]
            
            if is_regression:
                fig = px.scatter(
                    x=best_results['actual'],
                    y=best_results['predictions'],
                    title=f"Predictions vs Actual - {best_model}",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                # Add perfect prediction line
                min_val = min(best_results['actual'].min(), best_results['predictions'].min())
                max_val = max(best_results['actual'].max(), best_results['predictions'].max())
                fig.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", width=2, dash="dash")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Confusion matrix for classification
                cm = confusion_matrix(best_results['actual'], best_results['predictions'])
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title=f"Confusion Matrix - {best_model}",
                    labels=dict(x="Predicted", y="Actual", color="Count")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            best_model_obj = st.session_state.ml_models[key_prefix][best_model]['model']
            features_list = st.session_state.ml_models[key_prefix][best_model]['features']
            
            if hasattr(best_model_obj, 'feature_importances_'):
                importances = best_model_obj.feature_importances_
                feat_df = pd.DataFrame({'Feature': features_list, 'Importance': importances})
                feat_df = feat_df.sort_values('Importance', ascending=True).tail(10)
                
                fig = px.bar(
                    feat_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title=f"Top 10 Feature Importances - {best_model}"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(best_model_obj, 'coef_'):
                coefs = best_model_obj.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
                feat_df = pd.DataFrame({'Feature': features_list, 'Coefficient': coefs})
                feat_df['Abs_Coefficient'] = feat_df['Coefficient'].abs()
                feat_df = feat_df.sort_values('Abs_Coefficient', ascending=True).tail(10)
                
                fig = px.bar(
                    feat_df, 
                    x='Coefficient', 
                    y='Feature',
                    orientation='h',
                    title=f"Top 10 Feature Coefficients - {best_model}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Step 8: Model Export
        if key_prefix in st.session_state.ml_models:
            st.subheader("Step 8: Save Your Model")
            
            models_available = list(st.session_state.ml_models[key_prefix].keys())
            model_to_export = st.selectbox(
                "Select model to download:",
                models_available,
                key=f"{key_prefix}_export_model"
            )
            
            if st.button("📁 Download Model", key=f"{key_prefix}_download_btn"):
                try:
                    model_data = st.session_state.ml_models[key_prefix][model_to_export]
                    
                    # Create model package
                    model_package = {
                        'model': model_data['model'],
                        'scaler': model_data['scaler'],
                        'label_encoder': model_data['label_encoder'],
                        'features': model_data['features'],
                        'target': model_data['target'],
                        'problem_type': model_data['problem_type'],
                        'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Serialize model
                    model_buffer = io.BytesIO()
                    pickle.dump(model_package, model_buffer)
                    model_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Download Trained Model",
                        data=model_buffer.getvalue(),
                        file_name=f"vizify_{model_to_export.lower().replace(' ', '_')}_model.pkl",
                        mime="application/octet-stream"
                    )
                    
                    st.info("💡 Use this model file to make predictions on new data in Python!")
                    
                    # Show usage code
                    with st.expander("How to use this model in Python"):
                        st.code(f"""
import pickle
import pandas as pd
import numpy as np

# Load the model
with open('vizify_{model_to_export.lower().replace(' ', '_')}_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Extract components
model = model_package['model']
scaler = model_package['scaler']
features = model_package['features']

# Make predictions on new data
# new_data should be a DataFrame with the same features used in training
# new_data = pd.DataFrame(...) 

# Preprocess the new data (same as training)
if scaler:
    new_data_scaled = scaler.transform(new_data[features])
    predictions = model.predict(new_data_scaled)
else:
    predictions = model.predict(new_data[features])

print(predictions)
                        """, language='python')
                
                except Exception as e:
                    st.error(f"Error creating model download: {e}")
        
        # Step 9: Live Prediction Playground
        if key_prefix in st.session_state.ml_models:
            st.markdown("---")
            st.subheader("Step 9: 🎮 Live Prediction Playground")
            st.markdown("Adjust the values below to see real-time predictions from your best model!")
            
            models_available = list(st.session_state.ml_models[key_prefix].keys())
            model_to_use_name = st.selectbox("Select model for live predictions:", models_available, key=f"{key_prefix}_live_model")
            
            model_data = st.session_state.ml_models[key_prefix][model_to_use_name]
            live_model = model_data['model']
            live_scaler = model_data['scaler']
            live_features = model_data['features']
            
            col1, col2 = st.columns([0.7, 0.3])
            
            with col1:
                input_data = {}
                for feat in live_features:
                    # Retrieve min, max, mean from original processed dataset X
                    if feat in X.columns:
                        if X[feat].dtype in [np.float64, np.int64]:
                            min_val = float(X[feat].min())
                            max_val = float(X[feat].max())
                            mean_val = float(X[feat].mean())
                            if min_val == max_val:
                                input_data[feat] = st.number_input(feat, value=mean_val, key=f"{key_prefix}_live_{feat}")
                            else:
                                input_data[feat] = st.slider(feat, min_value=min_val, max_value=max_val, value=mean_val, key=f"{key_prefix}_live_{feat}")
                        else:
                            input_data[feat] = st.text_input(feat, value="0", key=f"{key_prefix}_live_{feat}")
                    else:
                        input_data[feat] = st.number_input(feat, value=0.0, key=f"{key_prefix}_live_{feat}")
                         
            with col2:
                # Predict instantly
                st.markdown("### 🎯 Live Prediction")
                try:
                    input_df = pd.DataFrame([input_data])
                    if live_scaler:
                        input_scaled = live_scaler.transform(input_df)
                        pred = live_model.predict(input_scaled)[0]
                    else:
                        pred = live_model.predict(input_df)[0]
                    
                    # If classification, show label
                    le = model_data['label_encoder']
                    if le:
                        pred_label = le.inverse_transform([int(pred)])[0]
                        st.markdown(f"<h2 style='color: #10B981;'>{pred_label}</h2>", unsafe_allow_html=True)
                        
                        # Show probabilities if supported
                        if hasattr(live_model, "predict_proba"):
                            probs = live_model.predict_proba(input_scaled if live_scaler else input_df)[0]
                            max_prob = max(probs) * 100
                            st.caption(f"Confidence: {max_prob:.1f}%")
                    else:
                        st.markdown(f"<h2 style='color: #3B82F6;'>{pred:.4f}</h2>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")


    # ======================================================================
    # CONVERSATIONAL AI CHAT INTERFACE
    # ======================================================================
    def render_chat_interface(df):
        st.sidebar.markdown("---")
        st.sidebar.subheader("💬 Chat with Data")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Model selection
        st.sidebar.subheader("🤖 AI Settings")
        st.session_state.selected_model = st.sidebar.selectbox(
            "Select AI Model:",
            ["gemini-2.0-flash", "gemini-flash-latest", "gemini-2.0-flash-lite-preview-02-05"],
            index=0,
            help="Switch models if you encounter rate limits (429 errors)."
        )

        # Chat expander in sidebar (or main area if preferred, but sidebar is less intrusive)
        with st.sidebar.expander("Open Chat", expanded=False):
            # Display chat messages from history on app rerun
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("Ask about your data..."):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                try:
                    client = genai.Client(api_key=st.session_state.api_key)
                    model_name = st.session_state.selected_model

                    # Construct Context
                    data_summary = df.describe(include='all').to_string()
                    chart_context = ""
                    if st.session_state.dashboard_items:
                        chart_context = f"User has generated {len(st.session_state.dashboard_items)} charts: " + ", ".join([item['type'] for item in st.session_state.dashboard_items])
                    
                    full_prompt = f"""
                    You are an expert data analyst assistant. User has loaded a dataset.
                    
                    Dataset Summary (Top stats):
                    {data_summary[:2000]}... (truncated)

                    Current Dashboard Context:
                    {chart_context}

                    User Question: {prompt}

                    Your Goal: Provide a deep, insightful answer.
                    1.  Do NOT just read the numbers back to the user.
                    2.  Offer PLAUSIBLE HYPOTHESES or "out-of-the-box" reasons for the trends (e.g., economic factors, human behavior, business context).
                    3.  Mention what other data would help confirm these hypotheses.
                    """
                    
                    # Retry logic for chat
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=full_prompt
                        )
                        bot_response = response.text
                    except Exception as e:
                        if "429" in str(e):
                             # One retry
                             time.sleep(5)
                             response = client.models.generate_content(
                                model=model_name,
                                contents=full_prompt
                            )
                             bot_response = response.text
                        else:
                            raise e

                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(bot_response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.chat_message("assistant").markdown(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    # Update PLOT_FUNCTIONS dictionary to include ML training

    PLOT_FUNCTIONS = {
        "Distribution Plot": render_distribution_plot,
        "Categorical Plot": render_categorical_plot,
        "Scatter Plot": render_scatter_plot,
        "Correlation Heatmap": render_heatmap,
        "Text-to-Chart (AI)": render_text_to_chart,
        "ML Model Training": render_ml_training,  # NEW ML FEATURE
    }

    # ======================================================================
    # FIGURE GENERATOR FUNCTIONS (for PDF export)
    # ======================================================================
    def generate_distribution_fig(df, item_id):
        key = f"item_{item_id}_num_col"
        if key not in st.session_state: return None
        selected_col = st.session_state[key]
        return px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}", marginal="box")

    def generate_categorical_fig(df, item_id):
        key = f"item_{item_id}_cat_col"
        if key not in st.session_state: return None
        selected_col = st.session_state[key]
        counts = df[selected_col].value_counts(dropna=False).reset_index().head(25)
        counts.columns = [selected_col, 'count']
        return px.bar(counts, x=selected_col, y='count', title=f"Counts of {selected_col}")

    def generate_scatter_fig(df, item_id):
        x_key, y_key, c_key = f"item_{item_id}_x_axis", f"item_{item_id}_y_axis", f"item_{item_id}_color"
        if x_key not in st.session_state or y_key not in st.session_state: return None
        x_axis, y_axis, color = st.session_state[x_key], st.session_state[y_key], st.session_state[c_key]
        color_kw = {} if color == "(none)" else {"color": color}
        return px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs. {x_axis}", **color_kw)
    
    def generate_heatmap_fig(df, item_id):
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) < 2: return None
        corr = df[num_cols].corr(numeric_only=True)
        return px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")

    def generate_ml_training_fig(df, item_id):
        """Generate a summary figure for ML training results"""
        key = f"item_{item_id}"
        if key not in st.session_state.ml_results:
            return None
        
        results = st.session_state.ml_results[key]
        if not results:
            return None
        
        # Create a simple bar chart of model performance
        model_names = list(results.keys())
        if len(model_names) == 0:
            return None
        
        # Determine metric based on problem type (check first model)
        first_result = results[model_names[0]]
        if 'Accuracy' in first_result:
            # Classification
            scores = [results[name]['Accuracy'] for name in model_names]
            title = "Model Accuracy Comparison"
            y_label = "Accuracy"
        else:
            # Regression
            scores = [results[name]['R²'] for name in model_names]
            title = "Model R² Comparison"
            y_label = "R² Score"
        
        fig = px.bar(
            x=model_names,
            y=scores,
            title=title,
            labels={'x': 'Model', 'y': y_label}
        )
        
        return fig

    PLOT_GENERATORS = {
        "Distribution Plot": generate_distribution_fig,
        "Categorical Plot": generate_categorical_fig,
        "Scatter Plot": generate_scatter_fig,
        "Correlation Heatmap": generate_heatmap_fig,
        "ML Model Training": generate_ml_training_fig,  # NEW ML FEATURE
    }

    # ======================================================================
    # MAIN APP LAYOUT
    # ======================================================================
    st.session_state.setdefault("dashboard_items", [])
    st.session_state.setdefault("api_key", "")

    # --- UX Improvements ---
    def setup_ux():
        st.markdown("""
        <style>
            /* Import Google Font */
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
            
            html, body, [class*="css"]  {
                font-family: 'Outfit', sans-serif;
            }

            /* HIDE DEFAULT STREAMLIT ELEMENTS */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            /* MAIN TITLE GRADIENT */
            .main-title {
                font-size: 3.5rem;
                font-weight: 700;
                background: linear-gradient(90deg, #A78BFA, #3B82F6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                font-size: 1.2rem;
                color: #94A3B8;
                margin-bottom: 2rem;
            }

            /* PRIMARY BUTTONS */
            .stButton > button {
                background: linear-gradient(45deg, #7C3AED, #3B82F6);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.6rem 1.2rem;
                font-weight: 600;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
                color: white;
            }

            /* CARDS / CONTAINERS */
            div[data-testid="stExpander"] {
                background-color: rgba(30, 41, 59, 0.5);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            /* CHAT BUBBLES */
            .stChatMessage {
                background-color: rgba(30, 41, 59, 0.5);
                border-radius: 12px;
                border: 1px solid rgba(148, 163, 184, 0.1);
            }
            div[data-testid="stChatMessageContent"] {
                color: #E2E8F0;
            }

            /* SIDEBAR */
            section[data-testid="stSidebar"] {
                background-color: #0F172A;
                border-right: 1px solid rgba(148, 163, 184, 0.1);
            }

            /* DATAFRAME */
            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 8px;
            }

            /* Style Sidebar Navigation Radio list */
            div[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
                gap: 8px;
            }
            div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
                background: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 8px;
                padding: 10px 14px;
                color: #94A3B8;
                font-weight: 500;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                width: 100%;
                display: flex;
                align-items: center;
                cursor: pointer;
            }
            div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
                background: rgba(124, 58, 237, 0.15);
                border-color: rgba(124, 58, 237, 0.4);
                color: #FFFFFF;
                transform: translateX(4px);
            }
            /* Hide Streamlit radio button standard circle widget */
            div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
                display: none !important;
            }
            div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
                background: linear-gradient(90deg, rgba(124, 58, 237, 0.25), rgba(59, 130, 246, 0.25)) !important;
                border-color: #3B82F6 !important;
                color: #FFFFFF !important;
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
            }
            div[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
                display: none;
            }
            
            /* Metric cards improvements */
            div[data-testid="metric-container"] {
                background-color: rgba(30, 41, 59, 0.4) !important;
                border: 1px solid rgba(148, 163, 184, 0.1) !important;
                border-radius: 12px !important;
                padding: 1.2rem !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
                transition: transform 0.2s ease !important;
            }
            div[data-testid="metric-container"]:hover {
                transform: translateY(-2px) !important;
                border-color: rgba(148, 163, 184, 0.2) !important;
            }

        </style>
        """, unsafe_allow_html=True)

    setup_ux()

    st.markdown('<div class="main-title">Vizify AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Next-Generation Automated Data Analysis & Insights</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_file = st.file_uploader("1) Upload CSV", type="csv")
        st.session_state.api_key = st.text_input("2) Gemini API Key (optional for AI insights)", type="password")
        
        # Display navigation only if file is uploaded
        if uploaded_file is not None:
            st.markdown("---")
            st.subheader("🗺️ Navigation")
            page = st.radio(
                "Go to Page:",
                ["📊 Data Explorer", "🧹 Data Profiler", "🤖 ML Studio", "💬 AI Data Agent"],
                key="dashboard_page_nav"
            )
        else:
            page = "📊 Data Explorer"

        if uploaded_file is not None and page == "📊 Data Explorer":
            st.markdown("---")
            st.header("Add Charts to Dashboard")
            chart_type_to_add = st.selectbox("Select a chart type:", list(PLOT_FUNCTIONS.keys()))
            if st.button(f"➕ Add {chart_type_to_add}", use_container_width=True):
                st.session_state.dashboard_items.append({"type": chart_type_to_add, "id": time.time()})
                st.rerun()

    if uploaded_file is None:
        st.markdown("""
        <div style="background-color: rgba(30, 41, 59, 0.4); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 2.5rem; text-align: center; margin-top: 2rem;">
            <h2 style="color: #A78BFA; font-weight: 600;">Welcome to Vizify Dashboard Studio</h2>
            <p style="color: #94A3B8; font-size: 1.1rem; margin-bottom: 2rem;">
                A next-generation platform for automated data analysis, visualization, and no-code machine learning.
            </p>
            <div style="background: rgba(124, 58, 237, 0.1); border: 1px dashed rgba(124, 58, 237, 0.3); border-radius: 8px; padding: 1.5rem; color: #E2E8F0; display: inline-block;">
                👈 <strong>To get started, upload a CSV dataset in the sidebar.</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if "df_cleaned" in st.session_state and st.session_state.get('uploaded_file_name') == uploaded_file.name:
        df_raw = st.session_state.df_cleaned.copy()
    else:
        df_raw = pd.read_csv(uploaded_file)
        
    df_raw = coerce_datetime_cols(df_raw)
    df = render_global_slicers(df_raw)

    # Render Chat Interface
    if page == "📊 Data Explorer":
        render_chat_interface(df)

    # ======================================================================
    # PAGES
    # ======================================================================
    if page == "📊 Data Explorer":
        st.markdown('<h3 style="color: #A78BFA; font-weight: 600;">📊 Data Explorer & Dashboard</h3>', unsafe_allow_html=True)
        
        if not st.session_state.dashboard_items:
            st.info("Your dashboard is empty. Add some charts from the sidebar to get started!")
        
        for i, item in enumerate(list(st.session_state.dashboard_items)):
            with st.container(border=True):
                header_cols = st.columns([0.85, 0.15])
                header_cols[0].subheader(item["type"])
                if header_cols[1].button("🗑️", key=f"del_{item['id']}", use_container_width=True, help="Delete"):
                    st.session_state.dashboard_items.pop(i)
                    st.rerun()
                
                render_function = PLOT_FUNCTIONS.get(item["type"])
                if render_function:
                    render_function(df, key_prefix=f"item_{item['id']}")
                else:
                    st.warning(f"Unknown chart type: {item['type']}")
        
        st.markdown("---")
        
        util_c1, util_c2, util_c3 = st.columns(3)
        with util_c1:
            if st.button("🔄 Reset Dashboard Items"):
                st.session_state.dashboard_items = []
                st.rerun()
        with util_c2:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Export Filtered CSV", data=csv_buf.getvalue(), file_name="filtered_data.csv")
        
        # --- PDF Export option ---
        with util_c3:
            if st.button("📄 Download Dashboard as PDF"):
                if not REPORTLAB_AVAILABLE:
                    st.error("PDF export requires `reportlab`. Please run `pip install reportlab`.")
                else:
                    with st.spinner("Generating PDF... This may take a moment."):
                        try:
                            pdf_buffer = io.BytesIO()
                            c = canvas.Canvas(pdf_buffer, pagesize=letter)
                            width, height = letter
                            y_pos = height - 50

                            c.setFont("Helvetica-Bold", 18)
                            c.drawString(50, y_pos, "Vizify Dashboard Report")
                            y_pos -= 20
                            c.setFont("Helvetica", 10)
                            c.drawString(50, y_pos, f"Exported on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                            y_pos -= 15
                            c.drawString(50, y_pos, f"Data Source: {uploaded_file.name} ({len(df)} filtered rows)")
                            y_pos -= 30

                            for item in st.session_state.dashboard_items:
                                generator_func = PLOT_GENERATORS.get(item['type'])
                                if not generator_func: continue

                                fig = generator_func(df, item['id'])
                                if fig is None: continue

                                # Check for page break
                                if y_pos < 350:
                                    c.showPage()
                                    y_pos = height - 50

                                c.setFont("Helvetica-Bold", 14)
                                c.drawString(50, y_pos, item['type'])
                                y_pos -= 280 # Reserve space for the image

                                # Convert plot to image and draw it
                                img_data = fig.to_image(format="png", width=600, height=250, scale=2)
                                img_reader = ImageReader(io.BytesIO(img_data))
                                c.drawImage(img_reader, 50, y_pos, width=500, height=250, preserveAspectRatio=True, anchor='n')
                                y_pos -= 20

                            c.save()
                            pdf_buffer.seek(0)

                            st.download_button(
                                label="✅ Download PDF Now",
                                data=pdf_buffer,
                                file_name=f"vizify_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                        except ImportError:
                            st.error("PDF export requires `kaleido`. Please run `pip install kaleido`.")
                        except Exception as e:
                            st.error(f"An error occurred during PDF generation: {e}")

    elif page == "🧹 Data Profiler":
        st.markdown('<h3 style="color: #A78BFA; font-weight: 600;">🧹 Data Profiler & Cleaner</h3>', unsafe_allow_html=True)
        st.markdown('Analyze data quality, view statistics, and perform cleaning operations.')
        st.markdown('---')
        
        # Row 1: Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Total Columns", f"{df.shape[1]}")
        
        missing_count = df.isnull().sum().sum()
        total_cells = df.size
        missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
        col3.metric("Missing Values", f"{missing_count:,}", f"{missing_pct:.1f}%")
        
        duplicate_count = df.duplicated().sum()
        col4.metric("Duplicate Rows", f"{duplicate_count:,}")
        
        # Row 2: Columns breakdown and Clean action
        c1, c2 = st.columns([0.6, 0.4])
        with c1:
            st.markdown("### 📊 Column Types & Health")
            # Build list of columns, their datatypes, and missing percentage
            col_health_data = []
            for col in df.columns:
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df) * 100)
                col_health_data.append({
                    "Column Name": col,
                    "Data Type": str(df[col].dtype),
                    "Non-Null Count": len(df) - null_count,
                    "Missing %": f"{null_pct:.1f}%"
                })
            st.dataframe(pd.DataFrame(col_health_data), use_container_width=True)
            
        with c2:
            st.markdown("### 🪄 Magic Data Cleaner")
            st.info("The automated data cleaner will:\n1. Drop columns with more than 90% missing values.\n2. Strip currency symbols ($) and commas from numeric fields stored as text.\n3. Attempt to coerce date/time fields.")
            
            # Action button
            if st.button("🪄 Run Magic Clean", use_container_width=True, type="primary"):
                df_temp = df_raw.copy()
                cleaned_actions = []
                for col in df_temp.columns:
                    if df_temp[col].isnull().sum() / len(df_temp) > 0.9:
                        df_temp = df_temp.drop(col, axis=1)
                        cleaned_actions.append(f"Dropped column '{col}' (>90% nulls)")
                    elif df_temp[col].dtype == 'object':
                        try:
                            # Strip $ and , and try conversion
                            cleaned_col = df_temp[col].astype(str).str.replace('$', '').str.replace(',', '')
                            converted = pd.to_numeric(cleaned_col)
                            df_temp[col] = converted
                            cleaned_actions.append(f"Converted '{col}' to numeric")
                        except:
                            pass
                st.session_state.df_cleaned = df_temp
                st.session_state.uploaded_file_name = uploaded_file.name
                
                # Show results in a toast or callout
                st.success("Magic cleaning completed successfully!")
                for action in cleaned_actions:
                    st.caption(f"- {action}")
                time.sleep(1.5)
                st.rerun()

        # Row 3: Descriptive Statistics
        st.markdown("### 📈 Summary Statistics")
        st.dataframe(df.describe(include='all').T.fillna("N/A"), use_container_width=True)

    elif page == "🤖 ML Studio":
        st.markdown('<h3 style="color: #3B82F6; font-weight: 600;">🤖 No-Code Machine Learning Studio</h3>', unsafe_allow_html=True)
        st.markdown('Train, evaluate, and compare multiple models, inspect feature importance, and test live predictions.')
        st.markdown('---')
        
        render_ml_training(df, key_prefix="ml_studio_page")

    elif page == "💬 AI Data Agent":
        st.markdown('<h3 style="color: #10B981; font-weight: 600;">💬 AI Data Agent (Python Interpreter)</h3>', unsafe_allow_html=True)
        st.markdown('Ask questions in natural language. The agent will write and run Python code to analyze your data and visualize results.')
        st.markdown('---')
        
        if "agent_chat_history" not in st.session_state:
            st.session_state.agent_chat_history = []

        api_key = st.session_state.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API Key in the sidebar to enable the AI Data Agent.")
            return

        # Render past messages
        for msg in st.session_state.agent_chat_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    # Assistant message
                    st.markdown(msg["content"])
                    
                    # Display Plotly chart if generated
                    if "fig" in msg and msg["fig"] is not None:
                        st.plotly_chart(msg["fig"], use_container_width=True)
                    
                    # Display error if execution failed
                    if "error" in msg and msg["error"]:
                        st.error(f"Execution Error: {msg['error']}")
                        
                    # Display stdout prints if any
                    if "stdout" in msg and msg["stdout"].strip():
                        with st.expander("Output Prints"):
                            st.code(msg["stdout"])
                            
                    # Display generated code
                    if "code" in msg and msg["code"]:
                        with st.expander("Show Generated Python Code"):
                            st.code(msg["code"], language="python")

        if prompt := st.chat_input("Ask the Data Agent (e.g., 'What is the correlation between sales and profit?' or 'Show me the distribution of age')"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.agent_chat_history.append({"role": "user", "content": prompt})
            
            # Now generate answer
            with st.spinner("AI is thinking and writing code..."):
                try:
                    client = genai.Client(api_key=api_key)
                    model_name = st.session_state.get('selected_model', 'gemini-2.0-flash')
                    
                    # Define schema context
                    schema = df.dtypes.to_string()
                    sample_data = df.head(3).to_string()
                    
                    sys_prompt = f"""You are a senior data analyst and expert Python programmer.
You have access to a pandas DataFrame named 'df' loaded in memory.
DataFrame columns and datatypes:
{schema}

DataFrame sample (first 3 rows):
{sample_data}

User Request: {prompt}

Your Task:
Generate valid Python code that will perform the requested analysis or visualization.
Rules:
1. The DataFrame 'df' is already defined. Do NOT re-read or create a mock DataFrame.
2. If the user wants to calculate something (e.g., mean, groups, correlation), calculate it and print() the results, OR assign the result to the variable 'result'.
3. If the user wants a chart, generate a Plotly Express figure and assign it to the variable 'fig' (e.g. `fig = px.bar(...)` or `fig = px.scatter(...)`). Choose a professional dark layout.
4. Output ONLY valid, executable Python code. Do NOT wrap it in markdown code blocks like ```python. Do NOT include comments or explanation text in your code.
5. If the request cannot be answered with code, print an appropriate message.
"""
                    
                    response = client.models.generate_content(
                        model=model_name,
                        contents=sys_prompt
                    )
                    
                    code_to_run = response.text.replace("```python", "").replace("```", "").strip()
                    
                    # Run the code in a sandboxed environment and capture stdout/results
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
                        'plt': plt,
                        'sns': sns,
                        'result': None,
                        'fig': None
                    }
                    
                    try:
                        exec(code_to_run, globals(), local_vars)
                        result = local_vars.get('result')
                        fig = local_vars.get('fig')
                    except Exception as e:
                        error_msg = str(e)
                    finally:
                        sys.stdout = old_stdout
                        captured_stdout = redirected_output.getvalue()
                    
                    # Ask Gemini to write a summary analysis of the execution results
                    summary_text = ""
                    if error_msg:
                        summary_text = f"An error occurred while running the analysis code: `{error_msg}`. Please try rephrasing your request."
                    else:
                        summary_prompt = f"""You are a data analyst summarizing execution results.
User question: "{prompt}"
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
                    
                    # Store in chat history
                    st.session_state.agent_chat_history.append({
                        "role": "assistant",
                        "content": summary_text,
                        "code": code_to_run,
                        "stdout": captured_stdout,
                        "error": error_msg,
                        "fig": fig
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing AI Agent: {e}")

def run_pdf_generator_cli():
    """Interactive Command-Line Interface to configure and run the PDF report generator."""
    print("\n--- Starting PDF Report Generator ---")

    while True:
        file_path = input("Enter the path to your CSV file: ")
        if os.path.exists(file_path) and file_path.lower().endswith('.csv'):
            break
        else:
            print("❌ File not found or is not a CSV. Please provide a valid path.")

    output_prefix = input("Enter a name for your output report (e.g., My_Data_Report): ")
    if not output_prefix:
        output_prefix = os.path.splitext(os.path.basename(file_path))[0] + "_Report"
        print(f"ℹ️ No name provided. Using default: '{output_prefix}'")

    sample_size = None
    if input("Dataset large? Use random sampling for faster plotting? (yes/no): ").lower() in ['y', 'yes']:
        try:
            sample_size = int(input("Enter sample size (e.g., 50000): "))
        except ValueError:
            print("⚠️ Invalid number. Sampling will be skipped.")
            sample_size = None
    
    plot_choices = {
        "1": ("Basic Statistics", "basic_statistics"), "2": ("Distribution Plots", "distribution_plots"),
        "3": ("Categorical Plots", "categorical_plots"), "4": ("Pie Charts", "pie_charts"),
        "5": ("Correlation Heatmap", "correlation_heatmap"), "6": ("Scatter Plots", "scatter_plots"),
        "7": ("Outlier Detection", "outlier_detection_plots"), "8": ("Line Charts", "line_charts"),
    }
    print("\nSelect the plots you want to generate:")
    for key, (desc, _) in plot_choices.items():
        print(f"  {key}: {desc}")
    selection_str = input("Enter numbers separated by commas (e.g., 1,3,5), or leave blank for all: ")
    
    selected_methods = []
    if not selection_str.strip():
        selected_methods = [name for _, name in plot_choices.values()]
        print("ℹ️ No selection made. Generating all plots.")
    else:
        for key in selection_str.split(','):
            key = key.strip()
            if key in plot_choices:
                selected_methods.append(plot_choices[key][1])
            else:
                print(f"⚠️ Invalid selection '{key}' will be ignored.")

    api_key = None
    if input("Enable AI-powered interpretations? (yes/no): ").lower() in ['y', 'yes']:
        api_key = os.getenv("GEMINI_API_KEY") or getpass.getpass("Please enter your Gemini API key: ")
        if not api_key:
            print("⚠️ No API key provided. Running in plots-only mode.")

    try:
        start_time = time.time()
        viz = Vizify(file_path, output_prefix=output_prefix, api_key=api_key, sample_size=sample_size)
        print("🚀 Starting visualization report generation...")
        viz.pdf.add_page()
        viz.pdf.chapter_title("Vizify Data Report")
        viz.pdf.set_font('Helvetica', '', 12)
        viz.pdf.multi_cell(0, 10, f"An automated analysis of the file: {viz.data.attrs.get('name', '')}\n"
                                f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for method_name in selected_methods:
            method_to_call = getattr(viz, method_name)
            method_to_call()
        
        viz.pdf.output(viz.pdf_filename)
        print(f"✅ All visualizations saved in {viz.pdf_filename}")
        viz.generate_html_report()
        
        end_time = time.time()
        duration = end_time - start_time
        print("-" * 60)
        print(f"🎉 Report generation complete!")
        print(f"Total time taken: {duration:.2f} seconds.")
        print("-" * 60)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during report generation: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'run_dashboard':
        try:
            from .dashboard import start_server
        except (ImportError, ValueError):
            from dashboard import start_server
        start_server()
    else:
        print("=" * 60)
        print("      Welcome to Vizify: The Automated Analysis Tool!      ")
        print("=" * 60)
        while True:
            print("\nChoose your output format:")
            print("  1: Static PDF Report")
            print("  2: Interactive Dashboard")
            choice = input("Enter your choice (1 or 2): ")
            if choice in ['1', '2']:
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        if choice == '1':
            if not PDF_LIBS_AVAILABLE:
                print("❌ PDF generation libraries not found.")
                print("Please run: pip install fpdf2 google-generativeai")
            else:
                run_pdf_generator_cli()
        elif choice == '2':
            print("Launching Vizify Studio Dashboard... Your browser will open shortly.")
            try:
                from .dashboard import start_server
                start_server()
            except (ImportError, ValueError):
                from dashboard import start_server
                start_server()
            except Exception as e:
                print(f"An error occurred while launching Dashboard: {e}")