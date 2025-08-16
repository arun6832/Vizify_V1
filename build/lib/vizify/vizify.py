# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import warnings
# import os
# import tempfile
# import getpass
# import time  # Import the time module
# from itertools import combinations

# # Import the Google Generative AI library
# import google.generativeai as genai
# from google.api_core.exceptions import GoogleAPICallError

# # Import the PDF library
# from fpdf import FPDF

# # --- Configuration ---
# warnings.filterwarnings("ignore")


# # --- Core Functions ---

# def interpret_graph_with_gemini(image_path: str, prompt: str = "Briefly describe this data visualization in 2-3 sentences, summarizing the key insight."):
#     """Uses Gemini 1.5 Pro to generate a description for a given chart image."""
#     try:
#         model = genai.GenerativeModel('gemini-1.5-pro-latest')
#         image_file = genai.upload_file(path=image_path)
#         response = model.generate_content([prompt, image_file])
#         return response.text
#     except GoogleAPICallError as e:
#         return f"[API Error] {e}"
#     except Exception as e:
#         return f"[Error] An unexpected error occurred: {e}"


# # --- PDF Styling Class ---

# class PDF(FPDF):
#     """Custom PDF class to handle headers, footers, and chapter styling."""
#     def header(self):
#         self.set_font('Helvetica', 'B', 12)
#         self.cell(0, 10, 'Vizify', 0, False, 'R')
#         self.ln(10)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Helvetica', 'I', 8)
#         self.cell(0, 10, 'Created Using Vizify', 0, False, 'C')
#         self.cell(0, 10, f'Page {self.page_no()}', 0, False, 'R')

#     def chapter_title(self, title):
#         self.set_font('Helvetica', 'B', 24)
#         self.set_fill_color(240, 240, 240)
#         self.cell(0, 20, title, 0, 1, 'C', True)
#         self.ln(10)

#     def plot_title(self, title):
#         self.set_font('Helvetica', 'B', 14)
#         self.set_text_color(50, 50, 50)
#         self.cell(0, 10, title, 0, 1, 'L')
#         self.ln(2)

#     def interpretation_text(self, text):
#         self.set_font('Helvetica', '', 10)
#         self.set_text_color(80, 80, 80)
#         self.multi_cell(0, 5, "Interpretation: " + text)
#         self.ln(5)


# # --- Main Vizify Class ---

# class Vizify:
#     """
#     An automated tool for generating data visualization reports from CSV files.
#     """
#     def __init__(self, file_path, output_prefix="Plots_Report", api_key=None,sample_size=None):
#         """
#         Initializes the Vizify analysis tool.

#         Args:
#             file_path (str): The path to the CSV data file.
#             output_prefix (str): The prefix for the output PDF and HTML files.
#             api_key (str, optional): A Google AI Studio API key.
#                                      If provided, enables plot interpretations.
#                                      If None, only plots are generated.
#         """
#         try:
#             self.data = pd.read_csv(file_path, encoding="utf-8", on_bad_lines='skip')
#         except FileNotFoundError:
#             print(f"❌ Error: The file at {file_path} was not found.")
#             raise
#         self.data.attrs['name'] = os.path.basename(file_path)
#         self.pdf_filename = f"{output_prefix}.pdf"
#         self.html_filename = f"{output_prefix}.html"
#         self.num_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
#         self.cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
#         self.time_cols = self._find_time_cols()
#         self.pdf = PDF('P', 'mm', 'A4')
#         self.pdf.set_auto_page_break(auto=True, margin=20)

#         # Configure API key and determine if interpretations are enabled
#         self.use_llm = False
#         if api_key:
#             try:
#                 genai.configure(api_key=api_key)
#                 self.use_llm = True
#                 print("✅ Gemini API key configured successfully. Interpretations are ENABLED.")
#             except Exception as e:
#                 print(f"⚠️ Failed to configure Gemini API key: {e}. Interpretations will be DISABLED.")
#         else:
#             print("ℹ️ No API key provided. Interpretations are DISABLED.")

#     def _find_time_cols(self):
#         """Identifies and converts date/time columns in the dataframe."""
#         time_cols = []
#         for col in self.data.select_dtypes(include=["object", "datetime64"]).columns:
#             try:
#                 pd.to_datetime(self.data[col], errors='raise', infer_datetime_format=True)
#                 self.data[col] = pd.to_datetime(self.data[col])
#                 time_cols.append(col)
#             except (ValueError, TypeError):
#                 continue
#         return time_cols

#     def _add_plot_to_pdf(self, fig, title):
#         """A helper function to save a plot to a temp file and add it to the PDF."""
#         self.pdf.add_page()
#         self.pdf.plot_title(title)
        
#         tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#         temp_path = tmpfile.name
#         tmpfile.close()

#         try:
#             fig.savefig(temp_path, format='png', dpi=300, bbox_inches='tight')
#             self.pdf.image(temp_path, x=10, w=self.pdf.w - 20)
#             self.pdf.ln(5)

#             if self.use_llm:
#                 print(f"Interpreting: {title}...")
#                 description = interpret_graph_with_gemini(temp_path)
#                 self.pdf.interpretation_text(description)
#         finally:
#             os.remove(temp_path)
#             plt.close(fig)

#     def basic_statistics(self):
#         """Adds a summary statistics table to the PDF."""
#         self.pdf.add_page()
#         self.pdf.chapter_title("Basic Statistics")
#         stats = self.data.describe(include='all').T.fillna("N/A")
#         stats = stats.applymap(lambda x: (str(x)[:25] + '...') if isinstance(x, str) and len(str(x)) > 25 else x)
        
#         fig, ax = plt.subplots(figsize=(8, len(stats) * 0.4))
#         ax.axis('tight'); ax.axis('off')
#         table = ax.table(cellText=stats.values, colLabels=stats.columns, rowLabels=stats.index, loc="center", cellLoc='center')
#         table.auto_set_font_size(False); table.set_fontsize(8)

#         tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#         temp_path = tmpfile.name
#         tmpfile.close()
#         try:
#             fig.savefig(temp_path, format='png', dpi=300, bbox_inches='tight')
#             self.pdf.image(temp_path, x=10, w=self.pdf.w - 20)
#         finally:
#             os.remove(temp_path)
#             plt.close(fig)

#     def distribution_plots(self):
#         """Generates and adds distribution plots for all numerical columns."""
#         if not self.num_cols: return
#         self.pdf.add_page()
#         self.pdf.chapter_title("Univariate Analysis: Numerical")
#         for col in self.num_cols:
#             fig, ax = plt.subplots(figsize=(8, 5))
#             sns.histplot(self.data[col].dropna(), kde=True, ax=ax, color="skyblue", bins=30)
#             plt.tight_layout()
#             self._add_plot_to_pdf(fig, f"Distribution of {col}")

#     def categorical_plots(self):
#         """Generates and adds count plots for categorical columns."""
#         if not self.cat_cols: return
#         self.pdf.add_page()
#         self.pdf.chapter_title("Univariate Analysis: Categorical")
#         for col in self.cat_cols:
#             if 1 < self.data[col].nunique() < 30:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 sns.countplot(y=self.data[col].astype(str), ax=ax, palette="pastel", order=self.data[col].value_counts().index)
#                 for p in ax.patches:
#                     ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
#                                 ha='left', va='center', fontsize=8, color='gray', xytext=(5, 0), textcoords='offset points')
#                 plt.tight_layout()
#                 self._add_plot_to_pdf(fig, f"Count of {col}")

#     def pie_charts(self):
#         """Generates and adds pie charts for categorical columns with few unique values."""
#         if not self.cat_cols: return
#         self.pdf.add_page()
#         self.pdf.chapter_title("Proportional Analysis: Pie Charts")
#         for col in self.cat_cols:
#             if 1 < self.data[col].nunique() < 10:
#                 fig, ax = plt.subplots(figsize=(8, 8))
#                 counts = self.data[col].value_counts()
#                 ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
#                        colors=sns.color_palette("pastel"), wedgeprops={"edgecolor": "white"})
#                 ax.axis('equal')
#                 self._add_plot_to_pdf(fig, f"Pie Chart for {col}")

#     def correlation_heatmap(self):
#         """Generates and adds a correlation heatmap."""
#         if len(self.num_cols) > 1:
#             fig, ax = plt.subplots(figsize=(10, 8))
#             corr = self.data[self.num_cols].corr()
#             sns.heatmap(corr, annot=True, cmap="viridis", linewidths=0.5, ax=ax, annot_kws={"size": 8})
#             plt.tight_layout()
#             self._add_plot_to_pdf(fig, "Correlation Heatmap")

#     def scatter_plots(self):
#         """Generates and adds scatter plots for pairs of numerical columns."""
#         if len(self.num_cols) < 2: return
#         self.pdf.add_page()
#         self.pdf.chapter_title("Bivariate Analysis: Scatter Plots")
#         for col1, col2 in combinations(self.num_cols, 2):
#             fig, ax = plt.subplots(figsize=(10, 6))
#             sns.scatterplot(data=self.data, x=col1, y=col2, ax=ax, alpha=0.6)
#             plt.tight_layout()
#             self._add_plot_to_pdf(fig, f"Scatter Plot: {col1} vs {col2}")

#     def outlier_detection_plots(self):
#         """Generates plots highlighting outliers in numerical columns."""
#         if not self.num_cols: return
#         self.pdf.add_page()
#         self.pdf.chapter_title("Outlier Detection Analysis")
#         for col in self.num_cols:
#             Q1 = self.data[col].quantile(0.25)
#             Q3 = self.data[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
            
#             outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            
#             fig, ax = plt.subplots(figsize=(10, 6))
#             sns.scatterplot(x=self.data.index, y=self.data[col], ax=ax, label='Normal')
#             if not outliers.empty:
#                 sns.scatterplot(x=outliers.index, y=outliers[col], color='red', ax=ax, label='Outlier', s=100)
            
#             plt.title(f"Outlier Detection in {col}")
#             plt.legend()
#             plt.tight_layout()
#             self._add_plot_to_pdf(fig, f"Outliers in {col}")

#     def line_charts(self):
#         """Generates line charts for time-series data."""
#         if not self.time_cols or not self.num_cols: return
#         self.pdf.add_page()
#         self.pdf.chapter_title("Time-Series Line Charts")
#         time_col = self.time_cols[0]
        
#         for num_col in self.num_cols:
#             fig, ax = plt.subplots(figsize=(12, 6))
#             time_series_data = self.data.sort_values(by=time_col)
#             sns.lineplot(data=time_series_data, x=time_col, y=num_col, ax=ax)
#             plt.title(f"Trend of {num_col} over Time")
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             self._add_plot_to_pdf(fig, f"Line Chart for {num_col}")

#     def generate_html_report(self):
#         """Generates a simple HTML file to embed and display the PDF report."""
#         html_content = f"""
#         <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Vizify Report</title>
#         <style>body{{font-family:sans-serif;margin:0;background-color:#f4f4f9;text-align:center;}} .container{{padding:20px;}} .pdf-embed{{border:1px solid #ddd;width:90%;height:80vh;max-width:1000px;}}</style>
#         </head><body><div class="container"><h1>Data Visualization Report</h1>
#         <p>The full PDF report is embedded below. <a href="{self.pdf_filename}">Download it here</a>.</p>
#         <embed src="{self.pdf_filename}" type="application/pdf" class="pdf-embed"></div></body></html>
#         """
#         with open(self.html_filename, "w", encoding="utf-8") as html_file:
#             html_file.write(html_content)
#         print(f"✅ HTML report ready: {self.html_filename}")

#     def show_all_visualizations(self):
#         """Runs all analysis and plotting functions to generate the final report."""
#         print("🚀 Starting visualization report generation...")
        
#         self.pdf.add_page()
#         self.pdf.chapter_title("Vizify Data Report")
#         self.pdf.set_font('Helvetica', '', 12)
#         self.pdf.multi_cell(0, 10, f"An automated analysis of the file: {self.data.attrs.get('name', '')}\n"
#                                 f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

#         self.basic_statistics()
#         self.distribution_plots()
#         self.categorical_plots()
#         self.pie_charts()
#         self.correlation_heatmap()
#         self.scatter_plots()
#         self.outlier_detection_plots()
#         self.line_charts()
        
#         self.pdf.output(self.pdf_filename)
#         print(f"✅ All visualizations saved in {self.pdf_filename}")
        
#         self.generate_html_report()


# if __name__ == "__main__":
#     print("=" * 60)
#     print("      Welcome to Vizify: The Automated Visualization Reporter!      ")
#     print("=" * 60)

#     # 1. Get user inputs for file path, report name, and sampling
#     while True:
#         file_path = input("Enter the path to your CSV file: ")
#         if os.path.exists(file_path) and file_path.lower().endswith('.csv'):
#             break
#         else:
#             print("❌ File not found or is not a CSV. Please provide a valid path.")

#     output_prefix = input("Enter a name for your output report (e.g., My_Data_Report): ")
#     if not output_prefix:
#         output_prefix = os.path.splitext(os.path.basename(file_path))[0] + "_Report"
#         print(f"ℹ️ No name provided. Using default: '{output_prefix}'")

#     sample_size = None
#     if input("Dataset large? Use random sampling for faster plotting? (yes/no): ").lower() in ['y', 'yes']:
#         try:
#             sample_size = int(input("Enter sample size (e.g., 50000): "))
#         except ValueError:
#             print("⚠️ Invalid number. Sampling will be skipped.")
#             sample_size = None
    
#     # --- MODIFICATION: Ask user to select which plots to generate ---
#     plot_choices = {
#         "1": ("Basic Statistics", "basic_statistics"),
#         "2": ("Distribution Plots", "distribution_plots"),
#         "3": ("Categorical Plots", "categorical_plots"),
#         "4": ("Pie Charts", "pie_charts"),
#         "5": ("Correlation Heatmap", "correlation_heatmap"),
#         "6": ("Scatter Plots", "scatter_plots"),
#         "7": ("Outlier Detection", "outlier_detection_plots"),
#         "8": ("Line Charts", "line_charts"),
#     }

#     print("\nSelect the plots you want to generate:")
#     for key, (desc, _) in plot_choices.items():
#         print(f"  {key}: {desc}")
    
#     selection_str = input("Enter numbers separated by commas (e.g., 1,3,5), or leave blank for all: ")
    
#     selected_methods = []
#     if not selection_str.strip():
#         selected_methods = [name for _, name in plot_choices.values()]
#         print("ℹ️ No selection made. Generating all plots.")
#     else:
#         for key in selection_str.split(','):
#             key = key.strip()
#             if key in plot_choices:
#                 selected_methods.append(plot_choices[key][1])
#             else:
#                 print(f"⚠️ Invalid selection '{key}' will be ignored.")

#     # 3. Get API Key if interpretations are wanted
#     api_key = None
#     if input("Enable AI-powered interpretations? (yes/no): ").lower() in ['y', 'yes']:
#         api_key = os.getenv("GEMINI_API_KEY") or getpass.getpass("Please enter your Gemini API key: ")
#         if not api_key:
#             print("⚠️ No API key provided. Running in plots-only mode.")

#     # 4. Instantiate Vizify and run the selected methods
#     try:
#         start_time = time.time()
        
#         viz = Vizify(file_path, output_prefix=output_prefix, api_key=api_key, sample_size=sample_size)
        
#         print("🚀 Starting visualization report generation...")
#         viz.pdf.add_page()
#         viz.pdf.chapter_title("Vizify Data Report")
#         viz.pdf.set_font('Helvetica', '', 12)
#         viz.pdf.multi_cell(0, 10, f"An automated analysis of the file: {viz.data.attrs.get('name', '')}\n"
#                                 f"Generated on: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')}")

#         # Call only the methods the user selected
#         for method_name in selected_methods:
#             method_to_call = getattr(viz, method_name)
#             method_to_call()
        
#         viz.pdf.output(viz.pdf_filename)
#         print(f"✅ All visualizations saved in {viz.pdf_filename}")
#         viz.generate_html_report()
        
#         end_time = time.time()
#         duration = end_time - start_time
#         print("-" * 60)
#         print(f"🎉 Report generation complete!")
#         print(f"Total time taken: {duration:.2f} seconds.")
#         print("-" * 60)

#     except Exception as e:
#         print(f"\n❌ An unexpected error occurred during report generation: {e}")
        
        
# vizify.py - All-in-One Data Analysis Tool

# --- Common Imports ---
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
import streamlit as st
import json
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# --- AI and PDF Generation Imports ---
try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPICallError
    from fpdf import FPDF
    PDF_LIBS_AVAILABLE = True
except ImportError:
    PDF_LIBS_AVAILABLE = False

# Optional dependencies
try:
    import streamlit as st
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

try:
    import google.generativeai as genai
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

# --- Configuration ---
warnings.filterwarnings("ignore")


# ==============================================================================
# SECTION 1: CORE AI AND PDF GENERATION LOGIC
# ==============================================================================

def interpret_graph_with_gemini(image_path: str, prompt: str = "Briefly describe this data visualization in 2-3 sentences, summarizing the key insight."):
    """Uses Gemini 1.5 Pro to generate a description for a given chart image."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        image_file = genai.upload_file(path=image_path)
        response = model.generate_content([prompt, image_file])
        return response.text
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
                genai.configure(api_key=api_key)
                self.use_llm = True
                print("✅ Gemini API key configured successfully. Interpretations are ENABLED.")
            except Exception as e:
                print(f"⚠️ Failed to configure Gemini API key: {e}. Interpretations will be DISABLED.")
        else:
            print("ℹ️ No API key provided. Interpretations are DISABLED.")

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
# SECTION 2: INTERACTIVE STREAMLIT DASHBOARD LOGIC
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
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            prompt = f"""
            You are a data analyst. A user has generated a plot titled '{plot_title}'.
            Based on the following sample of the data, provide a brief, insightful interpretation in 2-3 sentences.

            Data Sample:
            {df_sample.to_string()}
            """
            with st.spinner("🧠 Analyzing the chart with Gemini..."):
                response = model.generate_content(prompt)
                st.info(response.text)
        except Exception as e:
            st.error(f"Error generating interpretation: {e}")

    # ======================================================================
    # GLOBAL SLICERS (apply to entire dashboard)
    # ======================================================================
    def render_global_slicers(df):
        st.sidebar.subheader("🌐 Global Slicers")
        df = df.copy()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        dt_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()

        for col in cat_cols:
            vals = df[col].dropna().unique().tolist()
            default_vals = st.sidebar.multiselect(f"{col}", vals, default=vals)
            if default_vals:
                df = df[df[col].isin(default_vals)]

        for col in num_cols[:6]:
            col_min, col_max = float(df[col].min()), float(df[col].max())
            if np.isfinite(col_min) and np.isfinite(col_max) and col_min != col_max:
                r = st.sidebar.slider(f"{col} range", col_min, col_max, (col_min, col_max))
                df = df[(df[col] >= r[0]) & (df[col] <= r[1])]

        for col in dt_cols[:3]:
            min_d, max_d = df[col].min(), df[col].max()
            if pd.notna(min_d) and pd.notna(max_d) and min_d != max_d:
                r = st.sidebar.date_input(f"{col} range", (min_d.date(), max_d.date()))
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

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(df[[selected_col]].head(), fig.layout.title.text, st.session_state.api_key)

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

        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(counts, fig.layout.title.text, st.session_state.api_key)

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
        
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(df[[x_axis, y_axis]].head(), fig.layout.title.text, st.session_state.api_key)

    def render_heatmap(df, key_prefix):
        st.markdown("#### Correlation Heatmap")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) < 2:
            st.warning("At least two numerical columns are needed for a heatmap.")
            return

        corr = df[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Get AI Insight", key=f"{key_prefix}_ai_btn"):
            get_ai_interpretation(corr, fig.layout.title.text, st.session_state.api_key)

    PLOT_FUNCTIONS = {
        "Distribution Plot": render_distribution_plot,
        "Categorical Plot": render_categorical_plot,
        "Scatter Plot": render_scatter_plot,
        "Correlation Heatmap": render_heatmap,
    }

    # ======================================================================
    # NEW: FIGURE GENERATOR FUNCTIONS (for PDF export)
    # These recreate the figures without drawing Streamlit widgets
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

    PLOT_GENERATORS = {
        "Distribution Plot": generate_distribution_fig,
        "Categorical Plot": generate_categorical_fig,
        "Scatter Plot": generate_scatter_fig,
        "Correlation Heatmap": generate_heatmap_fig,
    }

    # ======================================================================
    # MAIN APP LAYOUT
    # ======================================================================
    st.session_state.setdefault("dashboard_items", [])
    st.session_state.setdefault("api_key", "")

    st.title("📊 Vizify: Build Your Own Dashboard")

    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded_file = st.file_uploader("1) Upload CSV", type="csv")
        st.session_state.api_key = st.text_input("2) Gemini API Key (optional for AI insights)", type="password")
        st.markdown("---")
        st.header("Add Charts to Dashboard")
        chart_type_to_add = st.selectbox("Select a chart type:", list(PLOT_FUNCTIONS.keys()))
        if st.button(f"➕ Add {chart_type_to_add}", use_container_width=True):
            st.session_state.dashboard_items.append({"type": chart_type_to_add, "id": time.time()})
            st.rerun()

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin building your dashboard.")
        return

    df_raw = pd.read_csv(uploaded_file)
    df_raw = coerce_datetime_cols(df_raw)
    df = render_global_slicers(df_raw)

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
    
    # --- UPDATED PDF Export option ---
    with util_c3:
        if st.button("📄 Download Dashboard as PDF"):
            if not REPORTLAB_AVAILABLE:
                st.error("PDF export requires `reportlab`. Please run `pip install reportlab`.")
                return

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

# ==============================================================================
# SECTION 3: MAIN LAUNCHER AND PDF-MODE CLI
# ==============================================================================

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
        run_dashboard()
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
            print("Launching Streamlit dashboard... Your browser will open shortly.")
            try:
                subprocess.run(["streamlit", "run", __file__, "run_dashboard"], check=True)
            except FileNotFoundError:
                print("❌ Error: 'streamlit' command not found.")
                print("Please install Streamlit and Plotly first: pip install streamlit plotly")
            except Exception as e:
                print(f"An error occurred while launching Streamlit: {e}")