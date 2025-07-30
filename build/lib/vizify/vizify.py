import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import tempfile
import getpass
import time  # Import the time module
from itertools import combinations

# Import the Google Generative AI library
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

# Import the PDF library
from fpdf import FPDF

# --- Configuration ---
warnings.filterwarnings("ignore")


# --- Core Functions ---

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


# --- PDF Styling Class ---

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


# --- Main Vizify Class ---

class Vizify:
    """
    An automated tool for generating data visualization reports from CSV files.
    """
    def __init__(self, file_path, output_prefix="Plots_Report", api_key=None,sample_size=None):
        """
        Initializes the Vizify analysis tool.

        Args:
            file_path (str): The path to the CSV data file.
            output_prefix (str): The prefix for the output PDF and HTML files.
            api_key (str, optional): A Google AI Studio API key.
                                     If provided, enables plot interpretations.
                                     If None, only plots are generated.
        """
        try:
            self.data = pd.read_csv(file_path, encoding="utf-8", on_bad_lines='skip')
        except FileNotFoundError:
            print(f"❌ Error: The file at {file_path} was not found.")
            raise
        self.data.attrs['name'] = os.path.basename(file_path)
        self.pdf_filename = f"{output_prefix}.pdf"
        self.html_filename = f"{output_prefix}.html"
        self.num_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
        self.time_cols = self._find_time_cols()
        self.pdf = PDF('P', 'mm', 'A4')
        self.pdf.set_auto_page_break(auto=True, margin=20)

        # Configure API key and determine if interpretations are enabled
        self.use_llm = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.use_llm = True
                print("✅ Gemini API key configured successfully. Interpretations are ENABLED.")
            except Exception as e:
                print(f"⚠️ Failed to configure Gemini API key: {e}. Interpretations will be DISABLED.")
        else:
            print("ℹ️ No API key provided. Interpretations are DISABLED.")

    def _find_time_cols(self):
        """Identifies and converts date/time columns in the dataframe."""
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
        """A helper function to save a plot to a temp file and add it to the PDF."""
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
        """Adds a summary statistics table to the PDF."""
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
        """Generates and adds distribution plots for all numerical columns."""
        if not self.num_cols: return
        self.pdf.add_page()
        self.pdf.chapter_title("Univariate Analysis: Numerical")
        for col in self.num_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(self.data[col].dropna(), kde=True, ax=ax, color="skyblue", bins=30)
            plt.tight_layout()
            self._add_plot_to_pdf(fig, f"Distribution of {col}")

    def categorical_plots(self):
        """Generates and adds count plots for categorical columns."""
        if not self.cat_cols: return
        self.pdf.add_page()
        self.pdf.chapter_title("Univariate Analysis: Categorical")
        for col in self.cat_cols:
            if 1 < self.data[col].nunique() < 30:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=self.data[col].astype(str), ax=ax, palette="pastel", order=self.data[col].value_counts().index)
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                                ha='left', va='center', fontsize=8, color='gray', xytext=(5, 0), textcoords='offset points')
                plt.tight_layout()
                self._add_plot_to_pdf(fig, f"Count of {col}")

    def pie_charts(self):
        """Generates and adds pie charts for categorical columns with few unique values."""
        if not self.cat_cols: return
        self.pdf.add_page()
        self.pdf.chapter_title("Proportional Analysis: Pie Charts")
        for col in self.cat_cols:
            if 1 < self.data[col].nunique() < 10:
                fig, ax = plt.subplots(figsize=(8, 8))
                counts = self.data[col].value_counts()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
                       colors=sns.color_palette("pastel"), wedgeprops={"edgecolor": "white"})
                ax.axis('equal')
                self._add_plot_to_pdf(fig, f"Pie Chart for {col}")

    def correlation_heatmap(self):
        """Generates and adds a correlation heatmap."""
        if len(self.num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = self.data[self.num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="viridis", linewidths=0.5, ax=ax, annot_kws={"size": 8})
            plt.tight_layout()
            self._add_plot_to_pdf(fig, "Correlation Heatmap")

    def scatter_plots(self):
        """Generates and adds scatter plots for pairs of numerical columns."""
        if len(self.num_cols) < 2: return
        self.pdf.add_page()
        self.pdf.chapter_title("Bivariate Analysis: Scatter Plots")
        for col1, col2 in combinations(self.num_cols, 2):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=self.data, x=col1, y=col2, ax=ax, alpha=0.6)
            plt.tight_layout()
            self._add_plot_to_pdf(fig, f"Scatter Plot: {col1} vs {col2}")

    def outlier_detection_plots(self):
        """Generates plots highlighting outliers in numerical columns."""
        if not self.num_cols: return
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
        """Generates line charts for time-series data."""
        if not self.time_cols or not self.num_cols: return
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
        """Generates a simple HTML file to embed and display the PDF report."""
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

    def show_all_visualizations(self):
        """Runs all analysis and plotting functions to generate the final report."""
        print("🚀 Starting visualization report generation...")
        
        self.pdf.add_page()
        self.pdf.chapter_title("Vizify Data Report")
        self.pdf.set_font('Helvetica', '', 12)
        self.pdf.multi_cell(0, 10, f"An automated analysis of the file: {self.data.attrs.get('name', '')}\n"
                                f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.basic_statistics()
        self.distribution_plots()
        self.categorical_plots()
        self.pie_charts()
        self.correlation_heatmap()
        self.scatter_plots()
        self.outlier_detection_plots()
        self.line_charts()
        
        self.pdf.output(self.pdf_filename)
        print(f"✅ All visualizations saved in {self.pdf_filename}")
        
        self.generate_html_report()


if __name__ == "__main__":
    print("=" * 60)
    print("      Welcome to Vizify: The Automated Visualization Reporter!      ")
    print("=" * 60)

    # 1. Get user inputs for file path, report name, and sampling
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
    
    # --- MODIFICATION: Ask user to select which plots to generate ---
    plot_choices = {
        "1": ("Basic Statistics", "basic_statistics"),
        "2": ("Distribution Plots", "distribution_plots"),
        "3": ("Categorical Plots", "categorical_plots"),
        "4": ("Pie Charts", "pie_charts"),
        "5": ("Correlation Heatmap", "correlation_heatmap"),
        "6": ("Scatter Plots", "scatter_plots"),
        "7": ("Outlier Detection", "outlier_detection_plots"),
        "8": ("Line Charts", "line_charts"),
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

    # 3. Get API Key if interpretations are wanted
    api_key = None
    if input("Enable AI-powered interpretations? (yes/no): ").lower() in ['y', 'yes']:
        api_key = os.getenv("GEMINI_API_KEY") or getpass.getpass("Please enter your Gemini API key: ")
        if not api_key:
            print("⚠️ No API key provided. Running in plots-only mode.")

    # 4. Instantiate Vizify and run the selected methods
    try:
        start_time = time.time()
        
        viz = Vizify(file_path, output_prefix=output_prefix, api_key=api_key, sample_size=sample_size)
        
        print("🚀 Starting visualization report generation...")
        viz.pdf.add_page()
        viz.pdf.chapter_title("Vizify Data Report")
        viz.pdf.set_font('Helvetica', '', 12)
        viz.pdf.multi_cell(0, 10, f"An automated analysis of the file: {viz.data.attrs.get('name', '')}\n"
                                f"Generated on: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Call only the methods the user selected
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