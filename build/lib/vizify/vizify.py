import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings("ignore")

class Vizify:
    def __init__(self, file_path):
        """Initialize Vizify with the dataset."""
        try:
            self.data = pd.read_csv(file_path, encoding="utf-8")  # Default UTF-8
        except UnicodeDecodeError:
            print("⚠️ UTF-8 decoding failed! Trying 'ISO-8859-1' encoding...")
            self.data = pd.read_csv(file_path, encoding="ISO-8859-1")  # Alternative encoding

        self.pdf_filename = "data_visualization_report.pdf"
        self.num_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()

    def add_page_title(self, pdf, title):
        """Helper function to add a title page in the PDF."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, title, fontsize=16, fontweight="bold", ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    def basic_statistics(self, pdf):
        """Generate basic statistics for numerical columns."""
        self.add_page_title(pdf, "Basic Statistics")
        stats = self.data.describe().T
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("tight")
        ax.axis("off")
        ax.table(cellText=stats.values, colLabels=stats.columns, rowLabels=stats.index, loc="center")
        pdf.savefig(fig)
        plt.close(fig)

    def correlation_heatmap(self, pdf):
        """Generate a correlation heatmap for numerical columns."""
        if len(self.num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(self.data[self.num_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Heatmap")
            pdf.savefig(fig)
            plt.close(fig)

    def distribution_plots(self, pdf):
        """Generate distribution plots for numerical columns."""
        self.add_page_title(pdf, "Distribution Plots")
        for col in self.num_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(self.data[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            pdf.savefig(fig)
            plt.close(fig)

    def box_plots(self, pdf):
        """Generate box plots for numerical columns."""
        self.add_page_title(pdf, "Box Plots")
        for col in self.num_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=self.data[col], ax=ax)
            ax.set_title(f"Box Plot of {col}")
            pdf.savefig(fig)
            plt.close(fig)

    def scatter_plots(self, pdf):
        """Generate scatter plots for numerical variables."""
        self.add_page_title(pdf, "Scatter Plots")
        for i in range(len(self.num_cols) - 1):
            for j in range(i + 1, len(self.num_cols)):
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x=self.data[self.num_cols[i]], y=self.data[self.num_cols[j]], ax=ax)
                ax.set_title(f"Scatter Plot of {self.num_cols[i]} vs {self.num_cols[j]}")
                pdf.savefig(fig)
                plt.close(fig)

    def violin_plots(self, pdf):
        """Generate violin plots for numerical columns."""
        self.add_page_title(pdf, "Violin Plots")
        for col in self.num_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(y=self.data[col], ax=ax)
            ax.set_title(f"Violin Plot of {col}")
            pdf.savefig(fig)
            plt.close(fig)

    def wordcloud(self, pdf):
        """Generate a word cloud for text data."""
        if len(self.cat_cols) > 0:
            text_data = " ".join(self.data[self.cat_cols].astype(str).values.flatten())
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Word Cloud of Categorical Data")
            pdf.savefig(fig)
            plt.close(fig)

    def outlier_detection(self, pdf):
        """Detect outliers in numerical columns using boxplots."""
        self.add_page_title(pdf, "Outlier Detection")
        for col in self.num_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(y=self.data[col], ax=ax)
            ax.set_title(f"Outliers in {col}")
            pdf.savefig(fig)
            plt.close(fig)

    def pie_chart(self, pdf):
        """Generate pie charts for categorical variables."""
        for cat_col in self.cat_cols:
            value_counts = self.data[cat_col].value_counts()
            if value_counts.empty:
                continue
            plt.figure(figsize=(8, 8))
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
                    startangle=140, colors=sns.color_palette("pastel"))
            plt.title(f"Pie Chart for {cat_col}", fontsize=12, fontweight="bold")
            plt.axis('equal')
            pdf.savefig()
            plt.close()

    def show_all_visualizations(self, output_pdf="data_visualization_report.pdf"):
        """Generate all visualizations and save them to a single PDF."""
        self.pdf_filename = output_pdf
        with PdfPages(self.pdf_filename) as pdf:
            self.basic_statistics(pdf)
            self.correlation_heatmap(pdf)
            self.distribution_plots(pdf)
            self.box_plots(pdf)
            self.scatter_plots(pdf)
            self.violin_plots(pdf)
            self.wordcloud(pdf)
            self.outlier_detection(pdf)
            self.pie_chart(pdf)
            print(f"✅ All visualizations saved in {self.pdf_filename}")