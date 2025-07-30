# Vizify

## Overview
**Vizify** is a Python package designed to automate the process of data visualization and analysis. It generates a comprehensive PDF report containing a wide variety of plots and visual summaries, helping users quickly explore both numerical and categorical data.

Whether you're a data scientist, analyst, or beginner, Vizify speeds up exploratory data analysis and makes your datasets visually insightful with minimal effort.

---

## ✨ Features

- **Basic Statistics:** Summarizes key numerical insights.
- **Correlation Heatmap:** Displays relationships between numerical features.
- **Distribution Plots:** Visualizes data distributions with histograms.
- **Box Plots:** Highlights outliers and data spread.
- **Scatter Plots:** Shows relationships between numerical variables.
- **Violin Plots:** Combines box plots and density estimates.
- **Word Cloud:** Extracts textual insights from categorical features.
- **Outlier Detection:** Identifies extreme values using box plots.
- **Stacked Bar Charts:** Compares categorical data distributions.
- **Line Plots:** Represents trends over time.
- **Pie Charts:** Displays categorical distributions.
- **Time-Series Analysis:** Visualizes and interprets temporal trends in date/time features.
- **Anomaly Detection:** Detects and visualizes unusual patterns or outliers in time-series data.
- **Automated PDF Generation:** Saves all visualizations in a structured PDF report.
- **AI-Powered Chart Insights:** Uses Gemini LLM to automatically interpret charts and provide natural-language explanations.

---

## 📦 Installation

Install Vizify using pip:

```bash
pip install vizify
```

---

## 🚀 Usage

### 1. Import the Package

```python
from vizify import Vizify
```

### 2. Generate a Basic Visualization Report

```python
from vizify import Vizify

# Initialize Vizify with a CSV file
viz = Vizify("your_data.csv")
viz.show_all_visualizations()
```

After execution, a file named `data_visualization_report.pdf` will be created in your working directory containing all the visualizations.

### 3. Generate a Visualization Report with AI Interpretation

```python
from vizify import Vizify

viz = Vizify("your_data.csv", api_key="YOUR_GEMINI_API_KEY")#Place your API KEY here
viz.show_all_visualizations()
```

After execution, a file named `data_visualization_report.pdf` will be created in your working directory containing all the visualizations.
---

## ⚙️ Optional Usage

You can also use specific methods for targeted visualizations:

```python
viz.plot_correlation_heatmap()
viz.plot_time_series(column="date_column", value="sales")
viz.detect_anomalies(column="sales", method="iqr")
```

---

## 📚 Dependencies

Vizify requires the following Python libraries:

- [pandas](https://pypi.org/project/pandas/)
- [numpy](https://pypi.org/project/numpy/)
- [seaborn](https://pypi.org/project/seaborn/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [missingno](https://pypi.org/project/missingno/)
- [wordcloud](https://pypi.org/project/wordcloud/)

These will be automatically installed when you install Vizify.

---

## 📈 Upcoming Features

- **Interactive HTML Export:** Export the report as a rich, browser-based dashboard.
- **AI-Powered Graph Insights:** Use open-source LLMs to automatically interpret charts.
- **Custom Theme Support:** Dark/light mode and aesthetic customizations.
- **Drill-Down Interactivity:** Clickable charts for deeper data exploration.

---

## 🧑‍💻 Contribution

Contributions are welcome! To contribute:

1. Fork the repository  
2. Create a new branch (`git checkout -b feature-xyz`)  
3. Commit your changes  
4. Push to the branch  
5. Submit a pull request  

---

## 📄 License

Vizify is released under the [MIT License](LICENSE.txt).

---

## 📬 Contact

- **Author:** Arun M  
- **Email:** arunpappulli@gmail.com  
- **GitHub:** [arun6832](https://github.com/arun6832)

---

Thank you for using **Vizify**! 🚀
