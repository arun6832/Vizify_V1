# Vizify

## Overview
Vizify is a Python package designed to automate the process of data visualization. It generates a comprehensive PDF report containing various visual representations of a dataset, helping users analyze numerical and categorical data efficiently.

## Features
- **Basic Statistics:** Summarizes key numerical insights.
- **Correlation Heatmap:** Displays relationships between numerical features.
- **Distribution Plots:** Visualizes data distributions with histograms.
- **Box Plots:** Highlights outliers and data spread.
- **Scatter Plots:** Shows relationships between numerical variables.
- **Violin Plots:** Combines box plot and density estimates.
- **Word Cloud:** Extracts textual insights from categorical features.
- **Outlier Detection:** Identifies extreme values using box plots.
- **Stacked Bar Charts:** Compares categorical data distributions.
- **Line Plots:** Represents trends over time.
- **Pie Charts:** Displays categorical distributions.
- **Automated PDF Generation:** Saves all visualizations in a structured PDF report.

## Installation
To install Vizify, use pip:
```bash
pip install vizify
```

## Usage
### Import the Package
```python
from vizify import Vizify
```

### Generate a Visualization Report
```python
# Initialize Vizify with a CSV file
viz = Vizify("data.csv")

# Generate and save all visualizations in a PDF report
viz.show_all_visualizations()
```

Upon execution, a file named `data_visualization_report.pdf` will be created in the working directory containing all visualizations.

## Dependencies
Vizify requires the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- missingno
- wordcloud

These dependencies will be installed automatically with the package.

## License
Vizify is released under the [MIT License](LICENSE.txt).

## Contribution
Contributions are welcome! If you would like to improve the package, please fork the repository, make your changes, and submit a pull request.

## Contact
For inquiries, please contact:
- **Author:** Arun M
- **Email:** arunpappulli@gmail.com
- **GitHub:** [GitHub Profile](https://github.com/arun6832)

---

Thank you for using Vizify! 