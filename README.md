# Vizify

Vizify is a Python data analysis library with a local interactive dashboard for exploring datasets, visualizing patterns, cleaning text, and training basic machine learning models.

## Features

- **Charts and themes:** D3-powered interactive charts with dark and light themes; Plotly is available for supported charts.
- **Text analysis and cleaning:** word and n-gram frequencies, rule-based sentiment estimates, pattern extraction, TF-IDF terms, and configurable text cleaning.
- **SQLite workspace:** imported datasets and query results are persisted locally for retrieval. CSV and SQL query results are ingested in chunks; table browsing is paginated.
- **File sources:** CSV, TSV, TXT, Excel (`.xlsx`, `.xls`), JSON/JSONL, XML, PDF text, and Parquet.
- **Database sources:** MySQL, PostgreSQL, SQL Server/Azure SQL, Oracle, IBM Db2, Databricks SQL Warehouse, and SQL Server ODBC. Imports are local snapshots.
- **Online and web sources:** OData JSON feeds, GitHub repository data, and HTML tables from public web pages.
- **Profiling and cleaning:** column summaries, missing values, filters, and a Magic Clean operation.
- **Machine learning:** basic regression and classification training, evaluation, live predictions, and model export.
- **Data Assistant:** optional Gemini-powered analysis. A valid Gemini API key is required.

The source picker also lists SharePoint, Salesforce, Dynamics 365, Google Analytics, Mailchimp, Microsoft Exchange, OLE DB, Active Directory, and R/Python scripts as planned integrations. They are marked unavailable in the UI until their authentication/driver workflows are implemented. DirectQuery is not currently supported; database connections import a snapshot into SQLite.

## Install

```bash
pip install vizify
```

Install optional file readers or database drivers with extras:

```bash
pip install 'vizify[files]'       # Parquet and PDF readers
pip install 'vizify[mysql]'       # MySQL driver
pip install 'vizify[postgres]'    # PostgreSQL driver
pip install 'vizify[sqlserver]'   # SQL Server ODBC driver (system ODBC driver also required)
pip install 'vizify[oracle]'
pip install 'vizify[db2]'
pip install 'vizify[databricks]'
pip install 'vizify[all]'         # all optional readers and database drivers
```

If an existing installation reports `No module named 'pymysql'`, run `python -m pip install --upgrade 'vizify[mysql]'` in the same Python environment used to launch Vizify, then restart the dashboard.

For development, install the Python dependencies and build the bundled dashboard:

```bash
pip install -r requirements.txt
cd frontend
npm install
npm run build
```

The dashboard is bundled into `vizify/static` and served by the local Flask server.

## Gemini API key

Vizify checks for a key in this order:

1. `GEMINI_API_KEY` environment variable.
2. The path in `VIZIFY_API_KEY_FILE`, if set.
3. A key entered in the dashboard sidebar for the current session.

`API.key` is reserved for PyPI publishing and is never used as a Gemini key or included in the Python package. A PyPI token (`pypi-…`) is not a Gemini API key. Keep publishing tokens and API key files out of source control.

PowerShell example:

```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
```

## Launch the dashboard

```python
from vizify import Vizify

Vizify.launch_dashboard("sales.csv")
```

Or pass a pandas dataframe:

```python
import pandas as pd
from vizify import Vizify

data = pd.read_excel("sales.xlsx")
Vizify.launch_dashboard(df=data)
```

Use **Get data** in the dashboard to open the source picker. Pick a file, configure a database connection, or select an online/web source. Database credentials are used only for the import request and are not stored in the SQLite workspace. The loaded result is a snapshot; connect again to refresh it.

## Database connection examples

Choose the provider in **Get data → Databases**, enter its connection details, and provide a read-only `SELECT` query. Example:

```sql
SELECT region, order_date, amount
FROM sales
WHERE order_date >= '2025-01-01'
```

Databricks requires the workspace hostname, access token, and SQL warehouse HTTP path, with optional catalog and schema. MySQL, PostgreSQL, SQL Server, Oracle, and Db2 use their respective host, port, database, username, and password fields. SQL Server ODBC accepts a driver connection string.

## SQLite workspace

By default, Vizify stores the active dataset in:

```text
~/.vizify/workspace.sqlite3
```

Set `VIZIFY_DATA_DIR` to choose another directory. The active dataset is restored when the dashboard restarts. Table browsing and filtering use SQLite; full-data charting, text analysis, and model training load data into pandas as needed, so those operations still use memory proportional to the data being analyzed.

The engine and NLP processor are also available directly:

```python
from vizify import SQLiteDataEngine, NLPProcessor

engine = SQLiteDataEngine()
metadata = engine.ingest_csv("sales.csv", filename="sales.csv")
page = engine.get_paginated_data(page=1, limit=50)
```

## Python report example

```python
from vizify import Vizify

report = Vizify("sales.csv", api_key=None)  # Uses configured Gemini key when available
report.show_all_visualizations()
```

## Package contents

The Python distribution contains the dashboard frontend build under `vizify/static`. It does not contain local datasets, SQLite workspaces, or API key files.
