import sqlite3
import os
import re
import math
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

# --- SECTION 1: SQLITE STORAGE ENGINE FOR LOW-MEMORY HARDWARE ---

class SQLiteDataEngine:
    """
    High-performance, low-memory storage engine backed by SQLite.
    Protects low-performance devices from RAM spikes by streaming datasets into disk/temp SQLite DB,
    performing pagination, aggregations, filtering, and stats directly in SQL.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path(os.getenv("VIZIFY_DATA_DIR", Path.home() / ".vizify"))
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "workspace.sqlite3")
        self.db_path = db_path
        self.raw_table = "dataset_raw"
        self.cleaned_table = "dataset_cleaned"
        self.filename = "dataset.csv"
        self.active_table = self.raw_table
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        conn.execute("CREATE TABLE IF NOT EXISTS vizify_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = conn.execute("SELECT value FROM vizify_metadata WHERE key='dataset' ").fetchone()
        if row:
            try:
                state = json.loads(row[0])
                self.filename = state.get("filename", self.filename)
                self.active_table = state.get("active_table", self.cleaned_table)
            except (TypeError, ValueError):
                pass
        conn.close()

    def has_dataset(self) -> bool:
        conn = self._get_connection()
        try:
            return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (self.active_table,)).fetchone() is not None
        finally:
            conn.close()

    def _save_state(self):
        conn = self._get_connection()
        conn.execute("INSERT OR REPLACE INTO vizify_metadata(key, value) VALUES('dataset', ?)",
                     (json.dumps({"filename": self.filename, "active_table": self.active_table}),))
        conn.commit()
        conn.close()

    def ingest_dataframe(self, df: pd.DataFrame, filename: str = "dataset.csv") -> Dict[str, Any]:
        """Streams a pandas DataFrame into SQLite table with automatic type mapping."""
        self.filename = filename
        conn = self._get_connection()
        
        sanitized_df = df.copy()
        sanitized_df.columns = [str(c).strip().replace(' ', '_').replace('.', '_').replace('-', '_') for c in sanitized_df.columns]
        
        sanitized_df.to_sql(self.raw_table, conn, if_exists='replace', index=False, chunksize=5000)
        sanitized_df.to_sql(self.cleaned_table, conn, if_exists='replace', index=False, chunksize=5000)
        
        self.active_table = self.cleaned_table
        self._create_indexes(conn, self.cleaned_table, sanitized_df)
        conn.execute("INSERT OR REPLACE INTO vizify_metadata(key, value) VALUES('dataset', ?)",
                     (json.dumps({"filename": self.filename, "active_table": self.active_table}),))
        conn.close()
        
        return self.get_metadata()

    def ingest_csv(self, file_path_or_buffer, filename: str = "dataset.csv", chunksize: int = 10000) -> Dict[str, Any]:
        """Streams CSV file in chunks directly into SQLite, keeping memory minimal."""
        self.filename = filename
        conn = self._get_connection()
        
        first_chunk = True
        for chunk in pd.read_csv(file_path_or_buffer, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip'):
            chunk.columns = [str(c).strip().replace(' ', '_').replace('.', '_').replace('-', '_') for c in chunk.columns]
            mode = 'replace' if first_chunk else 'append'
            chunk.to_sql(self.raw_table, conn, if_exists=mode, index=False)
            chunk.to_sql(self.cleaned_table, conn, if_exists=mode, index=False)
            first_chunk = False
            
        self.active_table = self.cleaned_table
        conn.execute("INSERT OR REPLACE INTO vizify_metadata(key, value) VALUES('dataset', ?)",
                     (json.dumps({"filename": self.filename, "active_table": self.active_table}),))
        conn.commit()
        conn.close()
        return self.get_metadata()

    def ingest_excel(self, file_path_or_buffer, filename: str = "dataset.xlsx") -> Dict[str, Any]:
        df = pd.read_excel(file_path_or_buffer)
        return self.ingest_dataframe(df, filename=filename)

    def ingest_sqlalchemy(self, connection_url: str, query: str, filename: str = "database query") -> Dict[str, Any]:
        """Read a user-selected SQL query in chunks and persist the result locally."""
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise RuntimeError("SQLAlchemy is required for database connections") from exc
        engine = create_engine(connection_url, pool_pre_ping=True)
        conn = self._get_connection()
        first_chunk = True
        try:
            for chunk in pd.read_sql_query(query, engine, chunksize=10000):
                chunk.columns = [str(c).strip().replace(' ', '_').replace('.', '_').replace('-', '_') for c in chunk.columns]
                mode = 'replace' if first_chunk else 'append'
                chunk.to_sql(self.raw_table, conn, if_exists=mode, index=False, chunksize=5000)
                chunk.to_sql(self.cleaned_table, conn, if_exists=mode, index=False, chunksize=5000)
                first_chunk = False
            if first_chunk:
                raise ValueError("The query returned no columns or rows")
            self.filename = filename
            self.active_table = self.cleaned_table
            conn.execute("INSERT OR REPLACE INTO vizify_metadata(key, value) VALUES('dataset', ?)",
                         (json.dumps({"filename": self.filename, "active_table": self.active_table}),))
            conn.commit()
        finally:
            conn.close()
            engine.dispose()
        return self.get_metadata()

    def _create_indexes(self, conn, table_name: str, sample_df: pd.DataFrame):
        try:
            cursor = conn.cursor()
            for col in sample_df.columns[:15]:
                safe_col = f'"{col}"'
                index_name = f"idx_{table_name}_{re.sub(r'[^a-zA-Z0-9]', '', col)}"
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({safe_col})")
            conn.commit()
        except Exception:
            pass

    def get_columns(self) -> List[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.active_table})")
        cols = [row['name'] for row in cursor.fetchall()]
        conn.close()
        return cols

    def get_column_types(self) -> Dict[str, str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.active_table})")
        cols = cursor.fetchall()
        conn.close()
        return {row['name']: row['type'] for row in cols}

    def get_row_count(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.active_table}")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_metadata(self) -> Dict[str, Any]:
        """Computes metadata and column health using lightweight SQL queries."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({self.active_table})")
        table_info = cursor.fetchall()
        cols = [row['name'] for row in table_info]
        total_rows = self.get_row_count()
        
        df_sample = pd.read_sql_query(f"SELECT * FROM {self.active_table} LIMIT 500", conn)
        
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
        time_cols = []
        for col in categorical_cols:
            try:
                pd.to_datetime(df_sample[col], errors='raise')
                time_cols.append(col)
            except Exception:
                pass
                
        numeric_ranges = {}
        for col in numeric_cols:
            try:
                cursor.execute(f'SELECT MIN("{col}"), MAX("{col}") FROM {self.active_table} WHERE "{col}" IS NOT NULL')
                row = cursor.fetchone()
                numeric_ranges[col] = [
                    float(row[0]) if row[0] is not None and math.isfinite(row[0]) else 0.0,
                    float(row[1]) if row[1] is not None and math.isfinite(row[1]) else 0.0
                ]
            except Exception:
                numeric_ranges[col] = [0.0, 0.0]

        categorical_values = {}
        for col in categorical_cols:
            try:
                cursor.execute(f'SELECT DISTINCT "{col}" FROM {self.active_table} WHERE "{col}" IS NOT NULL LIMIT 20')
                vals = [r[0] for r in cursor.fetchall()]
                categorical_values[col] = vals
            except Exception:
                categorical_values[col] = []

        columns_health = []
        for col in cols:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM {self.active_table} WHERE "{col}" IS NULL')
                null_count = cursor.fetchone()[0]
                null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
                columns_health.append({
                    'name': col,
                    'type': str(df_sample[col].dtype) if col in df_sample else 'text',
                    'non_null': total_rows - null_count,
                    'missing_pct': f"{null_pct:.1f}%"
                })
            except Exception:
                pass

        summary_stats = df_sample.describe(include='all').T.fillna('N/A').to_dict() if len(df_sample) > 0 else {}
        conn.close()

        return {
            'filename': self.filename,
            'shape': (total_rows, len(cols)),
            'columns': cols,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'time_cols': time_cols,
            'numeric_ranges': numeric_ranges,
            'categorical_values': categorical_values,
            'missing_count': sum(int(float(ch['missing_pct'].replace('%',''))*total_rows/100) for ch in columns_health),
            'missing_pct': 0.0,
            'duplicate_count': 0,
            'columns_health': columns_health,
            'summary_stats': summary_stats
        }

    def get_paginated_data(self, page: int = 1, limit: int = 50, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Returns paginated table rows without loading full dataset into memory."""
        offset = (page - 1) * limit
        conn = self._get_connection()
        
        where_clause, params = self._build_where_clause(filters)
        
        count_sql = f"SELECT COUNT(*) FROM {self.active_table} {where_clause}"
        cursor = conn.cursor()
        cursor.execute(count_sql, params)
        total_records = cursor.fetchone()[0]
        
        data_sql = f"SELECT * FROM {self.active_table} {where_clause} LIMIT ? OFFSET ?"
        params_with_paging = params + [limit, offset]
        
        df_page = pd.read_sql_query(data_sql, conn, params=params_with_paging)
        conn.close()
        
        return {
            'data': df_page.fillna('').to_dict(orient='records'),
            'columns': df_page.columns.tolist(),
            'page': page,
            'limit': limit,
            'total_records': total_records,
            'total_pages': math.ceil(total_records / limit) if limit > 0 else 1
        }

    def _build_where_clause(self, filters: Optional[Dict]) -> Tuple[str, List[Any]]:
        if not filters:
            return "", []
            
        conditions = []
        params = []
        
        categorical = filters.get('categorical', {})
        for col, vals in categorical.items():
            if vals:
                placeholders = ','.join(['?'] * len(vals))
                conditions.append(f'"{col}" IN ({placeholders})')
                params.extend(vals)
                
        numeric = filters.get('numeric', {})
        for col, r in numeric.items():
            if r and len(r) == 2:
                conditions.append(f'"{col}" >= ? AND "{col}" <= ?')
                params.extend([r[0], r[1]])
                
        if conditions:
            return "WHERE " + " AND ".join(conditions), params
        return "", []

    def get_dataframe(self, limit: Optional[int] = 10000, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Loads a subset DataFrame for ML/Plotting, enforcing memory bounds."""
        conn = self._get_connection()
        where_clause, params = self._build_where_clause(filters)
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"SELECT * FROM {self.active_table} {where_clause} {limit_clause}"
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df

    def update_table_data(self, df_updated: pd.DataFrame):
        """Updates the cleaned dataset table in SQLite."""
        conn = self._get_connection()
        df_updated.to_sql(self.cleaned_table, conn, if_exists='replace', index=False, chunksize=5000)
        self.active_table = self.cleaned_table
        conn.execute("INSERT OR REPLACE INTO vizify_metadata(key, value) VALUES('dataset', ?)",
                     (json.dumps({"filename": self.filename, "active_table": self.active_table}),))
        self._create_indexes(conn, self.cleaned_table, df_updated)
        conn.close()

    # --- SQL AGGREGATION HELPERS FOR THE 20 CHART TYPES ---
    def get_category_counts(self, col: str, limit: int = 25) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        sql = f'SELECT "{col}" as label, COUNT(*) as count FROM {self.active_table} WHERE "{col}" IS NOT NULL GROUP BY "{col}" ORDER BY count DESC LIMIT ?'
        try:
            cursor.execute(sql, (limit,))
            rows = [{'label': str(r['label']), 'value': int(r['count'])} for r in cursor.fetchall()]
        except Exception:
            rows = []
        conn.close()
        return rows

    def get_funnel_data(self, stage_col: str, val_col: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        if val_col:
            sql = f'SELECT "{stage_col}" as stage, SUM("{val_col}") as val FROM {self.active_table} WHERE "{stage_col}" IS NOT NULL GROUP BY "{stage_col}" ORDER BY val DESC'
        else:
            sql = f'SELECT "{stage_col}" as stage, COUNT(*) as val FROM {self.active_table} WHERE "{stage_col}" IS NOT NULL GROUP BY "{stage_col}" ORDER BY val DESC'
        try:
            cursor.execute(sql)
            rows = [{'stage': str(r['stage']), 'value': float(r['val'] or 0)} for r in cursor.fetchall()]
        except Exception:
            rows = []
        conn.close()
        return rows

    def get_sunburst_data(self, level1: str, level2: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        if level2:
            sql = f'SELECT "{level1}" as p, "{level2}" as c, COUNT(*) as val FROM {self.active_table} WHERE "{level1}" IS NOT NULL AND "{level2}" IS NOT NULL GROUP BY "{level1}", "{level2}" ORDER BY val DESC LIMIT 50'
        else:
            sql = f'SELECT "{level1}" as p, "All" as c, COUNT(*) as val FROM {self.active_table} WHERE "{level1}" IS NOT NULL GROUP BY "{level1}" ORDER BY val DESC LIMIT 20'
        try:
            cursor.execute(sql)
            rows = [{'parent': str(r['p']), 'child': str(r['c']), 'value': int(r['val'])} for r in cursor.fetchall()]
        except Exception:
            rows = []
        conn.close()
        return rows



# --- SECTION 2: NLP TEXT ANALYSIS & CLEANING MODULE ---

class NLPProcessor:
    """
    Advanced NLP Engine for Text Diagnostics, Sentiment Polarity, N-grams, TF-IDF, Entity Extraction, and Text Cleaning.
    """
    STOPWORDS = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])

    POSITIVE_WORDS = set(["good", "great", "excellent", "awesome", "amazing", "wonderful", "fantastic", "positive", "love", "like", "best", "super", "happy", "joy", "benefit", "boost", "growth", "high", "success", "effective", "clean", "smooth", "perfect", "satisfied"])
    NEGATIVE_WORDS = set(["bad", "poor", "terrible", "horrible", "awful", "negative", "hate", "dislike", "worst", "fail", "failure", "issue", "problem", "error", "delay", "slow", "risk", "damage", "loss", "defect", "broken", "critical", "refund", "complaint"])

    @staticmethod
    def detect_text_columns(df: pd.DataFrame) -> List[str]:
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().astype(str)
                if len(sample) > 0:
                    avg_len = sample.str.len().mean()
                    if avg_len >= 3:
                        text_cols.append(col)
        return text_cols

    @staticmethod
    def analyze_text(df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Comprehensive NLP diagnostics on a target text column."""
        if col not in df.columns:
            return {'error': f"Column '{col}' not found"}
            
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            return {'error': 'Selected column has no valid text data'}
            
        char_counts = series.str.len()
        word_counts = series.apply(lambda s: len(s.split()))
        
        metrics = {
            'total_rows': len(series),
            'avg_char_length': round(float(char_counts.mean()), 2),
            'max_char_length': int(char_counts.max()),
            'min_char_length': int(char_counts.min()),
            'avg_word_count': round(float(word_counts.mean()), 2),
            'total_words': int(word_counts.sum()),
        }

        words = []
        sentences = []
        for text in series:
            s_words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            words.extend(s_words)
            sentences.append(s_words)
            
        vocab_richness = round(len(set(words)) / len(words), 4) if words else 0.0
        metrics['vocabulary_richness'] = vocab_richness

        filtered_words = [w for w in words if w not in NLPProcessor.STOPWORDS]
        word_freq = Counter(filtered_words).most_common(50)
        
        wordcloud_data = [{'text': w, 'value': count} for w, count in word_freq]

        tfidf_data = []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
            matrix = vectorizer.fit_transform(series.tolist())
            terms = vectorizer.get_feature_names_out()
            scores = np.asarray(matrix.mean(axis=0)).ravel()
            top = np.argsort(scores)[-20:][::-1]
            tfidf_data = [{'text': str(terms[i]), 'value': round(float(scores[i]), 5)} for i in top]
        except (ValueError, ImportError):
            # Empty or one-token corpora may not produce a vocabulary.
            tfidf_data = []

        bigrams = []
        trigrams = []
        for s in sentences:
            filtered_s = [w for w in s if w not in NLPProcessor.STOPWORDS]
            for i in range(len(filtered_s) - 1):
                bigrams.append(f"{filtered_s[i]} {filtered_s[i+1]}")
            for i in range(len(filtered_s) - 2):
                trigrams.append(f"{filtered_s[i]} {filtered_s[i+1]} {filtered_s[i+2]}")

        bigram_freq = Counter(bigrams).most_common(15)
        trigram_freq = Counter(trigrams).most_common(15)

        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        compound_scores = []
        for s in sentences:
            pos_c = sum(1 for w in s if w in NLPProcessor.POSITIVE_WORDS)
            neg_c = sum(1 for w in s if w in NLPProcessor.NEGATIVE_WORDS)
            tot = len(s) if len(s) > 0 else 1
            score = (pos_c - neg_c) / tot
            compound_scores.append(score)
            
            if score > 0.05:
                sentiment_scores['positive'] += 1
            elif score < -0.05:
                sentiment_scores['negative'] += 1
            else:
                sentiment_scores['neutral'] += 1

        joined_text = " ".join(series.tolist()[:1000])
        emails = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', joined_text)))[:10]
        urls = list(set(re.findall(r'https?://[^\s]+', joined_text)))[:10]
        dates = list(set(re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b', joined_text)))[:10]
        phones = list(set(re.findall(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', joined_text)))[:10]

        entities = {
            'emails': emails,
            'urls': urls,
            'dates': dates,
            'phones': phones
        }

        nodes = [{'id': w, 'group': 1, 'radius': count} for w, count in word_freq[:20]]
        node_set = set(w['id'] for w in nodes)
        links = []
        link_counts = Counter()
        
        for s in sentences:
            filtered_s = [w for w in s if w in node_set]
            for i in range(len(filtered_s)):
                for j in range(i + 1, len(filtered_s)):
                    pair = tuple(sorted([filtered_s[i], filtered_s[j]]))
                    link_counts[pair] += 1
                    
        for (source, target), weight in link_counts.most_common(30):
            links.append({'source': source, 'target': target, 'value': weight})

        network_data = {'nodes': nodes, 'links': links}

        return {
            'column': col,
            'metrics': metrics,
            'wordcloud': wordcloud_data,
            'tfidf': tfidf_data,
            'bigrams': [{'text': b, 'value': count} for b, count in bigram_freq],
            'trigrams': [{'text': t, 'value': count} for t, count in trigram_freq],
            'sentiment': sentiment_scores,
            'avg_sentiment_score': round(float(np.mean(compound_scores)), 3) if compound_scores else 0.0,
            'entities': entities,
            'network': network_data
        }

    @staticmethod
    def clean_text_series(series: pd.Series, options: Dict[str, Any]) -> pd.Series:
        cleaned = series.fillna('').astype(str)
        
        if options.get('lower', False):
            cleaned = cleaned.str.lower()
            
        if options.get('remove_html', False):
            cleaned = cleaned.apply(lambda x: re.sub(r'<[^>]+>', ' ', x))
            
        if options.get('remove_urls', False):
            cleaned = cleaned.apply(lambda x: re.sub(r'https?://\S+|www\.\S+', ' ', x))
            
        if options.get('remove_digits', False):
            cleaned = cleaned.apply(lambda x: re.sub(r'\d+', ' ', x))
            
        if options.get('remove_punct', False):
            cleaned = cleaned.apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
            
        if options.get('custom_regex'):
            pattern = options.get('custom_regex')
            repl = options.get('custom_replace', '')
            try:
                cleaned = cleaned.apply(lambda x: re.sub(pattern, repl, x))
            except Exception:
                pass

        if options.get('remove_stopwords', False):
            sw = NLPProcessor.STOPWORDS
            cleaned = cleaned.apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in sw]))
            
        if options.get('normalize_space', False):
            cleaned = cleaned.apply(lambda x: ' '.join(x.split()))
            
        return cleaned
