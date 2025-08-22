import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import boto3
import json
import PyPDF2
import io
import re

# ----------------------
# Initialization
# ----------------------

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Default example text (no special characters so it passes validation)
default_text = "I love this product Its amazing and works perfectly"

# ----------------------
# Secrets & API Keys
# ----------------------

# Default values
hf_token = ""
aws_key = ""
aws_secret = ""
aws_region = "us-east-1"

# Load Hugging Face token from Streamlit secrets
hf_token = st.secrets.get("HF_TOKEN", "")

# Load AWS credentials from Streamlit secrets
aws_key = st.secrets.get("AWS_ACCESS_KEY", "")
aws_secret = st.secrets.get("AWS_SECRET_KEY", "")
aws_region = st.secrets.get("AWS_REGION", "us-east-1")

# ----------------------
# Sidebar configuration
# ----------------------
st.sidebar.header("Configuration")
use_vader = st.sidebar.checkbox("Use VADER", value=True, help="Free offline sentiment analysis")
use_hf = st.sidebar.checkbox("Use Hugging Face", value=False, help="Requires API token")
use_aws = st.sidebar.checkbox("Use AWS Comprehend", value=False, help="Requires AWS credentials")

# Fallback to sidebar inputs if secrets are missing
if use_hf and not hf_token:
    hf_token = st.sidebar.text_input(
        "Hugging Face API Token",
        type="password",
        help="Get free token at https://huggingface.co/settings/tokens"
    )

if use_aws and (not aws_key or not aws_secret):
    aws_key = st.sidebar.text_input("AWS Access Key", type="password")
    aws_secret = st.sidebar.text_input("AWS Secret Key", type="password")
    aws_region = st.sidebar.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"])

# ----------------------
# Helpers: Validation & Keyword Sentiment
# ----------------------

VALID_TEXT_PATTERN = re.compile(r"^[A-Za-z\s]+$")

def is_valid_text(s: str) -> bool:
    """Allow only letters and spaces (blocks numbers and special characters)."""
    return bool(VALID_TEXT_PATTERN.fullmatch(s.strip()))

def extract_words(text: str) -> list:
    """Tokenize to alphabetic words only (case-insensitive)."""
    return re.findall(r"[A-Za-z]+", text)

def keyword_sentiment_vader(words: list) -> dict:
    """
    Classify words using VADER lexicon.
    Returns dict: {word: {"label": POS/NEG/NEU, "score": float}}
    """
    out = {}
    for w in words:
        lw = w.lower()
        score = vader_analyzer.lexicon.get(lw, 0.0)
        if score > 0:
            label = "POSITIVE"
        elif score < 0:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        out[lw] = {"label": label, "score": float(score)}
    return out

def keyword_sentiment_hf(words: list, hf_token: str) -> dict:
    """
    Classify words with Hugging Face Inference API (batch request).
    Tries models with neutral class first, then SST-2 as fallback.
    Returns dict: {word: {"label": POS/NEG/NEU, "score": float}} (missing if failed).
    """
    if not hf_token or not words:
        return {}

    models_to_try = [
        "cardiffnlp/twitter-roberta-base-sentiment",         # 3-class: NEG/NEU/POS
        "distilbert-base-uncased-finetuned-sst-2-english"    # 2-class: NEG/POS
    ]

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": words}  # batch

    for model in models_to_try:
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                if response.status_code == 503:
                    # model loading; try next
                    continue
                else:
                    continue

            data = response.json()
            # Normalize shape: expect list matching 'words', each is list of {label,score}
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # single input got expanded; normalize to list[list[...]]
                data = [data]
            if not (isinstance(data, list) and len(data) == len(words)):
                # Some endpoints may return fewer entries (e.g., dropped empty tokens). Skip in that case.
                continue

            out = {}
            for w, preds in zip(words, data):
                scores = {}
                for item in preds:
                    lbl = item["label"].upper()
                    if lbl in ["POSITIVE", "POS", "LABEL_2"]:
                        scores["POSITIVE"] = item["score"]
                    elif lbl in ["NEGATIVE", "NEG", "LABEL_0"]:
                        scores["NEGATIVE"] = item["score"]
                    elif lbl in ["NEUTRAL", "NEU", "LABEL_1"]:
                        scores["NEUTRAL"] = item["score"]
                if scores:
                    top = max(scores, key=scores.get)
                    out[w.lower()] = {"label": top, "score": float(scores[top])}
            return out
        except Exception:
            continue

    # If all models failed
    return {}

def build_keyword_df(text: str, use_vader: bool, use_hf: bool, hf_token: str) -> pd.DataFrame:
    """Create a per-word sentiment DataFrame combining VADER and HF."""
    words = sorted(set(w.lower() for w in extract_words(text)))
    if not words:
        return pd.DataFrame(columns=["word", "VADER_label", "VADER_score", "HF_label", "HF_score"])

    vader_map = keyword_sentiment_vader(words) if use_vader else {}
    hf_map = keyword_sentiment_hf(words, hf_token) if use_hf and hf_token else {}

    rows = []
    for w in words:
        v = vader_map.get(w, {"label": None, "score": None})
        h = hf_map.get(w, {"label": None, "score": None})
        rows.append({
            "word": w,
            "VADER_label": v["label"],
            "VADER_score": v["score"],
            "HF_label": h["label"],
            "HF_score": h["score"]
        })
    df = pd.DataFrame(rows)
    # Order by VADER score magnitude then HF score (desc)
    df = df.sort_values(by=["VADER_score", "HF_score"], ascending=[False, False], na_position="last")
    return df.reset_index(drop=True)

# ----------------------
# File Loader Functions
# ----------------------
def load_csv(file):
    """
    Safe CSV loader that handles encoding issues, empty files,
    and alternate delimiters.
    """
    try:
        return pd.read_csv(file, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.EmptyDataError):
        st.warning("UTF-8 failed, retrying with Latin-1 and alternate delimiters.")
        file.seek(0)
        for delim in [",", ";", "\t", "|"]:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin-1", delimiter=delim)
                if not df.empty:
                    return df
            except pd.errors.EmptyDataError:
                continue
        raise pd.errors.EmptyDataError("No data found in the uploaded CSV.")

def load_file(file):
    """
    Load different file formats and return a DataFrame with text data.
    Supports CSV, TXT, and PDF files.
    """
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        # Use existing CSV loader
        return load_csv(file), 'csv'
    
    elif file_extension == 'txt':
        # Handle TXT files
        try:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # Split text into lines for analysis
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            df = pd.DataFrame({
                'line_number': range(1, len(lines) + 1),
                'text': lines
            })
            return df, 'txt'
        except Exception as e:
            st.error(f"Failed to read TXT file: {e}")
            return pd.DataFrame(), 'txt'
    
    elif file_extension == 'pdf':
        # Handle PDF files
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Split page text into sentences
                    sentences = [s.strip() for s in page_text.split('.') if s.strip()]
                    for i, sentence in enumerate(sentences, 1):
                        if len(sentence) > 10:  # Only include substantial sentences
                            text_content.append({
                                'page': page_num,
                                'sentence': i,
                                'text': sentence
                            })
            
            df = pd.DataFrame(text_content)
            return df, 'pdf'
        except Exception as e:
            st.error(f"Failed to read PDF file: {e}")
            return pd.DataFrame(), 'pdf'
    
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return pd.DataFrame(), 'unknown'

# ----------------------
# Sentiment Functions
# ----------------------

def analyze_text(text, use_hf, use_vader, use_aws, hf_token, aws_key, aws_secret, aws_region):
    """
    Analyze sentiment using selected providers.
    Returns a list of result dicts (one per provider).
    """
    results = []

    # VADER
    if use_vader:
        try:
            scores = vader_analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "POSITIVE"
            elif compound <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            results.append({
                "provider": "VADER",
                "label": label,
                "score": compound,
                "probs": {
                    "Positive": scores.get("pos", None),
                    "Negative": scores.get("neg", None),
                    "Neutral": scores.get("neu", None),
                    "Mixed": None
                }
            })
        except Exception as e:
            results.append({"provider": "VADER", "error": str(e)})

    # Hugging Face
    if use_hf and hf_token:
        try:
            # Try multiple models in order of preference
            models_to_try = [
                "cardiffnlp/twitter-roberta-base-sentiment",
                "distilbert-base-uncased-finetuned-sst-2-english"
            ]
            
            success = False
            last_status = None
            last_text = None
            for model in models_to_try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {hf_token}"}
                payload = {"inputs": text}
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=15)
                last_status = response.status_code
                last_text = response.text[:200]
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        # Handle different model output formats
                        scores_dict = {}
                        for item in data[0]:
                            label = item["label"].upper()
                            # Normalize label names
                            if label in ["POSITIVE", "POS", "LABEL_2"]:
                                scores_dict["POSITIVE"] = item["score"]
                            elif label in ["NEGATIVE", "NEG", "LABEL_0"]:
                                scores_dict["NEGATIVE"] = item["score"]
                            elif label in ["NEUTRAL", "NEU", "LABEL_1"]:
                                scores_dict["NEUTRAL"] = item["score"]
                        
                        if scores_dict:
                            label = max(scores_dict, key=scores_dict.get)
                            results.append({
                                "provider": "HuggingFace",
                                "label": label,
                                "score": scores_dict[label],
                                "probs": {
                                    "Positive": scores_dict.get("POSITIVE", None),
                                    "Negative": scores_dict.get("NEGATIVE", None),
                                    "Neutral": scores_dict.get("NEUTRAL", None),
                                    "Mixed": None
                                }
                            })
                            success = True
                            break
                elif response.status_code == 503:
                    # Model loading, try next model
                    continue
                else:
                    # Try next model
                    continue
            
            if not success:
                results.append({
                    "provider": "HuggingFace", 
                    "error": f"All models failed. Last error: HTTP {last_status}: {last_text}"
                })
                
        except Exception as e:
            results.append({"provider": "HuggingFace", "error": str(e)})

    # AWS Comprehend
    if use_aws and aws_key and aws_secret:
        try:
            client = boto3.client(
                "comprehend",
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=aws_region
            )
            resp = client.detect_sentiment(Text=text, LanguageCode="en")
            label = resp.get("Sentiment", "")
            scores = resp.get("SentimentScore", {})
            results.append({
                "provider": "AWS",
                "label": label.upper(),
                "score": max(scores.values()) if scores else None,
                "probs": {
                    "Positive": scores.get("Positive", None),
                    "Negative": scores.get("Negative", None),
                    "Neutral": scores.get("Neutral", None),
                    "Mixed": scores.get("Mixed", None)
                }
            })
        except Exception as e:
            results.append({"provider": "AWS", "error": str(e)})

    return results

def results_to_dataframe(results):
    """
    Convert analysis results to a pandas DataFrame for display.
    """
    rows = []
    for r in results:
        if "error" in r:
            rows.append({
                "Provider": r.get("provider", "Unknown"),
                "Label": "ERROR",
                "Score": None,
                "Positive": None,
                "Negative": None,
                "Neutral": None,
                "Mixed": None,
                "Error": r["error"]
            })
        else:
            probs = r.get("probs", {})
            rows.append({
                "Provider": r.get("provider", "Unknown"),
                "Label": r.get("label", "Unknown"),
                "Score": round(r.get("score", 0), 4) if r.get("score") is not None else None,
                "Positive": round(probs.get("Positive", 0), 4) if probs.get("Positive") is not None else None,
                "Negative": round(probs.get("Negative", 0), 4) if probs.get("Negative") is not None else None,
                "Neutral": round(probs.get("Neutral", 0), 4) if probs.get("Neutral") is not None else None,
                "Mixed": round(probs.get("Mixed", 0), 4) if probs.get("Mixed") is not None else None,
                "Error": ""
            })
    
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI
# ----------------------
st.title("Sentiment Analysis Dashboard")
st.markdown("Analyze text sentiment using multiple providers")

tab_single, tab_batch = st.tabs(["Single Text Analysis", "Batch File Analysis"])

# ----------------------
# Tab: Single Text
# ----------------------
with tab_single:
    st.subheader("Analyze a single text")
    text = st.text_area(
        "Enter text to analyze (letters and spaces only):",
        value=default_text,
        height=150
    )
    colA, colB = st.columns([1, 2])

    with colA:
        run_btn = st.button("Analyze", type="primary")

    if run_btn:
        # ---- Validation (prevent numbers and special characters) ----
        if not text.strip():
            st.warning("Please enter some text.")
        elif not is_valid_text(text):
            st.error("Invalid input. Only letters and spaces are allowed (no numbers or special characters).")
        else:
            with st.spinner("Running analysis..."):
                results = analyze_text(text, use_hf, use_vader, use_aws, hf_token, aws_key, aws_secret, aws_region)
                df = results_to_dataframe(results)

            with colB:
                st.subheader("Results")
                st.dataframe(df, use_container_width=True)

                st.markdown("### Provider Insights")
                for r in results:
                    with st.container(border=True):
                        if "error" in r:
                            st.error(f"**{r.get('provider','')}**: {r['error']}")
                        else:
                            p = r["probs"]
                            st.markdown(f"**{r['provider']}**")
                            st.write(f"**Label:** {r['label']}  |  **Confidence:** {round(r['score'], 4)}")
                            pie_df = pd.DataFrame({
                                "Class": ["Positive", "Negative", "Neutral", "Mixed"],
                                "Probability": [p["Positive"], p["Negative"], p["Neutral"], p["Mixed"]]
                            })
                            pie_fig = px.pie(pie_df, names="Class", values="Probability",
                                             title=f"Distribution - {r['provider']}")
                            st.plotly_chart(pie_fig, use_container_width=True)

                st.markdown("### Provider Comparison (Bar Chart)")
                plot_df = df.melt(id_vars=["Provider", "Label"],
                                  value_vars=["Positive", "Negative", "Neutral", "Mixed"],
                                  var_name="Class", value_name="Probability").dropna()
                if not plot_df.empty:
                    fig = px.bar(plot_df, x="Provider", y="Probability", color="Class",
                                 barmode="group", text="Probability",
                                 title="Class Probabilities by Provider")
                    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig.update_layout(yaxis_range=[0, 1], xaxis_title="", yaxis_title="Probability")
                    st.plotly_chart(fig, use_container_width=True)

                # -------- Keyword Sentiment (VADER + HF) --------
                st.markdown("### Keyword Sentiment (per-word)")
                kw_df = build_keyword_df(text, use_vader, use_hf, hf_token)
                if not kw_df.empty:
                    st.dataframe(kw_df, use_container_width=True)

                    col_kw1, col_kw2 = st.columns(2)
                    with col_kw1:
                        st.markdown("**VADER word sets**")
                        st.write("Positive:", ", ".join(kw_df.loc[kw_df["VADER_label"] == "POSITIVE", "word"]))
                        st.write("Neutral:", ", ".join(kw_df.loc[kw_df["VADER_label"] == "NEUTRAL", "word"]))
                        st.write("Negative:", ", ".join(kw_df.loc[kw_df["VADER_label"] == "NEGATIVE", "word"]))
                    with col_kw2:
                        if use_hf and hf_token:
                            st.markdown("**Hugging Face word sets**")
                            st.write("Positive:", ", ".join(kw_df.loc[kw_df["HF_label"] == "POSITIVE", "word"]))
                            st.write("Neutral:", ", ".join(kw_df.loc[kw_df["HF_label"] == "NEUTRAL", "word"]))
                            st.write("Negative:", ", ".join(kw_df.loc[kw_df["HF_label"] == "NEGATIVE", "word"]))
                        else:
                            st.info("Enable Hugging Face with a valid token to see HF word sets.")

# ----------------------
# Tab: Batch File Analysis
# ----------------------
with tab_batch:
    st.subheader("Batch sentiment analysis")
    st.write("Upload a file (CSV, TXT, or PDF) for sentiment analysis. Results can be downloaded as CSV or JSON.")

    file = st.file_uploader("Upload file", type=["csv", "txt", "pdf"])
    if file:
        try:
            df_in, file_type = load_file(file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            df_in = pd.DataFrame()
            file_type = 'unknown'

        if not df_in.empty:
            st.write("Preview:")
            st.dataframe(df_in.head(), use_container_width=True)
            
            # Dynamic column selection based on file type
            if file_type == 'csv':
                text_col = st.selectbox("Select text column", options=list(df_in.columns))
            elif file_type == 'txt':
                text_col = 'text'
                st.info("Using 'text' column for analysis")
            elif file_type == 'pdf':
                text_col = 'text'
                st.info("Using 'text' column for analysis")
            else:
                st.error("Unknown file type")
                text_col = None

            if text_col and st.button("Run batch analysis", type="primary"):
                rows_out = []
                progress = st.progress(0)
                status = st.empty()
                total = len(df_in)

                for i, row in df_in.iterrows():
                    txt = str(row[text_col])
                    res = analyze_text(txt, use_hf, use_vader, use_aws, hf_token, aws_key, aws_secret, aws_region)
                    for r in res:
                        if "error" in r:
                            row_data = {
                                "index": i,
                                "provider": r.get("provider", ""),
                                "label": "ERROR",
                                "score": np.nan,
                                "pos": np.nan,
                                "neg": np.nan,
                                "neu": np.nan,
                                "mixed": np.nan,
                                "error": r["error"],
                                "original_text": txt[:100] + "..." if len(txt) > 100 else txt
                            }
                            # Add file-specific metadata
                            if file_type == 'pdf':
                                row_data["page"] = row.get('page', np.nan)
                                row_data["sentence"] = row.get('sentence', np.nan)
                            elif file_type == 'txt':
                                row_data["line_number"] = row.get('line_number', np.nan)
                            rows_out.append(row_data)
                        else:
                            p = r.get("probs", {})
                            row_data = {
                                "index": i,
                                "provider": r.get("provider", ""),
                                "label": r.get("label", ""),
                                "score": r.get("score", np.nan),
                                "pos": p.get("Positive", np.nan),
                                "neg": p.get("Negative", np.nan),
                                "neu": p.get("Neutral", np.nan),
                                "mixed": p.get("Mixed", np.nan),
                                "error": "",
                                "original_text": txt[:100] + "..." if len(txt) > 100 else txt
                            }
                            # Add file-specific metadata
                            if file_type == 'pdf':
                                row_data["page"] = row.get('page', np.nan)
                                row_data["sentence"] = row.get('sentence', np.nan)
                            elif file_type == 'txt':
                                row_data["line_number"] = row.get('line_number', np.nan)
                            rows_out.append(row_data)
                    
                    if total:
                        progress.progress(int((i + 1) / total * 100))
                        status.text(f"Processed {i + 1}/{total}")

                df_out = pd.DataFrame(rows_out)
                st.success("Batch analysis complete!")
                st.dataframe(df_out, use_container_width=True)

                # Analytics Section
                st.markdown("### Sentiment Distribution Across Dataset")
                dist_df = df_out[df_out["label"] != "ERROR"]["label"].value_counts().reset_index()
                dist_df.columns = ["Sentiment", "Count"]
                dist_fig = px.bar(dist_df, x="Sentiment", y="Count", color="Sentiment",
                                  text="Count", title="Overall Sentiment Distribution")
                dist_fig.update_traces(textposition="outside")
                st.plotly_chart(dist_fig, use_container_width=True)

                # Provider Comparison
                if len(df_out["provider"].unique()) > 1:
                    st.markdown("### Provider Comparison")
                    provider_comparison = df_out[df_out["label"] != "ERROR"].groupby(["provider", "label"]).size().reset_index(name="count")
                    provider_fig = px.bar(provider_comparison, x="provider", y="count", color="label",
                                        title="Sentiment Distribution by Provider", barmode="group")
                    st.plotly_chart(provider_fig, use_container_width=True)

                # Create pivot table for merged summary
                pivot = df_out.pivot_table(index="index",
                                           columns="provider",
                                           values="label",
                                           aggfunc="first")
                merged = df_in.join(pivot, how="left")
                st.markdown("### Merged Summary (one row per input)")
                st.dataframe(merged, use_container_width=True)

                # Download Options
                st.markdown("### Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV Download (Raw Results)
                    csv_all = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Raw Results (CSV)", 
                        data=csv_all,
                        file_name=f"sentiment_results_raw_{file_type}.csv", 
                        mime="text/csv"
                    )

                with col2:
                    # CSV Download (Merged Summary)
                    csv_merged = merged.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Summary (CSV)", 
                        data=csv_merged,
                        file_name=f"sentiment_results_summary_{file_type}.csv", 
                        mime="text/csv"
                    )

                with col3:
                    # JSON Download
                    json_data = {
                        "metadata": {
                            "file_name": file.name,
                            "file_type": file_type,
                            "total_texts_analyzed": total,
                            "analysis_providers": df_out["provider"].unique().tolist(),
                            "timestamp": pd.Timestamp.now().isoformat()
                        },
                        "summary_statistics": {
                            "sentiment_distribution": dist_df.to_dict('records'),
                            "total_errors": len(df_out[df_out["label"] == "ERROR"]),
                            "success_rate": f"{((total * len(df_out['provider'].unique()) - len(df_out[df_out['label'] == 'ERROR'])) / (total * len(df_out['provider'].unique())) * 100):.2f}%"
                        },
                        "detailed_results": df_out.to_dict('records'),
                        "merged_summary": merged.to_dict('records')
                    }
                    
                    json_str = json.dumps(json_data, indent=2, default=str)
                    st.download_button(
                        "Download Results (JSON)", 
                        data=json_str.encode("utf-8"),
                        file_name=f"sentiment_results_{file_type}.json", 
                        mime="application/json"
                    )

                # Display summary statistics
                st.markdown("### Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Texts", total)
                with col2:
                    st.metric("Providers Used", len(df_out["provider"].unique()))
                with col3:
                    st.metric("Successful Analyses", len(df_out[df_out["label"] != "ERROR"]))
                with col4:
                    st.metric("Errors", len(df_out[df_out["label"] == "ERROR"]))
