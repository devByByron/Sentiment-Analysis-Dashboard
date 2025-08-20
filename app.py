import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import boto3

# ----------------------
# Initialization
# ----------------------

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Default example text
default_text = "I love this product! It's amazing and works perfectly."

# API Keys (fallback defaults)
hf_token = ""
aws_key = ""
aws_secret = ""
aws_region = "us-east-1"

# ✅ Load secrets from .streamlit/secrets.toml (if available)
if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]

if "AWS_ACCESS_KEY" in st.secrets and "AWS_SECRET_KEY" in st.secrets:
    aws_key = st.secrets["AWS_ACCESS_KEY"]
    aws_secret = st.secrets["AWS_SECRET_KEY"]
    aws_region = st.secrets.get("AWS_REGION", aws_region)

# Debug check (won’t leak actual secrets)
st.sidebar.write("🔒 Secrets loaded:", {
    "HF_TOKEN": "HF_TOKEN" in st.secrets,
    "AWS_ACCESS_KEY": "AWS_ACCESS_KEY" in st.secrets
})

# ----------------------
# Sidebar configuration
# ----------------------
st.sidebar.header("Configuration")
use_vader = st.sidebar.checkbox("Use VADER", value=True, help="Free offline sentiment analysis")
use_hf = st.sidebar.checkbox("Use Hugging Face", value=False, help="Requires API token")
use_aws = st.sidebar.checkbox("Use AWS Comprehend", value=False, help="Requires AWS credentials")

# ✅ Fallback to sidebar inputs if secrets are missing
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
# Sentiment Functions
# ----------------------

def analyze_text_hf(text, hf_token=""):
    """Analyze text using Hugging Face API"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
        headers = {"Content-Type": "application/json"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        response = requests.post(API_URL, headers=headers, json={"inputs": text})

        if response.status_code == 200:
            results = response.json()
            if isinstance(results, list) and len(results) > 0:
                scores = results[0] if isinstance(results[0], list) else results
                probs = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
                for item in scores:
                    label = item['label'].upper()
                    score = item['score']
                    if 'POSITIVE' in label or label == 'LABEL_2':
                        probs["Positive"] = score
                    elif 'NEGATIVE' in label or label == 'LABEL_0':
                        probs["Negative"] = score
                    elif 'NEUTRAL' in label or label == 'LABEL_1':
                        probs["Neutral"] = score
                max_label = max(probs, key=probs.get)
                max_score = probs[max_label]
                return {"provider": "Hugging Face", "label": max_label, "score": max_score, "probs": probs}

        elif response.status_code == 503:
            return {"provider": "Hugging Face", "error": "Model is loading, try again in a few seconds"}
        elif response.status_code == 402:
            return {"provider": "Hugging Face", "error": "Rate limit exceeded or payment required"}
        elif response.status_code == 404:
            return {"provider": "Hugging Face", "error": "Model not found"}
        else:
            return {"provider": "Hugging Face", "error": f"API Error: {response.status_code}"}

    except requests.exceptions.RequestException as e:
        return {"provider": "Hugging Face", "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"provider": "Hugging Face", "error": f"Unexpected error: {str(e)}"}


def analyze_text_vader(text):
    """Analyze text using VADER sentiment"""
    try:
        scores = vader_analyzer.polarity_scores(text)
        probs = {
            "Positive": scores['pos'],
            "Negative": scores['neg'],
            "Neutral": scores['neu'],
            "Mixed": 0
        }
        compound = scores['compound']
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return {"provider": "VADER", "label": label, "score": abs(compound), "probs": probs}
    except Exception as e:
        return {"provider": "VADER", "error": str(e)}


def analyze_text_aws(text, aws_key, aws_secret, aws_region):
    """Analyze text using AWS Comprehend"""
    try:
        client = boto3.client(
            "comprehend",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region,
        )
        response = client.detect_sentiment(Text=text, LanguageCode="en")
        sentiment = response["Sentiment"]
        scores = response["SentimentScore"]

        probs = {
            "Positive": scores["Positive"],
            "Negative": scores["Negative"],
            "Neutral": scores["Neutral"],
            "Mixed": scores["Mixed"],
        }
        return {"provider": "AWS Comprehend", "label": sentiment, "score": max(probs.values()), "probs": probs}
    except Exception as e:
        return {"provider": "AWS Comprehend", "error": str(e)}


def analyze_text(text, use_hf=True, use_vader=True, use_aws=False, hf_token="", aws_key="", aws_secret="", aws_region=""):
    """Main function to analyze text using multiple providers"""
    results = []
    if use_vader:
        results.append(analyze_text_vader(text))
    if use_hf:
        results.append(analyze_text_hf(text, hf_token))
    if use_aws and aws_key and aws_secret:
        results.append(analyze_text_aws(text, aws_key, aws_secret, aws_region))
    return results


def results_to_dataframe(results):
    """Convert results to DataFrame"""
    rows = []
    for r in results:
        if "error" in r:
            rows.append({
                "Provider": r.get("provider", ""),
                "Label": "ERROR",
                "Score": np.nan,
                "Positive": np.nan,
                "Negative": np.nan,
                "Neutral": np.nan,
                "Mixed": np.nan
            })
        else:
            p = r.get("probs", {})
            rows.append({
                "Provider": r.get("provider", ""),
                "Label": r.get("label", ""),
                "Score": r.get("score", np.nan),
                "Positive": p.get("Positive", np.nan),
                "Negative": p.get("Negative", np.nan),
                "Neutral": p.get("Neutral", np.nan),
                "Mixed": p.get("Mixed", np.nan)
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI
# ----------------------
st.title("Sentiment Analysis Dashboard")
st.markdown("Analyze text sentiment using multiple providers")

tab_single, tab_batch = st.tabs(["Single Text Analysis", "Batch CSV Analysis"])

# ----------------------
# Tab: Single Text
# ----------------------
with tab_single:
    st.subheader("Analyze a single text")
    text = st.text_area("Enter text to analyze:", value=default_text, height=150)
    colA, colB = st.columns([1, 2])

    with colA:
        run_btn = st.button("Analyze", type="primary")
        st.write("")

        if use_aws:
            st.info("AWS Comprehend is enabled. Charges may apply after free tier.", icon="⚠️")
        else:
            st.success("Free-only mode (Hugging Face + VADER).", icon="✅")

    if run_btn and text.strip():
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

# ----------------------
# Tab: Batch CSV
# ----------------------
with tab_batch:
    st.subheader("Batch sentiment on CSV")
    st.write("Upload a CSV and choose a text column. Results can be downloaded as CSV.")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df_in = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(df_in.head(), use_container_width=True)
        text_col = st.selectbox("Select text column", options=list(df_in.columns))

        if st.button("Run batch analysis", type="primary"):
            rows_out = []
            progress = st.progress(0)
            status = st.empty()
            total = len(df_in)

            for i, row in df_in.iterrows():
                txt = str(row[text_col])
                res = analyze_text(txt, use_hf, use_vader, use_aws, hf_token, aws_key, aws_secret, aws_region)
                for r in res:
                    if "error" in r:
                        rows_out.append({
                            "index": i,
                            "provider": r.get("provider", ""),
                            "label": "ERROR",
                            "score": np.nan,
                            "pos": np.nan,
                            "neg": np.nan,
                            "neu": np.nan,
                            "mixed": np.nan,
                            "error": r["error"]
                        })
                    else:
                        p = r.get("probs", {})
                        rows_out.append({
                            "index": i,
                            "provider": r.get("provider", ""),
                            "label": r.get("label", ""),
                            "score": r.get("score", np.nan),
                            "pos": p.get("Positive", np.nan),
                            "neg": p.get("Negative", np.nan),
                            "neu": p.get("Neutral", np.nan),
                            "mixed": p.get("Mixed", np.nan),
                            "error": ""
                        })
                if total:
                    progress.progress(int((i + 1) / total * 100))
                    status.text(f"Processed {i + 1}/{total}")

            df_out = pd.DataFrame(rows_out)
            st.success("Batch complete.")
            st.dataframe(df_out, use_container_width=True)

            st.markdown("### Sentiment Distribution Across Dataset")
            dist_df = df_out[df_out["label"] != "ERROR"]["label"].value_counts().reset_index()
            dist_df.columns = ["Sentiment", "Count"]
            dist_fig = px.bar(dist_df, x="Sentiment", y="Count", color="Sentiment",
                              text="Count", title="Overall Sentiment Distribution")
            dist_fig.update_traces(textposition="outside")
            st.plotly_chart(dist_fig, use_container_width=True)

            pivot = df_out.pivot_table(index="index",
                                       columns="provider",
                                       values="label",
                                       aggfunc="first")
            merged = df_in.join(pivot, how="left")
            st.markdown("### Merged Summary (one row per input)")
            st.dataframe(merged, use_container_width=True)

            csv_all = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download raw provider rows (CSV)", data=csv_all,
                               file_name="sentiment_results_raw.csv", mime="text/csv")

            csv_merged = merged.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download merged summary (CSV)", data=csv_merged,
                               file_name="sentiment_results_merged.csv", mime="text/csv")
