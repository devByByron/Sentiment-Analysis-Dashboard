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

# ----------------------
# Secrets & API Keys
# ----------------------

# Default values
hf_token = ""
aws_key = ""
aws_secret = ""
aws_region = "us-east-1"

# ✅ Load Hugging Face token from Streamlit secrets
hf_token = st.secrets.get("HF_TOKEN", "")

# ✅ Load AWS credentials from Streamlit secrets
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
# CSV Loader Function
# ----------------------
def load_csv(file):
    """
    Safe CSV loader that handles encoding issues, empty files,
    and alternate delimiters.
    """
    try:
        return pd.read_csv(file, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.EmptyDataError):
        st.warning("⚠️ UTF-8 failed, retrying with Latin-1 and alternate delimiters.")
        file.seek(0)
        for delim in [",", ";", "\t", "|"]:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin-1", delimiter=delim)
                if not df.empty:
                    return df
            except pd.errors.EmptyDataError:
                continue
        raise pd.errors.EmptyDataError("❌ No data found in the uploaded CSV.")

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
            api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {"inputs": text}
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    scores = {d["label"].upper(): d["score"] for d in data[0]}
                    label = max(scores, key=scores.get)
                    results.append({
                        "provider": "HuggingFace",
                        "label": label,
                        "score": scores[label],
                        "probs": {
                            "Positive": scores.get("POSITIVE", None),
                            "Negative": scores.get("NEGATIVE", None),
                            "Neutral": scores.get("NEUTRAL", None),
                            "Mixed": None
                        }
                    })
                else:
                    results.append({"provider": "HuggingFace", "error": "No result"})
            else:
                results.append({"provider": "HuggingFace", "error": f"HTTP {response.status_code}: {response.text}"})
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
        try:
            df_in = load_csv(file)
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            df_in = pd.DataFrame()

        if not df_in.empty:
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
