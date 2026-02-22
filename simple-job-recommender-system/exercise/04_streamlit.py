import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Deteksi otomatis: Kalau di Cloud pakai st.secrets, kalau di laptop pakai os.getenv
if "QDRANT_URL" in st.secrets:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
else:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# -----------------------

st.set_page_config(page_title="Data Scientist Job Recommender in Jakarta", layout="wide")

st.title("Data Scientist Job Recommender in Jakarta")

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource
def load_model():
    return SentenceTransformer(model_name)

@st.cache_resource
def load_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

model = load_model()
client = load_qdrant()

collection_name = "jobstreet_jobs"

query = st.text_area("Describe the job you are looking for", height=150)

top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Find Jobs") and query.strip() != "":
    with st.spinner("Searching..."):
        query_vector = model.encode(query).tolist()

        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points

    st.subheader("Recommended Jobs")

    for idx, hit in enumerate(search_result, start=1):
        payload = hit.payload

        st.markdown(f"### {idx}. {payload.get('role','')}")
        st.markdown(f"**Company:** {payload.get('company','')}")
        st.markdown(f"**Similarity Score:** {round(hit.score,4)}")
        st.markdown(payload.get("description","")[:500] + "...")
        st.markdown(f"[Apply Here]({payload.get('link','')})")
        st.markdown("---")