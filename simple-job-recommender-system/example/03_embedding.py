import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
from uuid import uuid4
from tqdm import tqdm

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("QDRANT_URL:", QDRANT_URL)
print("QDRANT_API_KEY exists:", QDRANT_API_KEY is not None)

data_path = "./data/jobstreet_data_jakarta_data_scientist_cleaned.csv"
print("Loading dataset from:", data_path)

df = pd.read_csv(data_path)
print("Total rows loaded:", len(df))
print("Columns:", df.columns.tolist())

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print("Loading model:", model_name)

model = SentenceTransformer(model_name)

texts = df["description_cleaned"].fillna("").tolist()

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

vector_size = len(embeddings[0])
print("Vector dimension:", vector_size)
print("Sample embedding (first 5 values):", embeddings[0][:5])

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant")

collection_name = "jobstreet_jobs"

print("Recreating collection:", collection_name)

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)

collections = client.get_collections()
print("Available collections:", [c.name for c in collections.collections])

points = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    point_id = str(uuid4())
    vector = embeddings[idx].tolist()

    payload = {
        "role": row["role"],
        "company": row["company"],
        "description": row["description"],
        "link": row["link"]
    }

    points.append(
        PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
    )

    if idx < 3:
        print("Preview point:", point_id)
        print("Role:", row["role"])
        print("Vector length:", len(vector))

print("Total points prepared:", len(points))

operation_info = client.upsert(
    collection_name=collection_name,
    points=points
)

print("Upsert operation info:", operation_info)

collection_info = client.get_collection(collection_name)
print("Collection info:", collection_info)

count_info = client.count(collection_name=collection_name)
print("Total vectors stored in Qdrant:", count_info.count)

print("Upload finished.")