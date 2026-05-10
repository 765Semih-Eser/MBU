import os
import fitz
import faiss
import pickle

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
os.makedirs("vectorstore", exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

folders = {
    "mevzuat": "data/mevzuat",
    "takvim": "data/takvim"
}

for category, path in folders.items():

    for file in os.listdir(path):

        if file.endswith(".pdf"):

            pdf = fitz.open(os.path.join(path, file))

            for page_num in range(len(pdf)):

                text = pdf[page_num].get_text()

                chunks = splitter.split_text(text)

                for chunk in chunks:

                    documents.append(chunk)

                    metadata.append({
                        "source": file,
                        "page": page_num + 1,
                        "type": category
                    })

print("Chunk sayısı:", len(documents))

# Embedding
embeddings = model.encode(documents)

# FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
os.makedirs(VECTOR_DIR, exist_ok=True)

faiss.write_index(
    index,
    os.path.join(VECTOR_DIR, "index.faiss")
)

with open(os.path.join(VECTOR_DIR, "data.pkl"), "wb") as f:
    pickle.dump((documents, metadata), f)
print("FAISS oluşturuldu.")