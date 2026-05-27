import os
import re
import fitz
import faiss
import pickle
import numpy as np
import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text)
    return text.split()

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------
# PATH
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
os.makedirs(VECTOR_DIR, exist_ok=True)

# --------------------------------
# MODEL
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

# --------------------------------
# SPLITTER
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", ":", ";", ",", " "]
)

# --------------------------------
# DATA FOLDERS
# --------------------------------

folders = {
    "mevzuat": "data/mevzuat",
    "takvim": "data/takvim"
}

# --------------------------------
# CLEAN FUNCTION
# --------------------------------

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_title(file):
    return re.sub(r"\.pdf$", "", file.lower())

# --------------------------------
# READ PDF
# --------------------------------

for category, path in folders.items():

    if not os.path.exists(path):
        print(f"Klasör yok: {path}")
        continue

    for file in os.listdir(path):

        if not file.endswith(".pdf"):
            continue

        full_path = os.path.join(path, file)

        print(f"İşleniyor: {file}")

        try:
            pdf = fitz.open(full_path)

            for page_num in range(len(pdf)):

                blocks = pdf[page_num].get_text("blocks")

                page_text = ""

                for b in blocks:
                    if len(b) > 4:
                        page_text += b[4] + "\n"

                page_text = clean_text(page_text)

                # -------------------------------
                # GÜÇLÜ GÜRÜLTÜ FİLTRESİ
                # -------------------------------
                if len(page_text.split()) < 20:
                    continue

                chunks = splitter.split_text(page_text)

                for chunk in chunks:

                    chunk = clean_text(chunk)

                    if len(chunk.split()) < 25:
                        continue

                    documents.append(chunk)

                    metadata.append({
                        "source": file,
                        "page": page_num + 1,
                        "type": category,
                        "title": clean_title(file)  # 🔥 KRİTİK
                    })

        except Exception as e:
            print(f"Hata: {file}")
            print(e)

# --------------------------------
# CHECK
# --------------------------------

print("Toplam chunk:", len(documents))

if len(documents) == 0:
    raise ValueError("Hiç veri yok! PDF extraction hatalı")

# --------------------------------
# EMBEDDING (GÜÇLENDİRİLDİ)
# --------------------------------

embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    batch_size=32,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# --------------------------------
# FAISS
# --------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# --------------------------------
# SAVE
# --------------------------------

faiss.write_index(
    index,
    os.path.join(VECTOR_DIR, "index.faiss")
)

with open(
    os.path.join(VECTOR_DIR, "data.pkl"),
    "wb"
) as f:
    pickle.dump((documents, metadata), f)

print("FAISS başarıyla oluşturuldu")
