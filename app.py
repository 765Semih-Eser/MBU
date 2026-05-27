import streamlit as st
import faiss
import pickle
import numpy as np
import sqlite3
import json
import os
import uuid
import re
import ollama
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---------------------------
# TOKENIZER
# ---------------------------

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text)
    return text.split()


# ---------------------------
# PATHS
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chat.db")

VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
DATA_PATH = os.path.join(VECTOR_DIR, "data.pkl")


# ---------------------------
# CHECK FILES
# ---------------------------

if not os.path.exists(INDEX_PATH):
    st.error("index.faiss bulunamadı.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("data.pkl bulunamadı.")
    st.stop()


# ---------------------------
# MODEL
# ---------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# LOAD VECTOR STORE
# ---------------------------

index = faiss.read_index(INDEX_PATH)

with open(DATA_PATH, "rb") as f:
    documents, metadata = pickle.load(f)


# ---------------------------
# BM25
# ---------------------------

tokenized_docs = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


# ---------------------------
# SQLITE
# ---------------------------

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats(
chat_id TEXT PRIMARY KEY,
title TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages(
id INTEGER PRIMARY KEY AUTOINCREMENT,
chat_id TEXT,
role TEXT,
content TEXT,
sources TEXT
)
""")

conn.commit()


# ---------------------------
# DB FUNCTIONS
# ---------------------------

def save_message(chat_id, role, content, sources=None):
    cursor.execute("""
    INSERT INTO messages(chat_id, role, content, sources)
    VALUES(?,?,?,?)
    """, (
        chat_id,
        role,
        content,
        json.dumps(sources) if sources else None
    ))
    conn.commit()


def load_messages(chat_id):
    cursor.execute("""
    SELECT role, content, sources
    FROM messages
    WHERE chat_id=?
    ORDER BY id
    """, (chat_id,))

    rows = cursor.fetchall()

    out = []
    for r in rows:
        out.append({
            "role": r[0],
            "content": r[1],
            "sources": json.loads(r[2]) if r[2] else None
        })
    return out


def create_chat(cid, title):
    cursor.execute("INSERT OR IGNORE INTO chats VALUES(?,?)", (cid, title))
    conn.commit()


def get_chats():
    cursor.execute("SELECT chat_id, title FROM chats")
    return cursor.fetchall()


# ---------------------------
# INIT CHAT
# ---------------------------

db = get_chats()

if len(db) == 0:
    cid = str(uuid.uuid4())
    create_chat(cid, "Yeni Sohbet")
    db = get_chats()

if "active_chat" not in st.session_state:
    st.session_state.active_chat = db[0][0]

chat_id = st.session_state.active_chat


# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.title("Sohbetler")

if st.sidebar.button("➕ Yeni Sohbet"):
    cid = str(uuid.uuid4())
    create_chat(cid, "Yeni Sohbet")
    st.session_state.active_chat = cid
    st.rerun()

for cid, title in get_chats():
    c1, c2 = st.sidebar.columns([5, 1])

    with c1:
        if st.button(title, key=cid):
            st.session_state.active_chat = cid
            st.rerun()

    with c2:
        if st.button("❌", key=f"del{cid}"):
            cursor.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
            cursor.execute("DELETE FROM chats WHERE chat_id=?", (cid,))
            conn.commit()
            st.rerun()


# ---------------------------
# CHAT UI
# ---------------------------

st.title("🎓 Üniversite RAG Asistanı")

messages = load_messages(chat_id)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["sources"]:
            with st.expander("Kaynaklar"):
                for s in msg["sources"]:
                    st.write(f"{s['source']} | Sayfa:{s['page']} | Tür:{s['type']}")


# ---------------------------
# INPUT
# ---------------------------

query = st.chat_input("Sorunuzu yaz")


# ---------------------------
# RAG PIPELINE
# ---------------------------

if query and query.strip():

    query = query.strip()

    # chat title update
    cursor.execute("SELECT title FROM chats WHERE chat_id=?", (chat_id,))
    row = cursor.fetchone()

    if row and row[0] == "Yeni Sohbet":
        cursor.execute(
            "UPDATE chats SET title=? WHERE chat_id=?",
            (query[:40], chat_id)
        )
        conn.commit()

    save_message(chat_id, "user", query)


    # ---------------------------
    # FILTER
    # ---------------------------

    query_lower = query.lower()
    filter_type = None

    for w in ["vize", "final", "büt", "sınav", "tarih", "ne zaman"]:
        if w in query_lower:
            filter_type = "takvim"
            break


    # ---------------------------
    # RETRIEVAL
    # ---------------------------

    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    D, I = index.search(q_emb, 15)

    bm25_scores = bm25.get_scores(tokenize(query))

    combined = []

    for score, idx in zip(D[0], I[0]):

        if idx == -1 or idx >= len(metadata):
            continue

        meta = metadata[idx]

        if filter_type and meta.get("type") != filter_type:
            continue

        semantic = 1 / (1 + float(score))

        keyword = float(bm25_scores[idx])
        keyword = keyword / (keyword + 1e-6)

        final_score = 0.75 * semantic + 0.25 * keyword

        combined.append((idx, final_score))


    # ---------------------------
    # SORT
    # ---------------------------

    combined = sorted(combined, key=lambda x: x[1], reverse=True)


    # ---------------------------
    # DIVERSITY FILTER
    # ---------------------------

    q_tokens = set(tokenize(query))

    filtered = []

    for idx, sc in combined:

        doc_tokens = set(tokenize(documents[idx]))

        if len(q_tokens.intersection(doc_tokens)) >= 2:
            filtered.append((idx, sc))

    combined = filtered if filtered else combined

    combined = [x for x in combined if x[1] > 0.25]


    # ---------------------------
    # RESULTS
    # ---------------------------

    results = []
    seen = set()

    for idx, sc in combined:

        if idx in seen:
            continue

        seen.add(idx)
        results.append((documents[idx], metadata[idx]))

    # fallback
    if len(results) == 0:
        for _, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(metadata):
                results.append((documents[idx], metadata[idx]))

    results = results[:3]


    # ---------------------------
    # CONTEXT BUILD
    # ---------------------------

    context = ""
    sources = []
    used = set()

    for doc, meta in results:

        src = meta.get("source", "")

        if src in used:
            continue

        used.add(src)

        context += f"\nBelge: {src}\n{doc}\n\n"
        sources.append(meta)


    # ---------------------------
    # PROMPT
    # ---------------------------

    prompt = f"""
Sen üniversite belge asistanısın.

Kurallar:
- Sadece verilen metni kullan
- Uydurma yapma
- Kaynak dışına çıkma
- Yoksa: "Bu bilgi belgelerde bulunamadı"

Metin:
{context}

Soru:
{query}

Cevap:
"""


    # ---------------------------
    # LLM
    # ---------------------------

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]


    # ---------------------------
    # CLEAN OUTPUT
    # ---------------------------

    lines = []
    seen_lines = set()

    for line in answer.split("\n"):
        line = line.strip()
        if len(line) < 3:
            continue
        if line in seen_lines:
            continue
        seen_lines.add(line)
        lines.append(line)

    answer = "\n".join(lines)


    # ---------------------------
    # SAVE + RERUN
    # ---------------------------

    save_message(chat_id, "assistant", answer, sources)

    st.rerun()
