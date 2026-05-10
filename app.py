import streamlit as st
import faiss
import pickle
import numpy as np
import sqlite3
import json
import os
import uuid
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------
# BASE PATH
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "chat.db")

VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
DATA_PATH = os.path.join(VECTOR_DIR, "data.pkl")

# ---------------------------
# CHECK VECTORSTORE
# ---------------------------
if not os.path.exists(INDEX_PATH):
    st.error("index.faiss bulunamadı. Önce ingest.py çalıştır.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("data.pkl bulunamadı. Önce ingest.py çalıştır.")
    st.stop()

# ---------------------------
# MODEL
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# LOAD FAISS
# ---------------------------
index = faiss.read_index(INDEX_PATH)

with open(DATA_PATH, "rb") as f:
    documents, metadata = pickle.load(f)

# ---------------------------
# SQLITE
# ---------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    title TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    role TEXT,
    content TEXT,
    sources TEXT
)
""")

conn.commit()

# ---------------------------
# FUNCTIONS
# ---------------------------
def save_message(chat_id, role, content, sources=None):

    cursor.execute("""
    INSERT INTO messages
    (chat_id, role, content, sources)
    VALUES (?, ?, ?, ?)
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

    messages = []

    for r in rows:

        messages.append({
            "role": r[0],
            "content": r[1],
            "sources": json.loads(r[2]) if r[2] else None
        })

    return messages


def create_chat(chat_id, title):

    cursor.execute("""
    INSERT OR IGNORE INTO chats
    VALUES (?, ?)
    """, (chat_id, title))

    conn.commit()


def get_chats():

    cursor.execute("""
    SELECT chat_id, title
    FROM chats
    """)

    return cursor.fetchall()


# ---------------------------
# INITIAL CHAT
# ---------------------------
db_chats = get_chats()

if len(db_chats) == 0:

    first_id = str(uuid.uuid4())

    create_chat(first_id, "Yeni Sohbet")

    db_chats = [(first_id, "Yeni Sohbet")]

# ---------------------------
# SESSION
# ---------------------------
if "active_chat" not in st.session_state:

    st.session_state.active_chat = db_chats[0][0]

chat_ids = [c[0] for c in db_chats]

if st.session_state.active_chat not in chat_ids:

    st.session_state.active_chat = db_chats[0][0]

chat_id = st.session_state.active_chat

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("Sohbetler")

# Yeni sohbet
if st.sidebar.button("➕ Yeni Sohbet"):

    new_id = str(uuid.uuid4())

    create_chat(new_id, "Yeni Sohbet")

    st.session_state.active_chat = new_id

    st.rerun()

# Boş sohbet temizle
if st.sidebar.button("🗑 Boş Sohbetleri Temizle"):

    cursor.execute("""
    DELETE FROM chats
    WHERE title='Yeni Sohbet'
    """)

    conn.commit()

    st.rerun()

# Güncel chatler
db_chats = get_chats()

# Sohbet listesi
for cid, title in db_chats:

    col1, col2 = st.sidebar.columns([4, 1])

    # sohbet aç
    with col1:

        if st.button(title, key=f"chat_{cid}"):

            st.session_state.active_chat = cid

            st.rerun()

    # sohbet sil
    with col2:

        if st.button("❌", key=f"del_{cid}"):

            cursor.execute(
                "DELETE FROM messages WHERE chat_id=?",
                (cid,)
            )

            cursor.execute(
                "DELETE FROM chats WHERE chat_id=?",
                (cid,)
            )

            conn.commit()

            remaining = get_chats()

            if remaining:

                st.session_state.active_chat = remaining[0][0]

            else:

                new_id = str(uuid.uuid4())

                create_chat(new_id, "Yeni Sohbet")

                st.session_state.active_chat = new_id

            st.rerun()

# ---------------------------
# MAIN UI
# ---------------------------
st.title("🎓 Üniversite RAG Asistanı")

messages = load_messages(chat_id)

# mesajlar
for msg in messages:

    if msg["role"] == "user":

        st.markdown(f"### 👤 Sen")
        st.write(msg["content"])

    else:

        st.markdown(f"### 🤖 Asistan")
        st.write(msg["content"])

        if msg["sources"]:

            with st.expander("📌 Kaynaklar"):

                for s in msg["sources"]:

                    st.write(
                        f"""
                        • {s['source']}
                        | Sayfa: {s['page']}
                        | Tür: {s['type']}
                        """
                    )

# ---------------------------
# INPUT
# ---------------------------
with st.form("chat_form", clear_on_submit=True):

    query = st.text_input("Sorunu yaz")

    submit = st.form_submit_button("Gönder")

# ---------------------------
# RAG
# ---------------------------
if submit and query:

    # chat title
    cursor.execute(
        "SELECT title FROM chats WHERE chat_id=?",
        (chat_id,)
    )

    current_title = cursor.fetchone()[0]

    if current_title == "Yeni Sohbet":

        new_title = query[:35]

        cursor.execute(
            "UPDATE chats SET title=? WHERE chat_id=?",
            (new_title, chat_id)
        )

        conn.commit()

    # embedding
    q_vec = model.encode([query])

    D, I = index.search(np.array(q_vec), k=5)

    context = ""

    sources = []

    for idx in I[0]:

        context += documents[idx] + "\n\n"

        sources.append(metadata[idx])

    # prompt
    prompt = f"""
SADECE TÜRKÇE cevap ver.

Sadece verilen üniversite dokümanlarını kullan.

Eğer cevap dokümanlarda yoksa:
"Bu bilgi dokümanlarda bulunamadı."
de.

BAĞLAM:
{context}

SORU:
{query}
"""

    # ollama
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    # save
    save_message(chat_id, "user", query)

    save_message(
        chat_id,
        "assistant",
        answer,
        sources
    )

    st.rerun()