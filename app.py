"""
╔══════════════════════════════════════════════════════════════╗
║              STUDY BUDDY AI — Streamlit RAG App              ║
║        Upload PDFs · Ask Questions · Get Grounded Answers    ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
    streamlit run app.py

SETUP:
    1. Install packages:  pip install -r requirements.txt
    2. Create a .env file with:  GOOGLE_API_KEY=your_key_here
    3. Run the app and upload your study PDFs from the sidebar
"""

# ─────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain — document loaders, text splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFace sentence-transformer embeddings (local, no API key)
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS — lightning-fast local vector store
from langchain_community.vectorstores import FAISS

# Google Gemini LLM via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain RAG chain + prompt
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ─────────────────────────────────────────────────────────────
# SECTION 2: ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────
load_dotenv()  # Reads GOOGLE_API_KEY from your .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 3: PAGE CONFIGURATION (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# SECTION 4: CUSTOM CSS — WARM SCHOLARLY AESTHETIC
# Design direction: editorial / notebook-inspired
# Fonts: Playfair Display (headings) + DM Sans (body)
# Palette: warm cream, deep forest green, amber accents
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --cream:      #faf6f0;
        --parchment:  #f0e8d8;
        --green-dark: #1a3328;
        --green-mid:  #2d5a45;
        --green-light:#4a8c6a;
        --amber:      #d97706;
        --amber-soft: #fef3c7;
        --text-main:  #1c2b22;
        --text-muted: #5a7262;
        --border:     rgba(45, 90, 69, 0.18);
        --shadow:     rgba(26, 51, 40, 0.12);
    }

    .stApp {
        background-color: var(--cream);
        background-image:
            radial-gradient(ellipse at 10% 20%, rgba(74,140,106,0.06) 0%, transparent 60%),
            radial-gradient(ellipse at 90% 80%, rgba(217,119,6,0.05) 0%, transparent 60%);
        font-family: 'DM Sans', sans-serif;
        color: var(--text-main);
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; }

    .hero {
        padding: 2rem 0 1.25rem;
        border-bottom: 2px solid var(--border);
        margin-bottom: 1.75rem;
    }
    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--green-light);
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: var(--green-dark);
        line-height: 1.1;
        margin: 0;
    }
    .hero-title em {
        font-style: italic;
        color: var(--amber);
    }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
        font-weight: 300;
    }

    [data-testid="stSidebar"] {
        background-color: var(--green-dark);
        border-right: none;
    }
    [data-testid="stSidebar"] * { color: #c8d8cc !important; }

    .sidebar-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e8f0ea !important;
        padding: 1rem 0 0.25rem;
        border-bottom: 1px solid rgba(200,216,204,0.2);
        margin-bottom: 1.25rem;
    }
    .sidebar-section {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: rgba(200,216,204,0.5) !important;
        margin: 1rem 0 0.5rem;
    }
    .pdf-pill {
        background: rgba(200,216,204,0.08);
        border: 1px solid rgba(200,216,204,0.15);
        border-radius: 6px;
        padding: 0.45rem 0.75rem;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
        color: #b0c8b8 !important;
        font-family: 'DM Mono', monospace;
    }

    [data-testid="stFileUploader"] {
        background: rgba(200,216,204,0.06) !important;
        border: 1.5px dashed rgba(200,216,204,0.3) !important;
        border-radius: 10px !important;
    }

    .stTextInput > div > div > input {
        background: white !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.97rem !important;
        padding: 0.8rem 1.1rem !important;
        box-shadow: 0 2px 8px var(--shadow) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--green-light) !important;
        box-shadow: 0 0 0 3px rgba(74,140,106,0.12) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #9aab9e !important; }

    .stButton > button {
        background: var(--green-dark) !important;
        color: #e8f0ea !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.7rem 1.8rem !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(26,51,40,0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: var(--green-mid) !important;
        transform: translateY(-1px) !important;
    }

    .chat-role-user {
        text-align: right;
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: var(--amber);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .chat-bubble-user {
        background: var(--amber-soft);
        border: 1px solid rgba(217,119,6,0.2);
        border-radius: 12px 12px 4px 12px;
        padding: 0.85rem 1.1rem;
        margin: 0 0 0.25rem auto;
        max-width: 82%;
        font-size: 0.93rem;
        color: var(--text-main);
        box-shadow: 0 2px 6px rgba(217,119,6,0.1);
    }
    .chat-role-ai {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: var(--green-light);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .chat-bubble-ai {
        background: white;
        border: 1.5px solid var(--border);
        border-radius: 12px 12px 12px 4px;
        padding: 0.85rem 1.1rem;
        margin: 0.25rem auto 0.75rem 0;
        max-width: 88%;
        font-size: 0.93rem;
        color: var(--text-main);
        line-height: 1.7;
        box-shadow: 0 2px 6px var(--shadow);
    }

    .sources-header {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.5rem 0 0.75rem;
    }
    .source-card {
        background: var(--parchment);
        border: 1px solid rgba(45,90,69,0.12);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 0.83rem;
        color: var(--text-muted);
        line-height: 1.65;
    }
    .source-meta {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--green-light);
        margin-bottom: 0.4rem;
        letter-spacing: 0.05em;
    }

    .badge-ok {
        display: inline-block;
        background: rgba(74,140,106,0.12);
        border: 1px solid rgba(74,140,106,0.3);
        color: var(--green-mid);
        border-radius: 20px;
        padding: 0.18rem 0.65rem;
        font-size: 0.72rem;
        font-family: 'DM Mono', monospace;
    }
    .badge-warn {
        display: inline-block;
        background: rgba(217,119,6,0.1);
        border: 1px solid rgba(217,119,6,0.3);
        color: var(--amber);
        border-radius: 20px;
        padding: 0.18rem 0.65rem;
        font-size: 0.72rem;
        font-family: 'DM Mono', monospace;
    }

    .section-divider {
        font-family: 'DM Mono', monospace;
        font-size: 0.67rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.5rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .section-divider::before, .section-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    .empty-state {
        text-align: center;
        padding: 3.5rem 1rem;
        color: var(--text-muted);
    }
    .empty-icon { font-size: 3rem; margin-bottom: 1rem; }
    .empty-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        color: var(--green-dark);
        margin-bottom: 0.5rem;
    }
    .empty-body {
        font-size: 0.88rem;
        line-height: 1.7;
        max-width: 380px;
        margin: auto;
    }

    hr { border-color: var(--border); }
    .stSpinner > div { border-top-color: var(--green-light) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECTION 5: SESSION STATE INITIALIZATION
# Session state persists across re-runs of the Streamlit app.
# We use it to store chat history and the built RAG chain.
# ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of {question, answer} dicts

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None     # holds the built LangChain RetrievalQA chain

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []  # filenames of processed PDFs

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []    # source chunks from the most recent answer


# ─────────────────────────────────────────────────────────────
# SECTION 6: CORE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def load_pdfs(uploaded_files: list) -> list:
    """
    Takes Streamlit UploadedFile objects, writes each to a temp file
    (PyPDFLoader needs a real disk path), loads pages, tags them
    with the original filename, and returns all LangChain Documents.
    """
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            # Tag each page so we know which file it came from
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Could not read `{uploaded_file.name}`: {e}")
        finally:
            os.unlink(tmp_path)  # clean up temp file from disk
    return all_docs


def split_into_chunks(documents: list) -> list:
    """
    Splits documents into overlapping text chunks.

    chunk_size=900    — roughly 120–150 words per chunk
    chunk_overlap=120 — overlap prevents losing context at chunk edges
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(documents)


def build_faiss_index(chunks: list) -> FAISS:
    """
    Generates sentence-transformer embeddings and stores them in FAISS.

    Model: all-MiniLM-L6-v2
      - Runs fully locally (no API key, no cost)
      - Excellent at semantic similarity for question answering
      - 384-dimensional vectors

    normalize_embeddings=True ensures correct cosine similarity scoring.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def create_rag_chain(vector_store: FAISS, api_key: str) -> RetrievalQA:
    """
    Builds the full RAG pipeline:
      1. Retriever  — finds top-4 semantically relevant chunks from FAISS
      2. Prompt     — strictly instructs Gemini to stay grounded
      3. LLM        — Google Gemini 1.5 Flash (fast, accurate, cost-effective)
      4. Chain      — ties everything together; returns answer + source docs
    """

    # Retrieve the 4 most semantically similar chunks for each query
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Prompt engineering: the key to preventing hallucinations
    prompt_template = """You are Study Buddy AI, a friendly and precise academic assistant.
Your ONLY job is to help students understand material from their uploaded study documents.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using the context excerpts provided below. Do NOT use outside knowledge.
2. If the context does not contain enough information to answer, respond EXACTLY with:
   "The uploaded study material does not contain enough information to answer this question."
3. Be clear, concise, and student-friendly. Use simple language where possible.
4. If the answer involves steps, lists, or definitions, format them clearly.
5. Never fabricate facts, dates, formulas, names, or figures.
6. Where relevant, mention the source document name in your answer.

────────────────────────────────────────
STUDY MATERIAL CONTEXT:
{context}
────────────────────────────────────────

STUDENT QUESTION: {question}

ANSWER:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    # Gemini 1.5 Flash: temperature=0.1 keeps answers factual, not creative
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=1024,
    )

    # "stuff" chain type = all retrieved chunks stuffed into one prompt call
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # needed to display source excerpts
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


# ─────────────────────────────────────────────────────────────
# SECTION 7: HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">📚 AI-Powered Study Assistant</div>
    <div class="hero-title">Study Buddy <em>AI</em></div>
    <div class="hero-sub">Upload your notes, textbooks, or PDFs — and ask anything.</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECTION 8: SIDEBAR — FILE UPLOADER + STATUS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📖 Study Buddy AI</div>', unsafe_allow_html=True)

    # Show API key status
    if not GOOGLE_API_KEY:
        st.markdown('<span class="badge-warn">⚠ NO API KEY</span>', unsafe_allow_html=True)
        st.caption("Add `GOOGLE_API_KEY=...` to a `.env` file.")
    else:
        st.markdown('<span class="badge-ok">✓ Gemini Ready</span>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Upload Documents</div>', unsafe_allow_html=True)

    # Multi-file PDF uploader
    uploaded_files = st.file_uploader(
        label="Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Process button — triggers indexing pipeline
    process_clicked = st.button("📥 Process & Index PDFs")

    # Show names of currently indexed PDFs
    if st.session_state.uploaded_names:
        st.markdown('<div class="sidebar-section">Indexed Documents</div>', unsafe_allow_html=True)
        for name in st.session_state.uploaded_names:
            st.markdown(f'<div class="pdf-pill">📄 {name}</div>', unsafe_allow_html=True)

    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
        'color:rgba(200,216,204,0.4);line-height:1.8;">'
        'Embeddings: all-MiniLM-L6-v2<br>'
        'Vector DB: FAISS<br>'
        'LLM: Gemini 1.5 Flash<br>'
        'Framework: LangChain</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# SECTION 9: PROCESS UPLOADED PDFs
# Triggered when user clicks "Process & Index PDFs".
# Runs the full pipeline: load → split → embed → index → chain.
# Everything is stored in session_state so it survives re-runs.
# ─────────────────────────────────────────────────────────────
if process_clicked:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF before processing.")
    elif not GOOGLE_API_KEY:
        st.error("🔑 `GOOGLE_API_KEY` missing. Add it to your `.env` file.")
    else:
        with st.spinner("📖 Reading and indexing your study materials..."):
            try:
                # Step 1 — Load raw text from PDFs
                documents = load_pdfs(uploaded_files)

                if not documents:
                    st.error("❌ No text could be extracted from the uploaded PDFs.")
                    st.stop()

                # Step 2 — Split into overlapping chunks
                chunks = split_into_chunks(documents)

                # Step 3 — Build FAISS vector index
                vector_store = build_faiss_index(chunks)

                # Step 4 — Assemble the RAG chain
                st.session_state.rag_chain = create_rag_chain(vector_store, GOOGLE_API_KEY)

                # Save metadata
                st.session_state.uploaded_names = [f.name for f in uploaded_files]

                # Reset chat when new docs are loaded
                st.session_state.chat_history = []
                st.session_state.last_sources = []

                st.success(
                    f"✅ Indexed {len(chunks)} chunks from "
                    f"{len(uploaded_files)} document(s). Ready to answer questions!"
                )

            except Exception as e:
                st.error(f"❌ Failed to process PDFs: {e}")


# ─────────────────────────────────────────────────────────────
# SECTION 10: MAIN CHAT INTERFACE
# ─────────────────────────────────────────────────────────────

# If no documents have been indexed yet, show the onboarding screen
if st.session_state.rag_chain is None:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No study material loaded yet</div>
        <div class="empty-body">
            Upload your PDFs or textbooks using the sidebar on the left,
            then click <strong>Process & Index PDFs</strong> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Render past conversation ──
    if st.session_state.chat_history:
        st.markdown('<div class="section-divider">Conversation</div>', unsafe_allow_html=True)

        for exchange in st.session_state.chat_history:
            # User message (right-aligned amber bubble)
            st.markdown('<div class="chat-role-user">You</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-bubble-user">{exchange["question"]}</div>',
                unsafe_allow_html=True,
            )
            # AI response (left-aligned white bubble)
            st.markdown('<div class="chat-role-ai">Study Buddy AI</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-bubble-ai">{exchange["answer"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Question input ──
    st.markdown('<div class="section-divider">Ask a Question</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        user_question = st.text_input(
            label="question",
            placeholder="e.g. Explain Newton's second law. What is mitosis? Summarise Chapter 3.",
            label_visibility="collapsed",
            key="question_input",
        )
    with col2:
        ask_clicked = st.button("Ask →")

    # ── Handle query ──
    if ask_clicked and user_question.strip():
        with st.spinner("🔍 Searching your study material..."):
            try:
                result = st.session_state.rag_chain.invoke(
                    {"query": user_question.strip()}
                )
                answer = result.get("result", "No answer generated.")
                source_docs = result.get("source_documents", [])

                # Persist to chat history
                st.session_state.chat_history.append({
                    "question": user_question.strip(),
                    "answer": answer,
                })
                st.session_state.last_sources = source_docs

                # Rerun refreshes the rendered chat view
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")

    elif ask_clicked and not user_question.strip():
        st.warning("⚠️ Please type a question before clicking Ask.")

    # ── Show retrieved source excerpts for the latest answer ──
    if st.session_state.last_sources:
        st.markdown(
            '<div class="sources-header">📎 Retrieved Source Excerpts</div>',
            unsafe_allow_html=True,
        )
        for i, doc in enumerate(st.session_state.last_sources, start=1):
            source_name = doc.metadata.get("source", "Unknown Document")
            page_num    = doc.metadata.get("page", "N/A")

            # Trim very long chunks for cleaner display
            preview = doc.page_content.strip()
            if len(preview) > 450:
                preview = preview[:450] + "…"

            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-meta">◈ {source_name} — Page {page_num}  |  Excerpt {i}</div>'
                f'{preview}'
                f'</div>',
                unsafe_allow_html=True,
            )