"""RAGenius AI — Intelligent Document Chatbot.

A ChatGPT-style Streamlit app that answers questions grounded in user-uploaded
PDF/CSV/TXT/image documents, using Gemini for generation and FAISS for
retrieval. Includes OCR for scanned PDFs and images, document comparison,
voice input/output, multi-session chat history with search, and more.
"""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from backend.chat_memory import ChatSessionStore
from backend.embeddings import EmbeddingEngine
from backend.loader import load_file
from backend.prompts import (
    build_comparison_prompt,
    build_faq_prompt,
    build_followup_prompt,
    build_key_insights_prompt,
    build_keywords_prompt,
    build_summary_prompt,
)
from backend.rag_chain import answer_question, generate_text, get_client
from backend.retriever import format_citation, highlight_terms
from backend.splitter import split_documents
from backend.utils import (
    estimated_reading_time_minutes,
    file_hash,
    human_timestamp,
    is_image_file,
    is_supported_file,
    search_snippet,
    truncate,
    validate_file_size,
    word_count,
)
from backend.vector_store import VectorStore
from backend.voice import TranscriptionError, transcribe_audio

load_dotenv()

VECTOR_DB_DIR = "vector_db"
APP_TITLE = "RAGenius AI"
APP_TAGLINE = "Chat with your documents — grounded, cited, instant."

# ----------------------------------------------------------------------------
# Page + session setup
# ----------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide", initial_sidebar_state="expanded")


def load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "store": VectorStore(),
        "engine": None,
        "client": None,
        "sessions": None,  # ChatSessionStore, built below
        "uploaded_meta": {},  # filename -> {status, chunks, hash, doc_type, raw_text, word_count, ocr_used}
        "theme": "light",
        "settings": {
            "temperature": 0.4,
            "max_output_tokens": 1024,
            "top_k": 4,
            "chunk_size": 800,
            "chunk_overlap": 150,
        },
        "search_mode": "semantic",
        "pending_followups": [],
        "last_retrieved": [],
        "last_query": "",
        "onboarding_dismissed": False,
        "chat_search_query": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.sessions is None:
        st.session_state.sessions = ChatSessionStore()
    if st.session_state.client is None:
        st.session_state.client = get_client()
    if st.session_state.engine is None:
        st.session_state.engine = EmbeddingEngine(client=st.session_state.client)

    if "restored" not in st.session_state:
        st.session_state.store.load(VECTOR_DB_DIR)
        st.session_state.restored = True


init_state()
load_css()

# ----------------------------------------------------------------------------
# Onboarding (first run only)
# ----------------------------------------------------------------------------


@st.dialog("👋 Welcome to RAGenius AI")
def show_onboarding() -> None:
    st.markdown(
        """
Chat with your own **PDF, CSV, TXT, and image** files — every answer is
grounded in your documents and cited, so you always know where it came from.

**Quick start**
1. 📤 Upload files from the sidebar (drag & drop supported)
2. 💬 Ask a question in the Chat tab — watch it stream in with citations
3. ✨ Open **Premium Insights** for summaries, keywords, and auto-FAQs
4. 🆚 Try **Compare Documents** to see two files side by side
5. 🎙️ Use the mic to ask questions by voice

**Shortcuts**: press `/` to jump to the chat box.
        """
    )
    if st.button("Let's go →", use_container_width=True):
        st.session_state.onboarding_dismissed = True
        st.rerun()


if not st.session_state.onboarding_dismissed and not st.session_state.uploaded_meta:
    show_onboarding()

# Keyboard shortcut: "/" focuses the chat input.
st.markdown(
    """
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.key === '/' && document.activeElement.tagName !== 'TEXTAREA' && document.activeElement.tagName !== 'INPUT') {
            const box = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (box) { box.focus(); e.preventDefault(); }
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — document management + settings + recent conversations
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f'<div class="rg-title">🧠 {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rg-subtitle">{APP_TAGLINE}</div>', unsafe_allow_html=True)
    st.divider()

    theme_choice = st.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
    st.session_state.theme = "dark" if theme_choice else "light"
    st.markdown(
        f'<script>document.documentElement.setAttribute("data-theme", "{st.session_state.theme}")</script>',
        unsafe_allow_html=True,
    )

    if st.session_state.client is None:
        st.warning("⚠️ `GEMINI_API_KEY` not set. Add it to your `.env` file to enable AI answers.")
    else:
        st.success(f"✅ Gemini connected · embeddings via **{st.session_state.engine.backend}**")

    st.subheader("📤 Upload Files")
    uploads = st.file_uploader(
        "Drag and drop PDF, CSV, TXT, or image files",
        type=["pdf", "csv", "txt", "jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Images and scanned PDFs are automatically OCR'd.",
    )

    if uploads:
        progress = st.progress(0, text="Processing files…")
        for i, upload in enumerate(uploads):
            filename = upload.name
            data = upload.read()

            if not is_supported_file(filename):
                st.error(f"❌ Unsupported file type: {filename}")
                continue

            ok, msg = validate_file_size(len(data))
            if not ok:
                st.error(f"❌ {filename}: {msg}")
                continue

            h = file_hash(data)
            if h in st.session_state.store.file_hashes:
                st.info(f"⏭️ Skipped duplicate: {filename}")
                continue

            try:
                st.session_state.uploaded_meta[filename] = {"status": "processing"}
                docs = load_file(data, filename)
                chunks = split_documents(
                    docs,
                    chunk_size=st.session_state.settings["chunk_size"],
                    chunk_overlap=st.session_state.settings["chunk_overlap"],
                )
                texts = [c.page_content for c in chunks]
                vectors = st.session_state.engine.embed_texts(texts)
                st.session_state.store.add(vectors, chunks)
                st.session_state.store.file_hashes.add(h)

                raw_text = "\n".join(d.page_content for d in docs)
                ocr_used = any(d.metadata.get("extraction") == "ocr" for d in docs) or is_image_file(filename)
                st.session_state.uploaded_meta[filename] = {
                    "status": "ready",
                    "chunks": len(chunks),
                    "hash": h,
                    "doc_type": chunks[0].metadata.get("type") if chunks else "unknown",
                    "raw_text": raw_text,
                    "word_count": word_count(raw_text),
                    "reading_time": estimated_reading_time_minutes(raw_text),
                    "ocr_used": ocr_used,
                }
                st.session_state.store.save(VECTOR_DB_DIR)
            except Exception as exc:  # noqa: BLE001
                st.session_state.uploaded_meta[filename] = {"status": "error", "error": str(exc)}
                st.error(f"❌ {filename}: {exc}")

            progress.progress((i + 1) / len(uploads), text=f"Processed {filename}")
        progress.empty()
        st.rerun()

    st.subheader("📁 Uploaded Documents")


    st.divider()

    if st.button("🗑️ Reset All Documents", use_container_width=True):
        st.session_state.store = VectorStore()
        st.session_state.uploaded_meta = {}
        st.session_state.store.save(VECTOR_DB_DIR)
        st.rerun()


    if not st.session_state.uploaded_meta:
        st.caption("No documents uploaded yet.")
    else:
        for filename, meta in list(st.session_state.uploaded_meta.items()):
            status = meta.get("status", "unknown")
            badge_class = {"ready": "rg-status-ready", "processing": "rg-status-processing", "error": "rg-status-error"}.get(
                status, "rg-status-processing"
            )
            ocr_badge = ' <span class="rg-status-badge rg-status-ocr">OCR</span>' if meta.get("ocr_used") else ""
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(
                    f'<div class="rg-file-row">📄 {filename} '
                    f'<span class="rg-status-badge {badge_class}">{status}</span>{ocr_badge}</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("🗑️", key=f"del_{filename}", help=f"Delete {filename}"):
                    old_hashes = st.session_state.store.file_hashes - {meta.get("hash", "")}
                    st.session_state.store.remove_source(filename)
                    remaining_docs = st.session_state.store.docs
                    if remaining_docs:
                        texts = [d.page_content for d in remaining_docs]
                        vectors = st.session_state.engine.embed_texts(texts)
                        st.session_state.store = VectorStore()
                        st.session_state.store.add(vectors, remaining_docs)
                    else:
                        st.session_state.store = VectorStore()
                    st.session_state.store.file_hashes = old_hashes
                    st.session_state.uploaded_meta.pop(filename, None)
                    st.session_state.store.save(VECTOR_DB_DIR)
                    st.rerun()

        if st.button("🗑️ Delete All Files", use_container_width=True):
            st.session_state.store = VectorStore()
            st.session_state.uploaded_meta = {}
            st.session_state.store.save(VECTOR_DB_DIR)
            st.rerun()

    st.divider()
    st.subheader("🕘 Recent Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.sessions.new_session()
        st.session_state.pending_followups = []
        st.rerun()

    for session in st.session_state.sessions.ordered_sessions():
        is_active = session.id == st.session_state.sessions.active_id
        cols = st.columns([5, 1])
        with cols[0]:
            label = f"{'🟢 ' if is_active else ''}{session.title}"
            if st.button(label, key=f"session_{session.id}", use_container_width=True):
                st.session_state.sessions.switch_to(session.id)
                st.session_state.pending_followups = []
                st.rerun()
        with cols[1]:
            if len(st.session_state.sessions.sessions) > 1 and st.button("✕", key=f"delsess_{session.id}"):
                st.session_state.sessions.delete(session.id)
                st.rerun()

    st.text_input("🔎 Search chat history", key="chat_search_query", placeholder="Search all conversations…")
    if st.session_state.chat_search_query:
        hits = st.session_state.sessions.search(st.session_state.chat_search_query)
        if not hits:
            st.caption("No matches.")
        else:
            for session, msg in hits[:8]:
                snippet = search_snippet(msg.content, st.session_state.chat_search_query)
                if st.button(f"💬 {session.title}: {snippet}", key=f"hit_{session.id}_{msg.timestamp}_{msg.content[:10]}"):
                    st.session_state.sessions.switch_to(session.id)
                    st.rerun()

    if st.button("🧹 Clear Current Chat", use_container_width=True):
        st.session_state.sessions.active.memory.clear()
        st.session_state.pending_followups = []
        st.rerun()

    st.divider()
    with st.expander("⚙️ Settings", expanded=False):
        s = st.session_state.settings
        s["temperature"] = st.slider("Temperature", 0.0, 1.0, s["temperature"], 0.05)
        s["max_output_tokens"] = st.slider("Max Output Tokens", 128, 4096, s["max_output_tokens"], 128)
        s["top_k"] = st.slider("Top-K Retrieval", 1, 10, s["top_k"], 1)
        s["chunk_size"] = st.slider("Chunk Size", 200, 2000, s["chunk_size"], 50)
        s["chunk_overlap"] = st.slider("Chunk Overlap", 0, 500, s["chunk_overlap"], 10)
        st.caption("Chunk size/overlap apply to newly uploaded files.")

    st.session_state.search_mode = st.radio(
        "🔍 Search Mode", ["semantic", "keyword"], horizontal=True,
        index=0 if st.session_state.search_mode == "semantic" else 1,
    )

    st.divider()
    st.subheader("⬇️ Export Chat")
    active_memory = st.session_state.sessions.active.memory
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "TXT", active_memory.export_txt() or "No messages yet.",
            file_name="ragenius_chat.txt", use_container_width=True,
        )
    with ec2:
        st.download_button(
            "Markdown", active_memory.export_markdown() or "No messages yet.",
            file_name="ragenius_chat.md", use_container_width=True,
        )

    with st.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("- `/` — focus the chat input\n- Click a citation pill to jump to that source in **Retrieved Chunks**")

# ----------------------------------------------------------------------------
# Main area — tabs
# ----------------------------------------------------------------------------

tab_chat, tab_insights, tab_compare, tab_search = st.tabs(
    ["💬 Chat", "✨ Premium Insights", "🆚 Compare Documents", "🔎 Retrieved Chunks"]
)

# ---- Chat tab ----
with tab_chat:
    active_session = st.session_state.sessions.active
    active_memory = active_session.memory

    if st.session_state.store.is_empty():
        st.info("👋 Upload a PDF, CSV, TXT, or image file from the sidebar to start chatting with your documents.")

    for idx, msg in enumerate(active_memory.messages):
        bubble_class = "rg-bubble-user" if msg.role == "user" else "rg-bubble-assistant"
        st.markdown(
            f'<div class="{bubble_class}">{msg.content}'
            f'<div class="rg-timestamp">{msg.timestamp}</div></div>',
            unsafe_allow_html=True,
        )
        if msg.citations:
            st.markdown(
                "".join(f'<span class="rg-citation-pill">📎 {c}</span>' for c in msg.citations),
                unsafe_allow_html=True,
            )
        if msg.role == "assistant":
            cols = st.columns([1, 1, 1, 7])
            with cols[0]:
                st.button("📋 Copy", key=f"copy_{idx}", help="Select and copy the text above")
            with cols[1]:
                if st.button("🔄 Regenerate", key=f"regen_{idx}"):
                    st.session_state["_regenerate_idx"] = idx
                    st.rerun()
            with cols[2]:
                if st.button("🔊", key=f"speak_{idx}", help="Read this answer aloud"):
                    st.markdown(
                        f"<script>window.speechSynthesis.cancel(); "
                        f"window.speechSynthesis.speak(new SpeechSynthesisUtterance({json.dumps(msg.content)}));</script>",
                        unsafe_allow_html=True,
                    )

    if st.session_state.pending_followups:
        st.caption("Suggested follow-ups:")
        fcols = st.columns(len(st.session_state.pending_followups))
        for i, q in enumerate(st.session_state.pending_followups):
            with fcols[i]:
                if st.button(q, key=f"followup_{i}"):
                    st.session_state["_followup_query"] = q
                    st.rerun()

    input_col, mic_col = st.columns([9, 1])
    with input_col:
        query = st.chat_input("Ask a question about your documents…")
    with mic_col:
        from audio_recorder_streamlit import audio_recorder

        audio_bytes = audio_recorder(text="", icon_size="1.5x", key="voice_recorder")

    if "_followup_query" in st.session_state:
        query = st.session_state.pop("_followup_query")

    if audio_bytes and audio_bytes != st.session_state.get("_last_audio"):
        st.session_state["_last_audio"] = audio_bytes
        with st.spinner("🎙️ Transcribing…"):
            try:
                query = transcribe_audio(st.session_state.client, audio_bytes, mime_type="audio/wav")
                st.toast(f"Heard: “{query}”")
            except TranscriptionError as exc:
                st.error(f"🎙️ {exc}")
                query = None

    if query:
        active_memory.add("user", query)
        active_session.maybe_set_title(query)
        st.markdown(
            f'<div class="rg-bubble-user">{query}<div class="rg-timestamp">{human_timestamp()}</div></div>',
            unsafe_allow_html=True,
        )

        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '<div class="rg-bubble-assistant rg-typing"><span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )

        answer_chunks: list[str] = []
        response_placeholder = st.empty()
        s = st.session_state.settings
        gen = answer_question(
            st.session_state.client,
            query,
            st.session_state.store,
            st.session_state.engine,
            history=active_memory.as_prompt_history(),
            top_k=s["top_k"],
            mode=st.session_state.search_mode,
            temperature=s["temperature"],
            max_output_tokens=s["max_output_tokens"],
        )
        typing_placeholder.empty()
        retrieved = []
        try:
            while True:
                piece = next(gen)
                answer_chunks.append(piece)
                response_placeholder.markdown(
                    f'<div class="rg-bubble-assistant">{"".join(answer_chunks)}▌</div>',
                    unsafe_allow_html=True,
                )
        except StopIteration as stop:
            retrieved = stop.value or []

        full_answer = "".join(answer_chunks) or "I couldn't generate a response."
        citations = [format_citation(c.doc) for c in retrieved]
        response_placeholder.markdown(
            f'<div class="rg-bubble-assistant">{full_answer}'
            f'<div class="rg-timestamp">{human_timestamp()}</div></div>',
            unsafe_allow_html=True,
        )
        if citations:
            st.markdown(
                "".join(f'<span class="rg-citation-pill">📎 {c}</span>' for c in citations),
                unsafe_allow_html=True,
            )

        active_memory.add("assistant", full_answer, citations=citations)
        st.session_state.last_retrieved = retrieved
        st.session_state.last_query = query

        if st.session_state.client is not None and full_answer:
            followup_raw = generate_text(
                st.session_state.client,
                build_followup_prompt(query, full_answer),
                max_output_tokens=150,
            )
            suggestions = [
                line.split(".", 1)[-1].strip("-• ").strip()
                for line in followup_raw.splitlines()
                if line.strip() and any(ch.isalpha() for ch in line)
            ]
            st.session_state.pending_followups = suggestions[:3]
        st.rerun()

# ---- Premium Insights tab ----
with tab_insights:
    ready_files = [f for f, m in st.session_state.uploaded_meta.items() if m.get("status") == "ready"]
    if not ready_files:
        st.info("Upload and process a document to unlock summaries, insights, keywords, and FAQs.")
    else:
        selected = st.selectbox("Choose a document", ready_files)
        meta = st.session_state.uploaded_meta[selected]

        c1, c2, c3 = st.columns(3)
        c1.metric("Word Count", f"{meta['word_count']:,}")
        c2.metric("Est. Reading Time", f"{meta['reading_time']} min")
        c3.metric("Chunks Indexed", meta["chunks"])
        if meta.get("ocr_used"):
            st.caption("📷 Text extracted via OCR for this document.")

        text_sample = truncate(meta["raw_text"], 12000)  # keep prompt sizes reasonable

        action_cols = st.columns(4)
        actions = {
            "📝 Summary": ("summary", build_summary_prompt),
            "💡 Key Insights": ("insights", build_key_insights_prompt),
            "🏷️ Keywords": ("keywords", build_keywords_prompt),
            "❓ FAQs": ("faqs", build_faq_prompt),
        }
        for col, (label, (state_key, builder)) in zip(action_cols, actions.items()):
            with col:
                if st.button(label, key=f"btn_{state_key}", use_container_width=True):
                    with st.spinner("Generating…"):
                        result = generate_text(st.session_state.client, builder(text_sample), max_output_tokens=600)
                    st.session_state[f"insight_{selected}_{state_key}"] = result

        for label, (state_key, _) in actions.items():
            cache_key = f"insight_{selected}_{state_key}"
            if cache_key in st.session_state:
                with st.container():
                    st.markdown(f'<div class="rg-card"><b>{label}</b><br><br>{st.session_state[cache_key]}</div>', unsafe_allow_html=True)

# ---- Compare Documents tab ----
with tab_compare:
    ready_files = [f for f, m in st.session_state.uploaded_meta.items() if m.get("status") == "ready"]
    if len(ready_files) < 2:
        st.info("Upload at least two documents to compare them side by side.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            doc_a = st.selectbox("Document A", ready_files, key="cmp_a")
        with c2:
            remaining = [f for f in ready_files if f != doc_a]
            doc_b = st.selectbox("Document B", remaining, key="cmp_b")

        if st.button("🆚 Compare", use_container_width=True):
            meta_a = st.session_state.uploaded_meta[doc_a]
            meta_b = st.session_state.uploaded_meta[doc_b]
            with st.spinner("Comparing documents…"):
                prompt = build_comparison_prompt(
                    doc_a, truncate(meta_a["raw_text"], 8000),
                    doc_b, truncate(meta_b["raw_text"], 8000),
                )
                result = generate_text(st.session_state.client, prompt, max_output_tokens=1200)
            st.session_state["_comparison_result"] = result

        if "_comparison_result" in st.session_state:
            st.markdown(f'<div class="rg-card">{st.session_state["_comparison_result"]}</div>', unsafe_allow_html=True)

# ---- Retrieved Chunks tab ----
with tab_search:
    st.caption("Chunks retrieved for your most recent question, with similarity scores and highlighted terms.")
    if not st.session_state.last_retrieved:
        st.info("Ask a question in the Chat tab to see retrieved chunks here.")
    else:
        for r in st.session_state.last_retrieved:
            with st.expander(f"📎 {format_citation(r.doc)} · score {r.score:.3f}"):
                highlighted = highlight_terms(r.doc.page_content, st.session_state.last_query)
                st.markdown(highlighted, unsafe_allow_html=True)
