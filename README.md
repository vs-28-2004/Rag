# 🧠 RAGenius AI — Intelligent Document Chatbot

A ChatGPT-style Retrieval-Augmented Generation (RAG) web app. Upload PDFs, CSVs,
TXT files, and images, and ask questions grounded in your own documents — with
citations, streaming responses, voice input/output, and a premium glassmorphic UI.

---

## ✨ Features

**Core RAG**
- **Multi-format upload**: PDF, CSV, TXT, and images (JPG/PNG/WEBP/BMP) with drag-and-drop, duplicate detection, and per-file delete
- **Full RAG pipeline**: extraction → chunking (`RecursiveCharacterTextSplitter`, 800/150) → Gemini embeddings (with local Sentence-Transformers fallback) → FAISS retrieval → grounded Gemini generation
- **OCR fallback**: scanned/image-only PDF pages and standalone image uploads are automatically run through Tesseract OCR and indexed like any other document
- **Source citations** on every answer (page / row range / section / OCR flag)
- **Two search modes**: semantic (vector) and keyword, with matched-term highlighting in retrieved chunks

**Chat experience**
- Streaming responses with Markdown rendering, timestamps, copy, and regenerate
- **Voice input**: record a question with the mic button — transcribed via Gemini's multimodal audio understanding
- **Voice output**: click 🔊 on any answer to have it read aloud (browser Speech Synthesis)
- Suggested follow-up questions after every answer
- **Recent conversations**: multi-session chat history in the sidebar, auto-titled from your first message, with a **New Chat** button
- **Search chat history** across all past conversations
- Chat export to TXT or Markdown
- Keyboard shortcut: press `/` to jump to the chat box
- First-run onboarding dialog

**Document tools**
- **Premium AI insights**: summary, key insights, keywords, auto-FAQs, word count, reading time
- **Compare Documents**: side-by-side AI comparison of any two uploaded files (summary, similarities, differences)
- Configurable settings: temperature, max tokens, top-K, chunk size/overlap
- Dark/light mode, responsive glassmorphic UI

## 📸 Screenshots

| Chat | Premium Insights | Compare Documents | Retrieved Chunks |
|---|---|---|---|
| _add screenshot_ | _add screenshot_ | _add screenshot_ | _add screenshot_ |

## 🏗️ Architecture

```
flowchart TD
    A[User uploads PDF/CSV/TXT/Image] --> B[backend/loader.py<br/>Extract text + metadata]
    B -->|no text layer| B2[backend/ocr.py<br/>Tesseract OCR fallback]
    B2 --> C
    B --> C[backend/splitter.py<br/>Chunk: 800 chars / 150 overlap]
    C --> D[backend/embeddings.py<br/>Gemini Embeddings API]
    D -->|fallback on failure| E[Sentence-Transformers local model]
    D --> F[backend/vector_store.py<br/>FAISS Index]
    E --> F
    F --> G[backend/retriever.py<br/>Top-K semantic / keyword search + highlighting]
    G --> H[backend/prompts.py<br/>Build grounded prompt]
    H --> I[backend/rag_chain.py<br/>Gemini "gemini-3.1-flash-lite" — streaming]
    I --> J[app.py<br/>Streamlit UI: chat, citations, insights, compare, voice]
    K[Mic input] --> L[backend/voice.py<br/>Gemini audio transcription] --> J

## 📦 Folder Structure

```
ragenius-ai/
├── app.py                 # Streamlit entrypoint
├── requirements.txt
├── .env.example
├── README.md
├── backend/
│   ├── loader.py           # PDF/CSV/TXT/image extraction + metadata
│   ├── ocr.py               # Tesseract OCR for scanned PDFs & images
│   ├── splitter.py         # Chunking
│   ├── embeddings.py       # Gemini + Sentence-Transformers fallback
│   ├── vector_store.py     # FAISS wrapper (add/search/save/load)
│   ├── retriever.py        # Semantic + keyword retrieval, citations, highlighting
│   ├── rag_chain.py        # Gemini generation (streaming + single-shot)
│   ├── voice.py             # Gemini audio transcription for voice input
│   ├── chat_memory.py      # Multi-session chat history + export + search
│   ├── prompts.py          # Prompt templates (RAG, insights, comparison)
│   └── utils.py            # Shared helpers
├── assets/
│   ├── styles.css
│   └── logo.png
├── data/uploads/
└── vector_db/              # Persisted FAISS index (auto-created)
```

## 🚀 Installation

```bash
git clone <your-repo-url> ragenius-ai
cd Rag
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**System dependency for OCR:** install Tesseract separately (it's a binary, not a Python package):

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr
# macOS
brew install tesseract
```

If Tesseract isn't installed, OCR-dependent uploads (scanned PDFs, images) will show
a friendly error, but the rest of the app keeps working normally.

## 🔑 Environment Setup

```bash
cp .env.example .env
```

Edit `.env` and add your key from [Google AI Studio](https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL="gemini-3.1-flash-lite"
GEMINI_EMBEDDING_MODEL=text-embedding-004
```

If no key is set, the app still runs: embeddings fall back to a local
Sentence-Transformers model, and the chat will show a friendly notice instead
of a generated answer. Voice input requires a valid key (it uses Gemini for
transcription); voice output works regardless since it runs in the browser.

## ▶️ Running Locally

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Grant
microphone permission in your browser if you want to use voice input.

## ☁️ Streamlit Community Cloud Deployment

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch, and `app.py` as the entrypoint.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Add a `packages.txt` file at the repo root containing `tesseract-ocr` so
   Streamlit Cloud installs the OCR binary automatically.
6. Deploy. FAISS, Sentence-Transformers, and the rest install from `requirements.txt`.

## 🗺️ Future Roadmap

- Keyboard shortcuts beyond `/` (e.g. `Ctrl+K` command palette, `Ctrl+D` theme toggle)
- Persisting recent conversations to disk (currently in-memory per session)
- Multi-language OCR and voice transcription
- Exact character-span citation highlighting inside the generated answer itself

## ⚠️ Notes

- Max upload size per file: 50MB (configurable in `backend/utils.py`).
- Deleting a file rebuilds the FAISS index from the remaining chunks.
- The vector index persists to `vector_db/` between runs; chat sessions do not (they reset on app restart).
- OCR is best-effort: heavily skewed, low-resolution, or handwritten scans may extract poorly.
