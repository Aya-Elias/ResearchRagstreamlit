# Research Paper Discovery & Interactive QA Assistant (Streamlit)

A web app converted from the original `paper_qa_pipeline` notebook — search arXiv
(or upload your own PDF(s)), get a summary, ask questions with page-cited answers,
and compare 2+ papers side by side.

## 1. Install (one-time)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> The first run will be a bit slow since `sentence-transformers` downloads the
> embedding and reranker models (a few hundred MB). They're cached locally after that.

## 2. API keys

You'll need:
- **Groq API Key** (required) — from [console.groq.com](https://console.groq.com)
- **Gemini API Key** (optional, only for AI figure/image analysis) — from [aistudio.google.com](https://aistudio.google.com)

Nothing goes in the code — enter both in the app's sidebar when it opens.

## 3. Run

```bash
streamlit run streamlit_app.py
```

The app opens automatically at `http://localhost:8501`.

## 4. How to use

1. Enter your Groq API Key in the sidebar (this is the only thing kept there —
   everything else is on the main page).
2. On the main page: either **Search arXiv** with a topic/question, or
   **Upload PDF(s)** directly.
3. From the search results, either:
   - Click **"Open this paper (chat)"** to chat with one paper, or
   - Tick the **Compare** checkbox on 2+ papers, then click **"Compare N selected papers"**.
4. Wait for processing (download → extract → clean → summarize → build embeddings).
5. Ask questions in the chat box. Single-paper answers cite page numbers;
   comparison answers cite both the paper tag (P1, P2, ...) and page number.
6. Use **"Back to results"** to return to the same search results without
   re-searching, or **"New search"** to start over.

## Notes

- "Analyze figures/images with AI" sends every figure in the paper to Gemini —
  it's slower and costs more calls, so leave it off unless you need it.
- Table extraction is on by default and gets folded into the same knowledge base
  so the assistant can reference numbers that live in tables.
- If you deploy this on a server (not your own machine), store the API keys as
  Streamlit Secrets instead of typing them in each time.
