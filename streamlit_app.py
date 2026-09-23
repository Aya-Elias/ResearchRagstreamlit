"""
Smart Research Paper Discovery & Interactive QA Assistant — Streamlit App
Ported from the original Colab notebook (paper_qa_pipeline).

Run with:  streamlit run streamlit_app.py
"""

import io
import os
import re
import time
from collections import Counter

import numpy as np
import requests
import streamlit as st
import xml.etree.ElementTree as ET

# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(
    page_title="مساعد اكتشاف الأبحاث العلمية",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .paper-card {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .score-badge {
        display:inline-block; padding:2px 10px; border-radius:12px;
        background:#2563eb22; color:#2563eb; font-weight:600; font-size:0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Sidebar — API keys & options
# =============================================================================
with st.sidebar:
    st.title("⚙️ الإعدادات")

    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        value=st.session_state.get("groq_api_key", ""),
        help="مطلوب لإعادة صياغة البحث، التلخيص، والإجابة على الأسئلة.",
    )
    gemini_api_key = st.text_input(
        "Gemini API Key (اختياري)", type="password",
        value=st.session_state.get("gemini_api_key", ""),
        help="مطلوب فقط لو فعّلتي تحليل الصور/الأشكال بالـ AI.",
    )
    st.session_state["groq_api_key"] = groq_api_key
    st.session_state["gemini_api_key"] = gemini_api_key

    st.divider()
    st.subheader("خيارات المعالجة")
    extract_tables_opt = st.checkbox("استخراج الجداول من الـ PDF", value=True)
    extract_images_opt = st.checkbox(
        "تحليل الصور/الأشكال بالـ AI (Gemini)", value=False,
        help="بيبطئ المعالجة لأنه بيبعت كل صورة للموديل على حدة.",
    )
    top_k = st.slider("عدد المقاطع المستخدمة في كل إجابة", 3, 10, 5)

    st.divider()
    if st.button("🔄 ابدأ من جديد بالكامل", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k not in ("groq_api_key", "gemini_api_key"):
                del st.session_state[k]
        st.rerun()

st.title("📄 مساعد اكتشاف الأبحاث والإجابة التفاعلية")
st.caption("دوري عن ورقة بحثية على arXiv أو ارفعي PDF بتاعك، واسألي أي سؤال عنها")

if not groq_api_key:
    st.info("⬅️ حطي Groq API Key في القائمة الجانبية عشان تقدري تبدئي.")
    st.stop()

from groq import Groq  # noqa: E402

groq_client = Groq(api_key=groq_api_key)

GROQ_MODEL = "openai/gpt-oss-20b"


# =============================================================================
# Cached heavy models (loaded once per server process)
# =============================================================================
@st.cache_resource(show_spinner="بيتم تحميل نموذج الـ embeddings (أول مرة بس)...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="بيتم تحميل نموذج الـ reranker (أول مرة بس)...")
def load_reranker_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-base")


embedding_model = load_embedding_model()


def embed_text(text: str):
    return embedding_model.encode(text)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================================================================
# Phase A — Discovery (arXiv search + semantic ranking)
# =============================================================================
def search_arxiv(query: str, max_results: int = 20):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        published = entry.find("atom:published", ns).text[:10]
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]

        papers.append({
            "title": title, "authors": authors, "published": published,
            "summary": summary, "pdf_url": pdf_url,
        })
    return papers


def rewrite_query(user_question: str) -> str:
    prompt = f"""Extract 3-6 key academic search terms from this question.
Return ONLY the terms separated by spaces, no explanation.

Question: {user_question}
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def semantic_search_papers(user_question: str, max_results: int = 20):
    search_query = rewrite_query(user_question)
    candidates = search_arxiv(search_query, max_results=max_results)
    question_embedding = embed_text(user_question)

    for paper in candidates:
        paper_embedding = embed_text(paper["summary"])
        paper["relevance_score"] = float(cosine_similarity(question_embedding, paper_embedding))

    ranked = sorted(candidates, key=lambda p: p["relevance_score"], reverse=True)
    return search_query, ranked


# =============================================================================
# Phase B — Download, Extract & Clean Text
# =============================================================================
def download_pdf(pdf_url: str, save_dir: str = "papers") -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = pdf_url.split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    local_path = os.path.join(save_dir, filename)

    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def extract_text_from_pdf(pdf_path: str) -> list:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = [{"page_number": i, "text": p.get_text()} for i, p in enumerate(doc, start=1)]
    doc.close()
    return pages


def _fix_hyphenation(text: str) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_page_number_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"(page\s*)?\d{1,4}(\s*of\s*\d{1,4})?", stripped, re.IGNORECASE))


def _detect_repeated_lines(pages: list, min_repeat_ratio: float = 0.4) -> set:
    first_lines, last_lines = [], []
    for page in pages:
        lines = [l.strip() for l in page["text"].split("\n") if l.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])

    total_pages = len(pages)
    threshold = max(2, int(total_pages * min_repeat_ratio))
    return {line for line, count in Counter(first_lines + last_lines).items() if count >= threshold}


def clean_pages(pages: list) -> list:
    repeated_lines = _detect_repeated_lines(pages)
    cleaned_pages = []
    for page in pages:
        lines = page["text"].split("\n")
        kept = [l for l in lines if l.strip() not in repeated_lines and not _is_page_number_line(l)]
        cleaned_text = "\n".join(kept)
        cleaned_text = _fix_hyphenation(cleaned_text)
        cleaned_text = _normalize_whitespace(cleaned_text)
        cleaned_pages.append({"page_number": page["page_number"], "text": cleaned_text})
    return cleaned_pages


def generate_executive_summary(cleaned_pages: list) -> str:
    sample_text = "\n\n".join(p["text"] for p in cleaned_pages[:4])
    prompt = f"""Based on this excerpt from an academic paper, write a short executive summary
covering: (1) the paper's goal/problem, (2) the methodology used, (3) key findings/results.
Keep it to 4-6 sentences total.

Paper excerpt:
{sample_text[:6000]}
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# Optional — Tables
# =============================================================================
def extract_tables_from_pdf(pdf_path: str) -> list:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    all_tables = []
    for page_num, page in enumerate(doc, start=1):
        tabs = page.find_tables()
        for idx, table in enumerate(tabs.tables):
            all_tables.append({"page_number": page_num, "table_index": idx, "rows": table.extract()})
    doc.close()
    return all_tables


def table_to_text(table: dict) -> str:
    lines = [f"[Table from page {table['page_number']}]"]
    for row in table["rows"]:
        clean_row = [cell.strip() if cell else "" for cell in row]
        lines.append(" | ".join(clean_row))
    return "\n".join(lines)


# =============================================================================
# Optional — Images via vision LLM (Gemini)
# =============================================================================
def extract_images_from_pdf(pdf_path: str) -> list:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "page_number": page_num, "image_index": img_index,
                "image_bytes": base_image["image"],
            })
    doc.close()
    return images


def describe_image(gemini_client, image_bytes: bytes, max_retries: int = 3) -> str:
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    "Describe this figure from an academic paper in 2-4 sentences. "
                    "If it's a diagram, explain what it represents. "
                    "If it's a chart/plot, describe what's being compared.",
                    image,
                ],
            )
            return response.text.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                raise


def process_images(gemini_client, images: list) -> list:
    described = []
    for img in images:
        try:
            description = describe_image(gemini_client, img["image_bytes"])
            described.append({
                "page_number": img["page_number"], "type": "figure",
                "text": f"[Figure from page {img['page_number']}]: {description}",
            })
        except Exception:
            pass
    return described


# =============================================================================
# Chunking
# =============================================================================
def split_into_paragraphs(text: str) -> list:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def adaptive_chunk_page(page_text, page_number, min_chunk_chars=300, max_chunk_chars=1400, overlap_chars=150):
    paragraphs = split_into_paragraphs(page_text)
    raw_chunks = []
    buffer = ""

    for para in paragraphs:
        if len(para) > max_chunk_chars:
            sentences = split_into_sentences(para)
            sentence_buffer = ""
            for sentence in sentences:
                if len(sentence_buffer) + len(sentence) <= max_chunk_chars:
                    sentence_buffer += (" " if sentence_buffer else "") + sentence
                else:
                    if sentence_buffer:
                        raw_chunks.append(sentence_buffer)
                    sentence_buffer = sentence
            if sentence_buffer:
                raw_chunks.append(sentence_buffer)
        else:
            if len(buffer) + len(para) <= max_chunk_chars:
                buffer += ("\n\n" if buffer else "") + para
            else:
                if buffer:
                    raw_chunks.append(buffer)
                buffer = para
    if buffer:
        raw_chunks.append(buffer)

    merged_chunks = []
    carry = ""
    for chunk in raw_chunks:
        candidate = (carry + "\n\n" + chunk) if carry else chunk
        if len(candidate) < min_chunk_chars:
            carry = candidate
        else:
            merged_chunks.append(candidate)
            carry = ""
    if carry:
        if merged_chunks:
            merged_chunks[-1] += "\n\n" + carry
        else:
            merged_chunks.append(carry)

    final_chunks = []
    for i, chunk in enumerate(merged_chunks):
        if i > 0 and overlap_chars > 0:
            prev_tail = merged_chunks[i - 1][-overlap_chars:]
            chunk = prev_tail + " ... " + chunk
        final_chunks.append({"page_number": page_number, "type": "text", "text": chunk})
    return final_chunks


def chunk_cleaned_pages(cleaned_pages: list) -> list:
    all_chunks = []
    for page in cleaned_pages:
        all_chunks.extend(adaptive_chunk_page(page["text"], page["page_number"]))
    return all_chunks


# =============================================================================
# Vector store (FAISS)
# =============================================================================
class PaperVectorStore:
    def __init__(self, chunks: list):
        import faiss
        self.chunks = chunks
        vectors = np.array([embed_text(c["text"]) for c in chunks], dtype="float32")
        faiss.normalize_L2(vectors)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 5) -> list:
        import faiss
        query_vector = np.array([embed_text(query)], dtype="float32")
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results


def build_paper_vector_store(paper_context: dict, do_tables: bool, do_images: bool, gemini_client=None) -> "PaperVectorStore":
    text_chunks = chunk_cleaned_pages(paper_context["cleaned_pages"])

    table_chunks = []
    if do_tables:
        try:
            tables = extract_tables_from_pdf(paper_context["local_path"])
            table_chunks = [
                {"page_number": t["page_number"], "type": "table", "text": table_to_text(t)}
                for t in tables
            ]
        except Exception as e:
            st.warning(f"تعذر استخراج الجداول: {e}")

    image_chunks = []
    if do_images and gemini_client is not None:
        try:
            images = extract_images_from_pdf(paper_context["local_path"])
            image_chunks = process_images(gemini_client, images)
        except Exception as e:
            st.warning(f"تعذر تحليل الصور: {e}")

    all_chunks = text_chunks + table_chunks + image_chunks
    return PaperVectorStore(all_chunks), len(text_chunks), len(table_chunks), len(image_chunks)


def rerank_chunks(question: str, chunks: list, top_k: int = 5) -> list:
    if not chunks:
        return chunks
    reranker_model = load_reranker_model()
    pairs = [[question, chunk["text"]] for chunk in chunks]
    rerank_scores = reranker_model.predict(pairs)
    for chunk, score in zip(chunks, rerank_scores):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# =============================================================================
# QA
# =============================================================================
def build_qa_context(retrieved_chunks: list) -> str:
    parts = []
    for chunk in retrieved_chunks:
        label = f"[Source: page {chunk['page_number']}, type: {chunk['type']}]"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


QA_SYSTEM_PROMPT = """You are answering questions about a specific academic paper, using ONLY the
context provided by the user in each turn. Follow these rules strictly:

1. Answer only from the given context. Do not use outside knowledge.
2. Do not infer or extrapolate beyond what is explicitly stated.
3. If the answer isn't in the context, say so clearly — do not guess.
4. Always cite the page number(s) your answer comes from, like (page 5).
5. For specific numbers, metrics, or exact terms, quote them precisely as written in the
   context, but keep any direct quotation short (under 15 words) — paraphrase the rest.
6. If sources seem to conflict with each other, mention both versions explicitly and note
   which page each one comes from.
7. State your confidence: if the answer is fully supported, answer normally; if it's only
   partially supported or ambiguous, say so explicitly.
8. Keep answers concise (2-4 sentences) unless the question genuinely needs more detail.
9. Don't refer to "the context" or "the provided text" — answer as if you've read the paper.
10. Use the recent conversation turns only to understand follow-up questions (e.g. "what about
    its size?" referring to something just discussed) — never as a source of facts.
"""


def answer_question(vector_store: "PaperVectorStore", question: str, conversation_history: list, top_k: int = 5) -> dict:
    retrieved = vector_store.search(question, top_k=15)
    retrieved = rerank_chunks(question, retrieved, top_k=top_k)
    context = build_qa_context(retrieved)

    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT}]
    for turn in conversation_history[-3:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_prompt})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.2, messages=messages
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": [(c["page_number"], c["type"]) for c in retrieved],
    }


# =============================================================================
# Shared processing helper (search-picked paper OR uploaded PDF)
# =============================================================================
def process_paper_pipeline(paper: dict, local_path: str):
    with st.status("جاري معالجة الورقة البحثية...", expanded=True) as status:
        st.write("📖 استخراج النص من الـ PDF...")
        pages = extract_text_from_pdf(local_path)

        st.write("🧹 تنظيف النص (إزالة الهيدر/الفوتر وأرقام الصفحات)...")
        cleaned = clean_pages(pages)

        st.write("✍️ إنشاء ملخص تنفيذي...")
        summary = generate_executive_summary(cleaned)

        paper_context = {
            "paper": paper, "local_path": local_path,
            "cleaned_pages": cleaned, "summary": summary,
        }

        st.write("🧩 تقطيع النص وبناء قاعدة المتجهات (embeddings)...")
        gemini_client = None
        if extract_images_opt and gemini_api_key:
            from google import genai
            gemini_client = genai.Client(api_key=gemini_api_key)

        vector_store, n_text, n_table, n_img = build_paper_vector_store(
            paper_context, extract_tables_opt, extract_images_opt, gemini_client
        )
        status.update(label="✅ الورقة جاهزة!", state="complete", expanded=False)

    st.session_state["paper_context"] = paper_context
    st.session_state["vector_store"] = vector_store
    st.session_state["chunk_counts"] = (n_text, n_table, n_img)
    st.session_state["conversation_history"] = []
    st.session_state["stage"] = "chat"
    st.rerun()


# =============================================================================
# Session state init
# =============================================================================
st.session_state.setdefault("stage", "search")
st.session_state.setdefault("ranked_list", [])
st.session_state.setdefault("shown_count", 0)
st.session_state.setdefault("conversation_history", [])

# =============================================================================
# STAGE: search / upload
# =============================================================================
if st.session_state["stage"] == "search":
    tab_search, tab_upload = st.tabs(["🔎 دوري في arXiv", "📤 ارفعي PDF بتاعك"])

    with tab_search:
        user_question = st.text_input(
            "عن إيه موضوع البحث اللي بتدوري عليه؟",
            placeholder="مثال: How do transformers detect anomalies in time series data?",
        )
        if st.button("بحث", type="primary", disabled=not user_question.strip()):
            with st.spinner("بيتم البحث والترتيب حسب الصلة..."):
                search_query, ranked = semantic_search_papers(user_question)
            st.session_state["ranked_list"] = ranked
            st.session_state["shown_count"] = 0
            st.session_state["last_search_query"] = search_query
            st.session_state["stage"] = "results"
            st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader("اختاري ملف PDF", type=["pdf"])
        if uploaded_file is not None and st.button("معالجة الملف المرفوع", type="primary"):
            os.makedirs("papers", exist_ok=True)
            local_path = os.path.join("papers", uploaded_file.name)
            with open(local_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            fake_paper = {"title": uploaded_file.name, "authors": [], "published": "", "summary": "", "pdf_url": None}
            process_paper_pipeline(fake_paper, local_path)

# =============================================================================
# STAGE: results (pick a paper)
# =============================================================================
elif st.session_state["stage"] == "results":
    st.subheader("نتائج البحث")
    if "last_search_query" in st.session_state:
        st.caption(f"الكلمات المستخدمة في البحث: `{st.session_state['last_search_query']}`")

    if st.button("⬅️ رجوع للبحث"):
        st.session_state["stage"] = "search"
        st.rerun()

    ranked_list = st.session_state["ranked_list"]
    shown = max(st.session_state["shown_count"], 5)
    page = ranked_list[:shown]

    if not page:
        st.warning("مفيش نتائج. جربي صياغة تانية للسؤال.")
    for i, paper in enumerate(page, start=1):
        with st.container(border=True):
            st.markdown(
                f"**{i}. {paper['title']}**  "
                f"<span class='score-badge'>score: {paper['relevance_score']:.3f}</span>",
                unsafe_allow_html=True,
            )
            authors = ", ".join(paper["authors"][:4]) + (" et al." if len(paper["authors"]) > 4 else "")
            st.caption(f"{authors} — {paper['published']}")
            with st.expander("الملخص (Abstract)"):
                st.write(paper["summary"])
            if st.button("✅ اختاري الورقة دي", key=f"pick_{i}"):
                with st.spinner("جاري تحميل الـ PDF..."):
                    local_path = download_pdf(paper["pdf_url"])
                process_paper_pipeline(paper, local_path)

    col1, col2 = st.columns(2)
    with col1:
        if shown < len(ranked_list):
            if st.button("⬇️ عرض المزيد"):
                st.session_state["shown_count"] = shown + 5
                st.rerun()
    with col2:
        st.write("مش لاقية ورقة مناسبة؟ ارفعي PDF بتاعك من تبويب البحث.")

# =============================================================================
# STAGE: chat
# =============================================================================
elif st.session_state["stage"] == "chat":
    paper_context = st.session_state["paper_context"]
    paper = paper_context["paper"]
    n_text, n_table, n_img = st.session_state.get("chunk_counts", (0, 0, 0))

    left, right = st.columns([2, 1])
    with left:
        st.subheader(paper["title"])
        if paper.get("authors"):
            st.caption(", ".join(paper["authors"][:6]) + (" et al." if len(paper["authors"]) > 6 else ""))
        with st.expander("📋 الملخص التنفيذي", expanded=True):
            st.write(paper_context["summary"])
    with right:
        st.metric("مقاطع نصية", n_text)
        st.metric("جداول", n_table)
        st.metric("أشكال محللة", n_img)
        if st.button("🔁 ابحث عن ورقة تانية"):
            st.session_state["stage"] = "results"
            st.rerun()
        if st.button("🆕 بحث جديد من الأول"):
            st.session_state["stage"] = "search"
            st.session_state["ranked_list"] = []
            st.rerun()

    st.divider()
    st.subheader("💬 اسألي عن الورقة")

    for turn in st.session_state["conversation_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            pages_cited = sorted(set(p for p, _ in turn["sources"]))
            st.caption(f"📌 المصادر: صفحات {pages_cited}")

    question = st.chat_input("اكتبي سؤالك عن الورقة...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("بيتم البحث في الورقة وصياغة الإجابة..."):
                result = answer_question(
                    st.session_state["vector_store"], question,
                    st.session_state["conversation_history"], top_k=top_k,
                )
            st.write(result["answer"])
            pages_cited = sorted(set(p for p, _ in result["sources"]))
            st.caption(f"📌 المصادر: صفحات {pages_cited}")

        st.session_state["conversation_history"].append({
            "question": question, "answer": result["answer"], "sources": result["sources"],
        })
