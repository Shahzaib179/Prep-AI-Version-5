"""Tutor document library: the "Personalized Learning" pipeline, made permanent for tutors.

    upload -> extract text (page numbers kept) -> overlapping chunks -> tokenise (keyword index)
           -> sentence-transformer embeddings -> vector index (FAISS) with per-chunk metadata
           -> hybrid search (semantic + BM25 keyword) -> passages handed to the quiz writer

The student Personalized Learning mode keeps everything in the browser session. Here the chunks, tokens and embeddings are
stored in the database (tables ``tutor_documents`` and ``tutor_doc_chunks``), so a document is processed ONCE and survives
Streamlit Cloud restarts. The FAISS index itself is cheap to rebuild from the stored vectors, so it is built in memory and
cached per process, exactly like the student memory index. Every query is scoped to the tutor who owns the document.

Pure logic (no Streamlit). The embedding function is injectable, so tests run without downloading a model.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from datetime import datetime
from typing import Any, Callable

import numpy as np

import doc_quiz
from db_core import cached_read, connect, db_identity

MAX_DOCS_PER_TUTOR = 30
MAX_CHUNKS_PER_TUTOR = 6000          # about 6 MB of vectors; keeps a free database and a free host comfortable
EMBED_BATCH = 64
W_SEMANTIC, W_KEYWORD = 0.65, 0.35
MODES = ("hybrid", "semantic", "keyword")

Embedder = Callable[[list[str]], np.ndarray]
Progress = Callable[[str, float], None]

# ---------------------------------------------------------------------------------------------- tokenising
_STOP = frozenset("""the and for with from this that what where when which into about how why are is of to in a an on be as at by it its or if
not but can will was were has have had you your they their them than then also such these those there here one two may more most
some any each per via our we us do does did""".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]+")


def _stem(w: str) -> str:
    """Very light suffix stripping so 'enzymes' matches 'enzyme' and 'cells' matches 'cell'. Short words are left alone."""
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith(("sses", "xes", "ches", "shes", "zes")):
        return w[:-2]
    for suf in ("ing", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lower-case keywords (stop words removed, light stemming). The same function indexes chunks and reads queries."""
    return [_stem(w) for w in _WORD.findall((text or "").lower()) if w not in _STOP]


# ---------------------------------------------------------------------------------------------- keyword index
class KeywordIndex:
    """BM25 (Okapi) over the stored chunk tokens. Exact keyword matching that complements embeddings (formulas, names, numbers)."""

    def __init__(self, token_lists: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in token_lists]
        self.len = np.array([len(t) for t in token_lists], dtype="float32")
        self.avg = float(self.len.mean()) if len(self.len) else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(token_lists)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * float(self.len[i]) / (self.avg or 1.0))
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        return out


# ---------------------------------------------------------------------------------------------- vector index
class _NumpyIndex:
    """Exact inner-product search with the same behaviour as faiss.IndexFlatIP (used only if FAISS cannot be imported)."""

    def __init__(self, dim: int):
        self.v = np.empty((0, dim), dtype="float32")

    @property
    def ntotal(self) -> int:
        return int(self.v.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self.v = np.vstack([self.v, np.asarray(vectors, dtype="float32")])

    def search(self, q: np.ndarray, k: int):
        sims = np.asarray(q, dtype="float32") @ self.v.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, 1), idx


_FAISS_OK: bool | None = None


def _faiss_works() -> bool:
    global _FAISS_OK
    if _FAISS_OK is None:
        try:
            import faiss
            ix = faiss.IndexFlatIP(4)
            ix.add(np.ones((1, 4), dtype="float32"))
            d, i = ix.search(np.ones((1, 4), dtype="float32"), 1)
            _FAISS_OK = bool(int(i[0][0]) == 0 and abs(float(d[0][0]) - 4.0) < 1e-3)
        except Exception:  # noqa: BLE001 - missing or incompatible FAISS build
            _FAISS_OK = False
    return _FAISS_OK


def make_index(vectors: np.ndarray):
    """FAISS IndexFlatIP over the stored vectors (cosine similarity, because the vectors are normalised)."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    if _faiss_works():
        import faiss
        ix = faiss.IndexFlatIP(vectors.shape[1])
    else:
        ix = _NumpyIndex(vectors.shape[1])
    if len(vectors):
        ix.add(vectors)
    return ix


def index_kind() -> str:
    return "FAISS" if _faiss_works() else "numpy (FAISS not installed)"


def default_embedder(texts: list[str]) -> np.ndarray:
    import rag  # imported lazily: loads the sentence-transformer model only when a document is actually processed
    return rag.embed_texts(texts)


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype="float32").reshape(-1).tobytes()


def _unblob(b: Any) -> np.ndarray:
    return np.frombuffer(bytes(b), dtype="float32")


# ---------------------------------------------------------------------------------------------- the store
class DocStore:
    """In-memory search structures for a set of one tutor's documents: chunks + metadata, FAISS index, BM25 index."""

    def __init__(self, tutor_id: str, chunks: list[dict[str, Any]], vectors: np.ndarray, token_lists: list[list[str]]):
        self.tutor_id = tutor_id
        self.chunks = chunks
        self.index = make_index(vectors) if len(chunks) else None
        self.keywords = KeywordIndex(token_lists)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 8, mode: str = "hybrid", embed: Embedder | None = None) -> list[dict[str, Any]]:
        """Best passages for ``query``. mode: hybrid (semantic + keyword), semantic, or keyword."""
        if not self.chunks or not (query or "").strip():
            return []
        mode = mode if mode in MODES else "hybrid"
        top_k = max(1, min(top_k, len(self.chunks)))
        pool = min(len(self.chunks), max(top_k * 4, 24))
        sem: dict[int, float] = {}
        if mode in ("hybrid", "semantic") and self.index is not None:
            q = (embed or default_embedder)([query])
            dist, idx = self.index.search(np.asarray(q, dtype="float32"), pool)
            sem = {int(i): float(d) for d, i in zip(dist[0], idx[0]) if int(i) >= 0}
        kw_all = self.keywords.scores(tokenize(query)) if mode in ("hybrid", "keyword") else {}
        kw_top = dict(sorted(kw_all.items(), key=lambda kv: -kv[1])[:pool])
        cand = set(sem) | set(kw_top)
        if not cand:
            return []
        kmax = max(kw_top.values()) if kw_top else 0.0
        out = []
        for i in cand:
            s = max(0.0, sem.get(i, 0.0))
            k = (kw_all.get(i, 0.0) / kmax) if kmax else 0.0
            score = s if mode == "semantic" else k if mode == "keyword" else W_SEMANTIC * s + W_KEYWORD * k
            c = dict(self.chunks[i])
            c.update(semantic_score=round(s, 4), keyword_score=round(k, 4), hybrid_score=round(score, 4))
            out.append(c)
        out.sort(key=lambda c: (-c["hybrid_score"], c["chunk_id"]))
        return out[:top_k]


_CACHE: dict[tuple, tuple[tuple, DocStore]] = {}
_CACHE_LOCK = threading.Lock()


def _in_clause(n: int) -> str:
    return ",".join("?" * n)


def _signature(tutor_id: str, doc_ids: tuple[int, ...]) -> tuple:
    with connect() as con:
        r = con.execute(f"SELECT COUNT(*), COALESCE(MAX(id),0), COALESCE(SUM(id),0) FROM tutor_doc_chunks WHERE tutor_id=? AND doc_id IN ({_in_clause(len(doc_ids))})",
                        (tutor_id, *doc_ids)).fetchone()
    return (int(r[0]), int(r[1]), int(r[2]))


def load_store(tutor_id: str, doc_ids: list[int]) -> DocStore:
    """Build (or fetch from the per-process cache) the search structures for the given documents. Other tutors' documents are never read."""
    ids = tuple(sorted({int(i) for i in doc_ids}))
    if not ids:
        return DocStore(tutor_id, [], np.empty((0, 4), dtype="float32"), [])
    key = (db_identity(), tutor_id, ids)
    sig = _signature(tutor_id, ids)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    with connect() as con:
        rows = [dict(r) for r in con.execute(
            f"SELECT c.id, c.doc_id, c.chunk_id, c.filename, c.page, c.text, c.tokens, c.embedding, d.title, d.subject "
            f"FROM tutor_doc_chunks c JOIN tutor_documents d ON d.id=c.doc_id "
            f"WHERE c.tutor_id=? AND d.tutor_id=? AND c.doc_id IN ({_in_clause(len(ids))}) ORDER BY c.doc_id, c.chunk_id", (tutor_id, tutor_id, *ids)).fetchall()]
    chunks, vecs, toks = [], [], []
    for n, r in enumerate(rows):
        chunks.append({"chunk_id": n, "doc_chunk_id": int(r["chunk_id"]), "doc_id": int(r["doc_id"]), "tutor_id": tutor_id, "title": r["title"], "subject": r["subject"],
                       "filename": r["filename"], "page": r["page"], "text": r["text"]})
        vecs.append(_unblob(r["embedding"]))
        toks.append((r["tokens"] or "").split())
    store = DocStore(tutor_id, chunks, np.vstack(vecs) if vecs else np.empty((0, 4), dtype="float32"), toks)
    with _CACHE_LOCK:
        if len(_CACHE) > 20:
            _CACHE.clear()
        _CACHE[key] = (sig, store)
    return store


# ---------------------------------------------------------------------------------------------- library (database)
@cached_read(30)
def list_documents(tutor_id: str) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT id, title, subject, filename, pages, characters, chunk_count, created_at FROM tutor_documents WHERE tutor_id=? ORDER BY id DESC", (tutor_id,)).fetchall()
    return [dict(r) for r in rows]


def library_usage(tutor_id: str) -> dict[str, int]:
    docs = list_documents(tutor_id)
    return {"documents": len(docs), "chunks": sum(int(d["chunk_count"] or 0) for d in docs),
            "max_documents": MAX_DOCS_PER_TUTOR, "max_chunks": MAX_CHUNKS_PER_TUTOR}


def delete_document(tutor_id: str, doc_id: int) -> bool:
    """Remove a document and all its chunks. Only the owner can do this."""
    with connect() as con:
        if not con.execute("SELECT 1 FROM tutor_documents WHERE id=? AND tutor_id=?", (int(doc_id), tutor_id)).fetchone():
            return False
        con.execute("DELETE FROM tutor_doc_chunks WHERE doc_id=? AND tutor_id=?", (int(doc_id), tutor_id))
        con.execute("DELETE FROM tutor_documents WHERE id=? AND tutor_id=?", (int(doc_id), tutor_id))
    return True


def chunk_preview(tutor_id: str, doc_id: int, limit: int = 5) -> list[dict[str, Any]]:
    """First few chunks of an owned document with their keywords: lets the tutor see what extraction and tokenising produced."""
    with connect() as con:
        rows = con.execute("SELECT c.chunk_id, c.page, c.text, c.tokens FROM tutor_doc_chunks c WHERE c.doc_id=? AND c.tutor_id=? ORDER BY c.chunk_id LIMIT ?",
                           (int(doc_id), tutor_id, int(limit))).fetchall()
    out = []
    for r in rows:
        top = [w for w, _ in Counter((r["tokens"] or "").split()).most_common(8)]
        out.append({"chunk": int(r["chunk_id"]), "page": r["page"], "keywords": ", ".join(top), "text": r["text"]})
    return out


def ingest(tutor_id: str, files: list[tuple[str, bytes]], subject: str = "", embed: Embedder | None = None,
           progress: Progress | None = None) -> list[dict[str, Any]]:
    """Process each file once and store it. Returns one result per file: status ``added`` / ``exists`` / ``error`` and a message.

    Each file is independent: a damaged file reports an error and the others are still saved. A file already in this tutor's
    library (same content) is not processed again.
    """
    say = progress or (lambda *_: None)
    embed = embed or default_embedder
    if not files:
        return []
    if len(files) > doc_quiz.MAX_FILES:
        raise ValueError(f"Upload at most {doc_quiz.MAX_FILES} documents at a time.")
    results: list[dict[str, Any]] = []
    for n, (name, data) in enumerate(files):
        base = n / len(files)
        span = 1 / len(files)
        name = re.sub(r"[\\/]+", "/", name).split("/")[-1] or "document"
        res: dict[str, Any] = {"filename": name, "status": "error", "doc_id": None, "pages": 0, "chunks": 0, "message": ""}
        results.append(res)
        digest = hashlib.sha256(data).hexdigest()
        with connect() as con:
            have = con.execute("SELECT id, pages, chunk_count FROM tutor_documents WHERE tutor_id=? AND sha256=?", (tutor_id, digest)).fetchone()
        if have:
            res.update(status="exists", doc_id=int(have["id"]), pages=int(have["pages"] or 0), chunks=int(have["chunk_count"] or 0), message="Already in your library; nothing to process again.")
            continue
        usage = library_usage(tutor_id)
        if usage["documents"] >= MAX_DOCS_PER_TUTOR:
            res["message"] = f"Your library is full ({MAX_DOCS_PER_TUTOR} documents). Delete one you no longer need."
            continue
        try:
            say(f"Reading {name}", base + 0.05 * span)
            prepared = doc_quiz.prepare([(name, data)])          # extract text with page numbers, validate, chunk with overlap
        except ValueError as exc:
            res["message"] = str(exc)
            continue
        chunks = prepared["chunks"]
        if usage["chunks"] + len(chunks) > MAX_CHUNKS_PER_TUTOR:
            res["message"] = f"This would exceed your library limit ({MAX_CHUNKS_PER_TUTOR} sections in total). Delete a document first."
            continue
        say(f"Indexing keywords in {name}", base + 0.2 * span)
        tokens = [tokenize(c["text"]) for c in chunks]
        vectors = []
        for s in range(0, len(chunks), EMBED_BATCH):
            say(f"Creating embeddings for {name} ({min(s + EMBED_BATCH, len(chunks))}/{len(chunks)})", base + (0.2 + 0.7 * s / len(chunks)) * span)
            try:
                vectors.append(np.asarray(embed([c["text"] for c in chunks[s:s + EMBED_BATCH]]), dtype="float32"))
            except Exception as exc:  # noqa: BLE001 - model download/memory problems
                res["message"] = f"Embeddings could not be created ({type(exc).__name__}). Try again in a minute."
                vectors = []
                break
        if not vectors:
            continue
        vec = np.vstack(vectors)
        say(f"Saving {name}", base + 0.95 * span)
        info = prepared["info"][0]
        title = re.sub(r"\.[A-Za-z0-9]+$", "", name)[:120] or name
        with connect() as con:
            doc_id = con.insert(
                "INSERT INTO tutor_documents(tutor_id,title,subject,filename,sha256,pages,characters,chunk_count,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (tutor_id, title, subject or "", name, digest, int(info["pages"]), int(info["characters"]), len(chunks), datetime.utcnow().isoformat()))
            con.executemany(
                "INSERT INTO tutor_doc_chunks(doc_id,tutor_id,chunk_id,filename,page,text,tokens,dim,embedding) VALUES(?,?,?,?,?,?,?,?,?)",
                [(doc_id, tutor_id, int(c["chunk_id"]), name, c.get("page"), c["text"], " ".join(t), int(vec.shape[1]), _blob(vec[i]))
                 for i, (c, t) in enumerate(zip(chunks, tokens))])
        res.update(status="added", doc_id=int(doc_id), pages=int(info["pages"]), chunks=len(chunks),
                   message="; ".join(prepared["warnings"]) or f"Saved: {info['pages']} page(s), {len(chunks)} searchable sections.")
    say("Done", 1.0)
    return results


# ---------------------------------------------------------------------------------------------- quiz planning
def plan_quiz(store: DocStore, count: int, topic: str = "", mode: str = "hybrid", embed: Embedder | None = None) -> list[dict[str, Any]]:
    """Rounds of passages for the quiz writer. With a topic/chapter the rounds draw on the best search hits; otherwise on the whole document."""
    return doc_quiz.plan_rounds(store.chunks, count, topic, lambda q, k: store.search(q, k, mode, embed))
