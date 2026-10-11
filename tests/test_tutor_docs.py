"""Tutor document library: extract -> chunk -> keywords -> embeddings -> stored vector index -> hybrid/keyword search -> quiz plan."""
import hashlib, io, pathlib, re, sys, tempfile, types, unittest
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
class _St(types.ModuleType):
    session_state = {}; secrets = {}
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
sys.modules["streamlit"] = _St("streamlit")
from unittest import mock as _mock; sys.modules.setdefault("gdown", _mock.MagicMock())
import numpy as np
import config; config.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "t.db"
import db, db_core, doc_quiz, migrations, tutor_docs as td  # noqa: E402
db.init_db()

DIM = 128
def embed(texts):
    """Deterministic stand-in for the sentence-transformer: hashed bag of words, L2-normalised (same shape/behaviour as the real one)."""
    out = np.zeros((len(texts), DIM), dtype="float32")
    for r, t in enumerate(texts):
        for w in td.tokenize(t):
            out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        n = np.linalg.norm(out[r]); out[r] /= n or 1
    return out

BIO = ("Mitochondria produce ATP through oxidative phosphorylation inside the cell. " * 14 + "\n\n" +
       "Ribosomes translate messenger RNA into proteins using amino acids. " * 14).encode()
CHEM = ("The molar mass of sodium chloride NaCl is 58.44 grams per mole. " * 14 + "\n\n" +
        "Le Chatelier principle describes how equilibrium shifts under stress. " * 14).encode()
def pdf(pages):
    from reportlab.pdfgen import canvas
    b = io.BytesIO(); c = canvas.Canvas(b)
    for text in pages:
        y = 800
        for k in range(0, len(text), 90): c.drawString(40, y, text[k:k + 90]); y -= 14
        c.showPage()
    c.save(); return b.getvalue()

class Tokens(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(td.tokenize("enzymes"), td.tokenize("enzyme")); self.assertEqual(td.tokenize("classes"), td.tokenize("class")); self.assertEqual(td.tokenize("mitochondrial"), ["mitochondrial"])
        t = td.tokenize("The Enzymes of the cells, and ATP-synthase 58.44")
        self.assertIn("enzyme", t); self.assertIn("atp-synthase", t); self.assertNotIn("the", t); self.assertNotIn("of", t)
        self.assertEqual(td.tokenize("cell"), td.tokenize("cells"))
    def test_bm25_prefers_rare_exact_terms(self):
        ix = td.KeywordIndex([td.tokenize(s) for s in ["cell membrane cell cell", "ribosome protein", "cell nucleus"]])
        s = ix.scores(td.tokenize("ribosome")); self.assertEqual(list(s), [1])
        s = ix.scores(td.tokenize("cell nucleus")); self.assertEqual(max(s, key=s.get), 2)
        self.assertEqual(ix.scores(["zzz"]), {})

class Library(unittest.TestCase):
    n = [0]
    def setUp(self):
        Library.n[0] += 1; k = Library.n[0]
        self.T, self.T2 = f"docT{k}", f"docT2_{k}"
        for s in (self.T, self.T2): db.ensure_student(s, s); db.set_role(s, "tutor")

    def test_migration_created_tables(self):
        self.assertGreaterEqual(migrations.current_version(), 10)
        with db.connect() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM tutor_documents").fetchone()[0] >= 0, True)

    def test_ingest_stores_chunks_metadata_keywords_and_vectors(self):
        stages = []
        r = td.ingest(self.T, [("bio.txt", BIO), ("chem.pdf", pdf(["Acids donate protons. " * 30, "Bases accept protons. " * 30]))], "Biology", embed, lambda m, f: stages.append((m, f)))
        self.assertEqual([x["status"] for x in r], ["added", "added"]); self.assertTrue(all(x["chunks"] >= 2 for x in r))
        self.assertTrue(any("embeddings" in m for m, _ in stages)); self.assertEqual(stages[-1][1], 1.0)
        self.assertEqual([f for _, f in stages], sorted(f for _, f in stages))
        store = td.load_store(self.T, [x["doc_id"] for x in r])
        self.assertEqual(len(store), sum(x["chunks"] for x in r)); self.assertEqual(store.index.ntotal, len(store))
        c = store.chunks[0]
        for k in ("doc_id", "filename", "page", "chunk_id", "doc_chunk_id", "tutor_id", "title", "text"): self.assertIn(k, c)
        self.assertEqual(c["tutor_id"], self.T)
        pages = {c["page"] for c in store.chunks if c["filename"] == "chem.pdf"}; self.assertEqual(pages, {1, 2})   # PDF page numbers survive
        with db.connect() as con:
            row = con.execute("SELECT tokens, dim, embedding FROM tutor_doc_chunks WHERE tutor_id=? ORDER BY id LIMIT 1", (self.T,)).fetchone()
        self.assertIn("mitochondria", row["tokens"]); self.assertEqual(row["dim"], DIM); self.assertEqual(len(bytes(row["embedding"])), DIM * 4)
        ov = [c["text"] for c in store.chunks if c["filename"] == "bio.txt"]
        self.assertTrue(len(ov) >= 2 and ov[0][-40:] in ov[1][:300])           # overlapping chunks

    def test_survives_restart_without_reembedding(self):
        r = td.ingest(self.T, [("bio.txt", BIO)], "Biology", embed)
        td._CACHE.clear(); db_core.clear_read_cache()
        def boom(_): raise AssertionError("must not embed documents again after a restart")
        store = td.load_store(self.T, [r[0]["doc_id"]])
        hits = store.search("ribosomes translate RNA", 3, "hybrid", embed)
        self.assertIn("Ribosomes", hits[0]["text"])
        self.assertEqual(td.list_documents(self.T)[0]["chunk_count"], r[0]["chunks"])
        again = td.ingest(self.T, [("renamed.txt", BIO)], "Biology", boom)     # same content: recognised, nothing processed
        self.assertEqual(again[0]["status"], "exists"); self.assertEqual(again[0]["doc_id"], r[0]["doc_id"]); self.assertEqual(len(td.list_documents(self.T)), 1)

    def test_search_modes(self):
        r = td.ingest(self.T, [("bio.txt", BIO), ("chem.txt", CHEM)], "", embed)
        store = td.load_store(self.T, [x["doc_id"] for x in r])
        h = store.search("molar mass of NaCl", 3, "hybrid", embed); self.assertEqual(h[0]["filename"], "chem.txt")
        self.assertTrue(0 <= h[0]["semantic_score"] <= 1 and 0 <= h[0]["keyword_score"] <= 1)
        self.assertEqual(h, sorted(h, key=lambda c: -c["hybrid_score"]))
        k = store.search("Chatelier", 3, "keyword"); self.assertTrue(k and all("Chatelier" in c["text"] for c in k))   # exact rare word, no embedding needed
        self.assertEqual(store.search("quantum chromodynamics", 3, "keyword"), [])
        s = store.search("proteins from amino acids", 2, "semantic", embed); self.assertEqual(s[0]["filename"], "bio.txt")
        self.assertEqual(store.search("", 3, "hybrid", embed), [])
        self.assertLessEqual(len(store.search("cell", 100, "hybrid", embed)), len(store))
        sub = td.load_store(self.T, [r[0]["doc_id"]]); self.assertTrue(all(c["doc_id"] == r[0]["doc_id"] for c in sub.search("cell", 10, "hybrid", embed)))

    def test_tutors_cannot_see_or_delete_each_others_documents(self):
        a = td.ingest(self.T, [("bio.txt", BIO)], "", embed)[0]["doc_id"]
        self.assertEqual(td.list_documents(self.T2), [])
        self.assertEqual(len(td.load_store(self.T2, [a])), 0)                  # asking for another tutor's id yields nothing
        self.assertFalse(td.delete_document(self.T2, a)); self.assertEqual(len(td.list_documents(self.T)), 1)
        self.assertEqual(td.chunk_preview(self.T2, a), [])
        b = td.ingest(self.T2, [("bio.txt", BIO)], "", embed)[0]; self.assertEqual(b["status"], "added")   # same file, different tutor: own copy
        self.assertTrue(td.delete_document(self.T, a)); self.assertEqual(td.list_documents(self.T), [])
        self.assertEqual(len(td.load_store(self.T2, [b["doc_id"]])) > 0, True)
        with db.connect() as con: self.assertEqual(con.execute("SELECT COUNT(*) FROM tutor_doc_chunks WHERE doc_id=?", (a,)).fetchone()[0], 0)

    def test_bad_files_are_reported_and_good_ones_still_saved(self):
        r = td.ingest(self.T, [("bad.pdf", b"not really a pdf"), ("tiny.txt", b"hi"), ("notes.exe", b"MZ" * 200), ("bio.txt", BIO)], "", embed)
        self.assertEqual([x["status"] for x in r], ["error", "error", "error", "added"])
        self.assertIn("could not be read", r[0]["message"]); self.assertIn("almost no readable text", r[1]["message"]); self.assertIn("Unsupported", r[2]["message"])
        self.assertEqual(len(td.list_documents(self.T)), 1)
        with self.assertRaises(ValueError): td.ingest(self.T, [("a%d.txt" % i, BIO + bytes([i])) for i in range(doc_quiz.MAX_FILES + 1)], "", embed)
        def down(_): raise RuntimeError("model download failed")
        r = td.ingest(self.T, [("chem.txt", CHEM)], "", down); self.assertEqual(r[0]["status"], "error"); self.assertIn("Embeddings could not be created", r[0]["message"])
        self.assertEqual(len(td.list_documents(self.T)), 1)                     # nothing half-saved

    def test_library_limits(self):
        old = td.MAX_DOCS_PER_TUTOR, td.MAX_CHUNKS_PER_TUTOR
        try:
            td.MAX_DOCS_PER_TUTOR = 1
            td.ingest(self.T, [("bio.txt", BIO)], "", embed)
            r = td.ingest(self.T, [("chem.txt", CHEM)], "", embed); self.assertIn("library is full", r[0]["message"])
            td.MAX_DOCS_PER_TUTOR = 30; td.MAX_CHUNKS_PER_TUTOR = 3
            r = td.ingest(self.T, [("chem.txt", CHEM)], "", embed); self.assertIn("library limit", r[0]["message"])
        finally:
            td.MAX_DOCS_PER_TUTOR, td.MAX_CHUNKS_PER_TUTOR = old

    def test_quiz_plan_covers_the_document_and_follows_the_topic(self):
        long_text = " ".join(f"Section {i} discusses topic{i} in detail with many words about biology and cells." for i in range(120)).encode()
        r = td.ingest(self.T, [("long.txt", long_text)], "", embed)
        store = td.load_store(self.T, [r[0]["doc_id"]])
        plan = td.plan_quiz(store, 20, "", "hybrid", embed); self.assertEqual(sum(p["n"] for p in plan), 20)
        used = {c["chunk_id"] for p in plan for c in p["chunks"]}; self.assertGreater(len(used), 1)
        plan = td.plan_quiz(store, 10, "topic7", "keyword", embed); self.assertIn("topic7", plan[0]["context"])
        self.assertIn("long.txt | page N/A", plan[0]["context"])
        qs, src = doc_quiz.generate(plan, lambda n, ctx: [{"question": f"Q about {ctx[:30]} {i}?", "options": ["a", "b", "c", "d"], "answer": 0} for i in range(n)], 10)
        self.assertEqual(len(qs), 10)

    def test_chunk_preview_and_usage(self):
        r = td.ingest(self.T, [("bio.txt", BIO)], "", embed)
        pv = td.chunk_preview(self.T, r[0]["doc_id"], 2); self.assertEqual(len(pv), 2); self.assertIn("mitochondria", pv[0]["keywords"])
        u = td.library_usage(self.T); self.assertEqual((u["documents"], u["chunks"]), (1, r[0]["chunks"]))
        self.assertIn(td.index_kind().split()[0], ("FAISS", "numpy"))

if __name__ == "__main__": unittest.main()
