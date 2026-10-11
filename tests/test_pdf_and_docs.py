"""PDF assignments (upload before deadline, marking, access control) and quizzes from a tutor's own documents."""
import io, pathlib, sys, tempfile, types, unittest, zipfile
from datetime import datetime, timedelta
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
class _St(types.ModuleType):
    session_state = {}; secrets = {}
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
sys.modules["streamlit"] = _St("streamlit")
from unittest import mock as _mock; sys.modules.setdefault("gdown", _mock.MagicMock())
import config; config.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "p.db"
import classes as cl, db, doc_quiz, pdf_assignments as pa  # noqa: E402
db.init_db()
N = datetime.utcnow()
def iso(delta): return (N + delta).isoformat(timespec="minutes")

def make_pdf(pages=1, text="Photosynthesis happens in the chloroplast. " * 20):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); c = canvas.Canvas(buf)
    for i in range(pages):
        y = 800
        for k in range(0, len(text), 90):
            c.drawString(40, y, text[k:k + 90]); y -= 14
        c.showPage()
    c.save(); return buf.getvalue()
PDF = make_pdf()

def mk(sid, role="student"):
    db.ensure_student(sid, sid)
    if role == "tutor": db.set_role(sid, "tutor")

class Pdf(unittest.TestCase):
    n = [0]
    def setUp(self):
        Pdf.n[0] += 1; k = Pdf.n[0]
        self.T, self.T2, self.A, self.B, self.C = f"pT{k}", f"pT2_{k}", f"pa{k}", f"pb{k}", f"pc{k}"
        for s in (self.T, self.T2): mk(s, "tutor")
        for s in (self.A, self.B, self.C): mk(s)
        self.cls = cl.create_class(self.T, "Batch"); self.cid = self.cls["id"]
        for s in (self.A, self.B): cl.join_class(s, self.cls["join_code"])
        ok, msg = pa.create_pdf_assignment(self.T, self.cid, "Homework 1", "Answer all", "../../hw 1.pdf", PDF, iso(timedelta(days=2)), False, 20)
        self.assertTrue(ok, msg)
        with db.connect() as con: self.aid = con.execute("SELECT MAX(id) FROM assignments").fetchone()[0]

    def test_validation(self):
        self.assertFalse(pa.validate_pdf(b"")[0]); self.assertIn("not a PDF", pa.validate_pdf(b"hello world")[1])
        self.assertIn("could not be opened", pa.validate_pdf(b"%PDF-1.4 garbage")[1])
        self.assertIn("larger than", pa.validate_pdf(b"%PDF-" + b"0" * (pa.MAX_BYTES + 1))[1])
        ok, _, pages = pa.validate_pdf(make_pdf(3)); self.assertTrue(ok); self.assertEqual(pages, 3)
        self.assertEqual(pa.sanitize_filename("../../My Work (final).PDF"), "My Work _final.pdf"); self.assertEqual(pa.sanitize_filename(""), "document.pdf")

    def test_assignment_appears_for_class_members_only(self):
        a = [x for x in cl.student_assignments(self.A) if x["kind"] == "pdf"]
        self.assertEqual(len(a), 1); self.assertIn(a[0]["status"], ("open", "due_soon")); self.assertEqual(a[0]["max_marks"], 20); self.assertTrue(a[0]["is_open"])
        self.assertEqual([x for x in cl.student_assignments(self.C) if x["kind"] == "pdf"], [])
        self.assertEqual(cl.pending_count(self.A) >= 1, True)

    def test_upload_in_time_replace_and_status(self):
        ok, msg, info = pa.submit_solution(self.A, self.aid, "my solution.pdf", PDF, "see page 2")
        self.assertTrue(ok, msg); self.assertFalse(info["late"])
        self.assertEqual([x for x in cl.student_assignments(self.A) if x["kind"] == "pdf"][0]["status"], "done")
        ok, _, info = pa.submit_solution(self.A, self.aid, "v2.pdf", make_pdf(2)); self.assertTrue(ok); self.assertTrue(info["replaced"])
        with db.connect() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM assignment_submissions WHERE assignment_id=?", (self.aid,)).fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM stored_files WHERE owner_id='"+self.A+"'").fetchone()[0], 1)      # the old file was removed
        self.assertEqual(pa.my_submission(self.A, self.aid)["filename"], "v2.pdf")

    def test_deadline_is_enforced_on_the_server(self):
        late_time = N + timedelta(days=3)
        ok, msg, info = pa.submit_solution(self.A, self.aid, "x.pdf", PDF, now=late_time)
        self.assertFalse(ok); self.assertTrue(info["blocked"]); self.assertIn("closed", msg)
        self.assertEqual([x for x in cl.student_assignments(self.A, now=late_time) if x["kind"] == "pdf"][0]["status"], "missed")
        cl.update_deadline(self.T, self.aid, iso(timedelta(days=2)), True)                                                # tutor allows late work
        ok, msg, info = pa.submit_solution(self.A, self.aid, "x.pdf", PDF, now=late_time)
        self.assertTrue(ok); self.assertTrue(info["late"]); self.assertIn("late", msg)
        self.assertEqual([x for x in cl.student_assignments(self.A, now=late_time) if x["kind"] == "pdf"][0]["status"], "done_late")

    def test_no_deadline_never_blocks(self):
        ok, _ = pa.create_pdf_assignment(self.T, self.cid, "Open", "", "o.pdf", PDF, None, False, None); self.assertTrue(ok)
        with db.connect() as con: aid = con.execute("SELECT MAX(id) FROM assignments").fetchone()[0]
        self.assertTrue(pa.submit_solution(self.A, aid, "o.pdf", PDF, now=N + timedelta(days=900))[0])

    def test_only_members_can_upload_and_bad_files_are_refused(self):
        self.assertFalse(pa.submit_solution(self.C, self.aid, "x.pdf", PDF)[0])
        self.assertFalse(pa.submit_solution(self.A, self.aid, "x.pdf", b"not a pdf")[0])
        self.assertFalse(pa.submit_solution(self.A, 99999, "x.pdf", PDF)[0])
        self.assertIsNone(pa.my_submission(self.A, self.aid))

    def test_marking_locks_then_reopen(self):
        pa.submit_solution(self.A, self.aid, "s.pdf", PDF)
        sid = pa.my_submission(self.A, self.aid)["id"]
        self.assertFalse(pa.grade(self.T2, sid, 10, "")[0])                       # another tutor cannot mark it
        self.assertFalse(pa.grade(self.T, sid, 25, "")[0]); self.assertFalse(pa.grade(self.T, sid, -1, "")[0])   # above the total / negative
        self.assertTrue(pa.grade(self.T, sid, 17.5, "Good work")[0])
        row = [r for r in cl.assignment_report(self.T, self.aid)["rows"] if r["student_id"] == self.A][0]
        self.assertEqual((row["score"], row["feedback"], row["graded"]), (17.5, "Good work", True))
        rep = cl.assignment_report(self.T, self.aid); self.assertEqual(rep["average"], 17.5); self.assertEqual(rep["completed"], 1); self.assertEqual([m["student_id"] for m in rep["missing"]], [self.B])
        ok, msg, info = pa.submit_solution(self.A, self.aid, "again.pdf", PDF); self.assertFalse(ok); self.assertTrue(info["locked"])
        self.assertTrue(pa.reopen(self.T, sid)); self.assertTrue(pa.submit_solution(self.A, self.aid, "again.pdf", PDF)[0])

    def test_downloads_respect_access(self):
        pa.submit_solution(self.A, self.aid, "s.pdf", PDF); sid = pa.my_submission(self.A, self.aid)["id"]
        self.assertEqual(pa.get_assignment_file(self.A, self.aid)[1], PDF); self.assertEqual(pa.get_assignment_file(self.T, self.aid)[1], PDF)
        self.assertIsNone(pa.get_assignment_file(self.C, self.aid)); self.assertIsNone(pa.get_assignment_file(self.T2, self.aid))
        self.assertEqual(pa.get_submission_file(self.A, sid)[1], PDF); self.assertEqual(pa.get_submission_file(self.T, sid)[1], PDF)
        self.assertIsNone(pa.get_submission_file(self.B, sid)); self.assertIsNone(pa.get_submission_file(self.T2, sid))

    def test_zip_of_all_solutions_and_storage_and_delete(self):
        pa.submit_solution(self.A, self.aid, "a.pdf", PDF); pa.submit_solution(self.B, self.aid, "b.pdf", make_pdf(2))
        z = zipfile.ZipFile(io.BytesIO(pa.submissions_zip(self.T, self.aid))); self.assertEqual(len(z.namelist()), 2)
        self.assertTrue(all(n.endswith(".pdf") and "/" not in n for n in z.namelist()))
        self.assertIsNone(pa.submissions_zip(self.T2, self.aid))
        self.assertGreater(pa.storage_used_mb(self.T), 0)
        self.assertFalse(cl.delete_assignment(self.T2, self.aid)); self.assertTrue(cl.delete_assignment(self.T, self.aid))
        with db.connect() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM assignment_submissions WHERE assignment_id=?", (self.aid,)).fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM stored_files WHERE owner_id IN ('"+self.A+"','"+self.B+"')").fetchone()[0], 0)

    def test_creation_rules(self):
        self.assertFalse(pa.create_pdf_assignment(self.T, self.cid, "x", "", "x.pdf", b"nope", None)[0])
        self.assertFalse(pa.create_pdf_assignment(self.T, self.cid, "x", "", "x.pdf", PDF, iso(timedelta(days=-1)))[0])
        self.assertFalse(pa.create_pdf_assignment(self.T2, self.cid, "x", "", "x.pdf", PDF, None)[0])               # not their class
        self.assertFalse(pa.create_pdf_assignment(self.T, self.cid, "x", "", "x.pdf", PDF, None, False, 5000)[0])

    def test_quiz_assignments_still_work_next_to_pdf_ones(self):
        code = db.create_shared_quiz(self.T, "Q", "Biology", "Cells", "Medium", [{"question": "Which organelle makes ATP?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "A", "concept": "c", "explanation": "e"}], 0)
        self.assertTrue(cl.create_assignment(self.T, self.cid, code, "", iso(timedelta(days=1)))[0])
        kinds = sorted(x["kind"] for x in cl.student_assignments(self.A)); self.assertIn("pdf", kinds); self.assertIn("quiz", kinds)
        self.assertEqual(sorted({r["kind"] for r in cl.list_assignments(self.T, self.cid)}), ["pdf", "quiz"])


class DocQuiz(unittest.TestCase):
    TEXT = " ".join(f"Section {i}. The topic number {i} explains concept {i} in detail with facts about item {i} and its properties." for i in range(1, 160))

    def test_prepare_reads_pdf_txt_and_rejects_bad_input(self):
        out = doc_quiz.prepare([("notes.pdf", make_pdf(2, self.TEXT[:900])), ("more.txt", self.TEXT.encode())])
        self.assertEqual([i["filename"] for i in out["info"]], ["notes.pdf", "more.txt"]); self.assertGreater(len(out["chunks"]), 5)
        for bad, msg in ((([]), "at least one"), ([("a.exe", b"x")], "Unsupported"), ([("a.txt", b"hi")], "almost no readable text"),
                         ([("a.txt", b"x" * (doc_quiz.MAX_FILE_MB * 1024 * 1024 + 1))], "larger than"), ([("a.pdf", b"%PDF-1.4 junk")], "could not be read")):
            with self.assertRaises(ValueError) as cm: doc_quiz.prepare(bad)
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(ValueError): doc_quiz.prepare([(f"f{i}.txt", self.TEXT.encode()) for i in range(doc_quiz.MAX_FILES + 1)])

    def test_rounds_cover_the_whole_document(self):
        chunks = doc_quiz.prepare([("book.txt", (self.TEXT * 6).encode())])["chunks"]
        plan = doc_quiz.plan_rounds(chunks, 25)
        self.assertEqual([r["n"] for r in plan], [9, 8, 8]); self.assertEqual(sum(r["n"] for r in plan), 25)
        firsts = [r["chunks"][0]["chunk_id"] for r in plan]; self.assertEqual(firsts, sorted(firsts)); self.assertEqual(len(set(firsts)), 3)   # different parts of the file
        self.assertTrue(all(len(r["context"]) <= doc_quiz.CONTEXT_CHARS + 1200 for r in plan))

    def test_topic_search_gives_each_round_different_passages(self):
        chunks = doc_quiz.prepare([("book.txt", (self.TEXT * 6).encode())])["chunks"]
        seen_queries = []
        def search(q, k): seen_queries.append((q, k)); return chunks[:k]
        plan = doc_quiz.plan_rounds(chunks, 20, "concept 7", search)
        self.assertEqual(seen_queries[0][0], "concept 7"); self.assertEqual(len(plan), 2)
        self.assertTrue(set(c["chunk_id"] for c in plan[0]["chunks"]).isdisjoint(c["chunk_id"] for c in plan[1]["chunks"]))

    def test_generate_dedupes_caps_and_survives_a_failed_round(self):
        chunks = doc_quiz.prepare([("book.txt", (self.TEXT * 6).encode())])["chunks"]; plan = doc_quiz.plan_rounds(chunks, 20)
        calls = []
        def make(n, ctx):
            calls.append(n)
            if len(calls) == 1: raise RuntimeError("rate limit")
            return [{"question": f"Question {len(calls)}-{i} about the text?"} for i in range(n)] + [{"question": "Question 2-0 about the text?"}]
        qs, sources = doc_quiz.generate(plan, make, 20)
        self.assertEqual(len(qs), len({q["question"] for q in qs})); self.assertLessEqual(len(qs), 20); self.assertTrue(sources)
        with self.assertRaises(RuntimeError): doc_quiz.generate(plan, lambda n, c: 1 / 0, 20)

    def test_fingerprint_changes_with_content(self):
        a = doc_quiz.fingerprint([("a.txt", b"one")]); self.assertEqual(a, doc_quiz.fingerprint([("a.txt", b"one")])); self.assertNotEqual(a, doc_quiz.fingerprint([("a.txt", b"two")]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
