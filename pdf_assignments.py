"""PDF assignments: a tutor assigns a PDF to a class; students upload their solution (PDF) before the deadline; the tutor marks it.

Deadline rule: a solution can be uploaded (or replaced) until the due time. After it, uploads are refused unless the assignment allows
late work, in which case the upload is flagged late. A solution that has been marked is locked until the tutor re-opens it.
Files are stored in the database (table stored_files), so they survive redeploys on any host. Pure logic (no Streamlit).
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from typing import Any

import classes
import config
from db_core import connect

MAX_BYTES = config.ASSIGNMENT_MAX_MB * 1024 * 1024


def sanitize_filename(name: str, fallback: str = "document") -> str:
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", (name or "").replace("\\", "/").split("/")[-1]).strip(" .")
    base = re.sub(r"\.pdf$", "", base, flags=re.I)[:80].strip(" ._-") or fallback
    return base + ".pdf"


def validate_pdf(data: bytes) -> tuple[bool, str, int]:
    """(ok, message, page count). Checks size, the PDF signature, that it opens, is not password-protected and is not huge."""
    if not data:
        return False, "The file is empty.", 0
    if len(data) > MAX_BYTES:
        return False, f"The file is larger than {config.ASSIGNMENT_MAX_MB} MB. Compress it or split it into parts.", 0
    if not data.lstrip()[:5] == b"%PDF-":
        return False, "This is not a PDF file. Export or scan your work as a PDF and upload that.", 0
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return False, "This PDF is password-protected. Remove the password and upload it again.", 0
        pages = len(reader.pages)
    except Exception:  # noqa: BLE001 - damaged files
        return False, "This PDF could not be opened. It may be damaged; export it again.", 0
    if pages < 1:
        return False, "The PDF has no pages.", 0
    if pages > config.ASSIGNMENT_MAX_PAGES:
        return False, f"The PDF has {pages} pages; the limit is {config.ASSIGNMENT_MAX_PAGES}.", pages
    return True, "OK", pages


def _store(con: Any, owner_id: str, filename: str, data: bytes) -> int:
    import hashlib
    return con.insert("INSERT INTO stored_files(owner_id,filename,mime,size_bytes,sha256,data,created_at) VALUES(?,?,?,?,?,?,?)",
                      (owner_id, filename, "application/pdf", len(data), hashlib.sha256(data).hexdigest(), data, datetime.utcnow().isoformat()))


# ------------------------------------------------------------------ tutor
def create_pdf_assignment(tutor_id: str, class_id: int, title: str, instructions: str, filename: str, data: bytes,
                          due_at_utc: str | None, allow_late: bool = False, max_marks: int | None = None) -> tuple[bool, str]:
    ok, msg, _ = validate_pdf(data)
    if not ok:
        return False, msg
    title = " ".join((title or "").split())[:120] or sanitize_filename(filename)[:-4]
    if max_marks is not None and not (1 <= int(max_marks) <= 1000):
        return False, "Total marks must be between 1 and 1000."
    if due_at_utc and datetime.fromisoformat(due_at_utc) <= classes.now_utc():
        return False, "The deadline must be in the future."
    with connect() as con:
        if not con.execute("SELECT 1 FROM classes WHERE id=? AND tutor_id=? AND is_active=1", (class_id, tutor_id)).fetchone():
            return False, "Class not found or closed."
        fid = _store(con, tutor_id, sanitize_filename(filename), data)
        code = "PDF" + classes.secrets.token_hex(4).upper()          # assignments need a unique code; PDF ones have no quiz behind it
        con.execute("INSERT INTO assignments(class_id,tutor_id,quiz_code,title,due_at,allow_late,created_at,kind,file_id,instructions,max_marks) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (class_id, tutor_id, code, title, due_at_utc, int(allow_late), classes.now_utc().isoformat(), "pdf", fid, (instructions or "").strip()[:2000], int(max_marks) if max_marks else None))
    return True, "PDF assignment created."


def grade(tutor_id: str, submission_id: int, marks: float | None, feedback: str) -> tuple[bool, str]:
    with connect() as con:
        r = con.execute("SELECT s.id,a.max_marks FROM assignment_submissions s JOIN assignments a ON a.id=s.assignment_id WHERE s.id=? AND a.tutor_id=?", (submission_id, tutor_id)).fetchone()
        if not r:
            return False, "Submission not found."
        if marks is not None:
            if marks < 0 or (r["max_marks"] and marks > r["max_marks"]):
                return False, f"Marks must be between 0 and {r['max_marks'] or 'the maximum'}."
        con.execute("UPDATE assignment_submissions SET marks=?, feedback=?, graded_at=? WHERE id=?", (marks, (feedback or "").strip()[:2000], datetime.utcnow().isoformat() if marks is not None or feedback.strip() else None, submission_id))
    return True, "Saved."


def reopen(tutor_id: str, submission_id: int) -> bool:
    """Clear the marks so the student may upload a corrected solution (only possible while the deadline allows it)."""
    with connect() as con:
        r = con.execute("SELECT s.id FROM assignment_submissions s JOIN assignments a ON a.id=s.assignment_id WHERE s.id=? AND a.tutor_id=?", (submission_id, tutor_id)).fetchone()
        if not r:
            return False
        con.execute("UPDATE assignment_submissions SET marks=NULL, graded_at=NULL WHERE id=?", (submission_id,))
    return True


def submissions_zip(tutor_id: str, assignment_id: int) -> bytes | None:
    """All solutions of one assignment as a ZIP (one PDF per student)."""
    with connect() as con:
        a = con.execute("SELECT title FROM assignments WHERE id=? AND tutor_id=? AND kind='pdf'", (assignment_id, tutor_id)).fetchone()
        if not a:
            return None
        rows = con.execute("SELECT s.student_id, COALESCE(st.name,s.student_id) AS name, f.data FROM assignment_submissions s JOIN stored_files f ON f.id=s.file_id "
                           "LEFT JOIN students st ON st.id=s.student_id WHERE s.assignment_id=? ORDER BY name", (assignment_id,)).fetchall()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            z.writestr(f"{sanitize_filename(r['name'])[:-4]}_{sanitize_filename(r['student_id'])[:-4]}.pdf", bytes(r["data"]))
    return buf.getvalue()


# ------------------------------------------------------------------ student
def submit_solution(student_id: str, assignment_id: int, filename: str, data: bytes, note: str = "", now: datetime | None = None) -> tuple[bool, str, dict[str, Any]]:
    """Upload or replace a solution. The deadline is checked here, on the server, at the moment of upload."""
    now = now or classes.now_utc()
    with connect() as con:
        a = con.execute("SELECT a.id,a.due_at,a.allow_late,a.kind,c.is_active FROM assignments a JOIN class_members m ON m.class_id=a.class_id JOIN classes c ON c.id=a.class_id "
                        "WHERE a.id=? AND m.student_id=?", (assignment_id, student_id)).fetchone()
        if not a or (a["kind"] or "quiz") != "pdf":
            return False, "This assignment is not available to you.", {}
        due = datetime.fromisoformat(a["due_at"]) if a["due_at"] else None
        late = bool(due and now > due)
        if late and not a["allow_late"]:
            return False, f"The deadline passed on {classes.fmt_local(a['due_at'])}. Uploads are closed; ask your tutor if you need more time.", {"blocked": True}
        existing = con.execute("SELECT id,file_id,graded_at,marks FROM assignment_submissions WHERE assignment_id=? AND student_id=?", (assignment_id, student_id)).fetchone()
        if existing and (existing["marks"] is not None or existing["graded_at"]):
            return False, "Your solution has already been marked, so it can no longer be replaced. Ask your tutor to re-open it.", {"locked": True}
        ok, msg, pages = validate_pdf(data)
        if not ok:
            return False, msg, {}
        fid = _store(con, student_id, sanitize_filename(filename), data)
        stamp = now.isoformat()
        if existing:
            con.execute("UPDATE assignment_submissions SET file_id=?, filename=?, size_bytes=?, submitted_at=?, is_late=?, note=? WHERE id=?",
                        (fid, sanitize_filename(filename), len(data), stamp, int(late), (note or "").strip()[:500], existing["id"]))
            if existing["file_id"]:
                con.execute("DELETE FROM stored_files WHERE id=?", (existing["file_id"],))
        else:
            con.execute("INSERT INTO assignment_submissions(assignment_id,student_id,file_id,filename,size_bytes,submitted_at,is_late,note) VALUES(?,?,?,?,?,?,?,?)",
                        (assignment_id, student_id, fid, sanitize_filename(filename), len(data), stamp, int(late), (note or "").strip()[:500]))
    return True, ("Uploaded, but after the deadline, so it is marked late." if late else "Solution uploaded in time."), {"late": late, "pages": pages, "replaced": bool(existing)}


def my_submission(student_id: str, assignment_id: int) -> dict[str, Any] | None:
    with connect() as con:
        r = con.execute("SELECT id,filename,size_bytes,submitted_at,is_late,note,marks,feedback,graded_at FROM assignment_submissions WHERE assignment_id=? AND student_id=?", (assignment_id, student_id)).fetchone()
    return dict(r) if r else None


# ------------------------------------------------------------------ downloads (access is checked on every call)
def get_assignment_file(user_id: str, assignment_id: int) -> tuple[str, bytes] | None:
    """The tutor's PDF: only for the owning tutor and members of the class."""
    with connect() as con:
        r = con.execute("SELECT f.filename,f.data FROM assignments a JOIN stored_files f ON f.id=a.file_id WHERE a.id=? AND a.kind='pdf' AND "
                        "(a.tutor_id=? OR EXISTS (SELECT 1 FROM class_members m WHERE m.class_id=a.class_id AND m.student_id=?))", (assignment_id, user_id, user_id)).fetchone()
    return (r["filename"], bytes(r["data"])) if r else None


def get_submission_file(user_id: str, submission_id: int) -> tuple[str, bytes] | None:
    """A solution: only for the student who uploaded it and the tutor who owns the assignment."""
    with connect() as con:
        r = con.execute("SELECT s.filename,f.data FROM assignment_submissions s JOIN assignments a ON a.id=s.assignment_id JOIN stored_files f ON f.id=s.file_id "
                        "WHERE s.id=? AND (s.student_id=? OR a.tutor_id=?)", (submission_id, user_id, user_id)).fetchone()
    return (r["filename"], bytes(r["data"])) if r else None


def storage_used_mb(tutor_id: str) -> float:
    """Database space used by this tutor's assignment sheets and their students' solutions."""
    with connect() as con:
        n = con.execute("SELECT COALESCE(SUM(f.size_bytes),0) FROM stored_files f WHERE f.id IN (SELECT file_id FROM assignments WHERE tutor_id=? AND file_id IS NOT NULL) "
                        "OR f.id IN (SELECT s.file_id FROM assignment_submissions s JOIN assignments a ON a.id=s.assignment_id WHERE a.tutor_id=?)", (tutor_id, tutor_id)).fetchone()[0]
    return round(int(n) / 1048576, 3)
