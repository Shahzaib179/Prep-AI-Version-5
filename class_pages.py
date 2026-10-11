"""Class screens: tutors manage classes and deadlines, students join classes and see what is due."""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

import streamlit as st

import classes
import config
import pdf_assignments
from db import list_shared_quizzes
from ui import hero

PDF_LABEL = {"done": "✅ Submitted", "done_late": "🟠 Submitted late", "open": "🟦 Not submitted", "due_soon": "🟡 Due soon, not submitted",
             "overdue": "🔴 Overdue (late upload allowed)", "missed": "⛔ Missed (uploads closed)"}
STATUS_LABEL = {"done": "✅ Done", "done_late": "🟠 Done late", "open": "🟦 Open", "due_soon": "🟡 Due soon",
                "overdue": "🔴 Overdue (late allowed)", "missed": "⛔ Missed"}


def _lazy_download(label: str, key: str, loader, filename: str) -> None:
    """Files are read from the database only when asked for, so a page with many assignments stays fast."""
    got = st.session_state.get(key)
    if got is None:
        if st.button(label, key=key + "_load"):
            data = loader()
            if data is None:
                st.error("File not found.")
            else:
                st.session_state[key] = data
                st.rerun()
        return
    st.download_button(f"⬇ {filename}", got, file_name=filename, mime="application/pdf", key=key + "_dl")


def _local_now() -> datetime:
    return classes.utc_iso_to_local(classes.now_utc().isoformat(timespec="minutes")) or datetime.utcnow()


def report_csv(report: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["student_id", "name", "status", "score", "submitted_at", "time_taken_sec"])
    for r in report["rows"]:
        w.writerow([r["student_id"], r["name"], r["status"], r["score"] if r["score"] is not None else "", r["submitted_at"] or "", r["time_taken_sec"] or ""])
    return buf.getvalue()


# ------------------------------------------------------------------ tutor
def render_tutor_classes(tutor_id: str) -> None:
    st.subheader("Classes and deadlines")
    st.caption("Create a class, give students its code, then assign a published quiz with a deadline. Students must START the quiz before the deadline.")
    with st.form("cls_new", clear_on_submit=True):
        name = st.text_input("New class name", placeholder="e.g. MDCAT 2027 – Morning batch", max_chars=80)
        if st.form_submit_button("Create class", type="primary"):
            try:
                c = classes.create_class(tutor_id, name)
                st.success(f"Class created. Share this code with students: **{c['join_code']}**")
            except ValueError as exc:
                st.error(str(exc))

    show_archived = st.checkbox("Show archived classes", key="cls_archived")
    items = classes.list_classes(tutor_id, include_archived=show_archived)
    if not items:
        st.info("No classes yet. Create one above.")
        return
    for c in items:
        label = f"{c['name']} · code {c['join_code']} · {c['members']} student(s) · {c['assignments']} assignment(s)" + ("" if c["is_active"] else " · ARCHIVED")
        with st.expander(label):
            _class_detail(tutor_id, c)


def _deadline_fields(prefix: str):
    has_due = st.checkbox("Set a deadline", value=True, key=f"dl_has_{prefix}")
    d1, d2 = st.columns(2)
    due_d = d1.date_input("Due date", value=_local_now().date() + timedelta(days=3), key=f"dl_d_{prefix}")
    due_t = d2.time_input("Due time", value=time(23, 59), key=f"dl_t_{prefix}")
    late = st.checkbox("Allow late " + ("start/upload" if prefix.startswith("p") else "start") + " (flagged as late)", value=False, key=f"dl_l_{prefix}")
    return due_d, due_t, has_due, late


def _class_detail(tutor_id: str, c: dict[str, Any]) -> None:
    cid = c["id"]
    st.markdown(f"**Class code:** `{c['join_code']}` — students enter it under *My Classes*.")
    mem = classes.members(tutor_id, cid)
    if mem:
        st.dataframe([{"Name": m["name"], "Student ID": m["student_id"], "Joined": classes.fmt_local(m["joined_at"])} for m in mem], hide_index=True, use_container_width=True)
        who = st.selectbox("Remove a student", [""] + [m["student_id"] for m in mem], format_func=lambda s: s or "—", key=f"cls_rm_{cid}")
        if who and st.button("Remove from class", key=f"cls_rmb_{cid}"):
            classes.remove_member(tutor_id, cid, who)
            st.rerun()
    else:
        st.caption("No students have joined yet.")

    if c["is_active"]:
        st.markdown("##### New assignment")
        kind = st.radio("Type", ["Quiz", "PDF assignment"], horizontal=True, key=f"cls_kind_{cid}", help="A quiz is auto-marked. A PDF assignment is a document students answer by uploading their own PDF, which you mark.")
        if kind == "Quiz":
            quizzes = [q for q in list_shared_quizzes(tutor_id)]
            if not quizzes:
                st.info("Publish a quiz first (Publish tab), then come back to assign it.")
            else:
                with st.form(f"cls_assign_{cid}"):
                    pick = st.selectbox("Published quiz", quizzes, format_func=lambda q: f"{q['title']} · {q['code']}")
                    title = st.text_input("Assignment title (optional)")
                    due_d, due_t, has_due, late = _deadline_fields(f"q{cid}")
                    if st.form_submit_button("Assign", type="primary"):
                        due = classes.local_to_utc_iso(due_d, due_t) if has_due else None
                        ok, msg = classes.create_assignment(tutor_id, cid, pick["code"], title, due, late)
                        (st.success if ok else st.error)(msg)
                        if ok:
                            st.rerun()
        else:
            with st.form(f"cls_pdf_{cid}"):
                title = st.text_input("Title", placeholder="e.g. Chapter 3 homework")
                instructions = st.text_area("Instructions for students (optional)", max_chars=2000)
                sheet = st.file_uploader(f"Assignment PDF (up to {config.ASSIGNMENT_MAX_MB} MB)", type=["pdf"], key=f"cls_pdf_file_{cid}")
                marks = st.number_input("Total marks (0 = not marked with numbers)", 0, 1000, 0)
                due_d, due_t, has_due, late = _deadline_fields(f"p{cid}")
                if st.form_submit_button("Assign PDF", type="primary"):
                    if not sheet:
                        st.error("Please choose the assignment PDF.")
                    else:
                        due = classes.local_to_utc_iso(due_d, due_t) if has_due else None
                        ok, msg = pdf_assignments.create_pdf_assignment(tutor_id, cid, title, instructions, sheet.name, sheet.getvalue(), due, late, int(marks) or None)
                        (st.success if ok else st.error)(msg)
                        if ok:
                            st.rerun()
            st.caption("Students upload their solution as a PDF before the deadline. Files are stored in the app's database.")

    for r in classes.list_assignments(tutor_id, cid):
        st.divider()
        _assignment_block(tutor_id, r)

    st.divider()
    if c["is_active"]:
        if st.button("Archive class", key=f"cls_arch_{cid}"):
            classes.set_class_active(tutor_id, cid, False)
            st.rerun()
    elif st.button("Restore class", key=f"cls_rest_{cid}"):
        classes.set_class_active(tutor_id, cid, True)
        st.rerun()


# ------------------------------------------------------------------ student
def render_my_classes(student_id: str, open_quiz: Callable[[str], None]) -> None:
    hero("🏫 My Classes", "Join your tutor's class with its code and see every assignment with its deadline.")
    with st.form("cls_join", clear_on_submit=True):
        code = st.text_input("Class code", placeholder="e.g. K7M2QX", max_chars=12)
        if st.form_submit_button("Join class", type="primary"):
            ok, msg, _ = classes.join_class(student_id, code)
            (st.success if ok else st.error)(msg)

    mine = classes.my_classes(student_id)
    if not mine:
        st.info("You have not joined a class yet.")
        return
    st.markdown("#### Assignments")
    items = classes.student_assignments(student_id)
    if not items:
        st.caption("No assignments yet.")
    for a in items:
        with st.container(border=True):
            if a["kind"] == "pdf":
                _student_pdf_assignment(student_id, a)
                continue
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**📝 {a['title']}** · {a['class_name']}")
            c1.caption(f"{STATUS_LABEL.get(a['status'], a['status'])} · due {classes.fmt_local(a['due_at'])}" + (f" · score {a['score']:.0f}%" if a["score"] is not None else ""))
            if a["status"] in ("open", "due_soon", "overdue") and a["is_open"]:
                if c2.button("Start", key=f"cls_go_{a['id']}", type="primary"):
                    open_quiz(a["quiz_code"])
    st.markdown("#### My classes")
    for c in mine:
        col1, col2 = st.columns([4, 1])
        col1.write(f"{c['name']} — {c['tutor_name']}" + ("" if c["is_active"] else " (closed)"))
        if col2.button("Leave", key=f"cls_leave_{c['id']}"):
            classes.leave_class(student_id, c["id"])
            st.rerun()


def _assignment_block(tutor_id: str, r: dict[str, Any]) -> None:
    pdf = r["kind"] == "pdf"
    labels = PDF_LABEL if pdf else STATUS_LABEL
    st.markdown(f"**{'📄 ' if pdf else '📝 '}{r['title']}** · " + ("PDF assignment" if pdf else f"quiz `{r['quiz_code']}`") + f" · due {classes.fmt_local(r['due_at'])}" + (" · late allowed" if r["allow_late"] else ""))
    if pdf and r.get("instructions"):
        st.caption(r["instructions"])
    unit = f" of {r['max_marks']}" if pdf and r.get("max_marks") else "%"
    avg = f" · average {r['average']}{unit}" if r["average"] is not None else ""
    st.caption(f"{r['completed']}/{r['members']} {'submitted' if pdf else 'completed'} · {r['late']} late{avg}")
    if r["rows"]:
        if pdf:
            st.dataframe([{"Student": x["name"], "Status": labels.get(x["status"], x["status"]), "Marks": x["score"], "Uploaded": classes.fmt_local(x["submitted_at"]) if x["submitted_at"] else "", "File": x["filename"] or ""} for x in r["rows"]], hide_index=True, use_container_width=True)
        else:
            st.dataframe([{"Student": x["name"], "Status": labels.get(x["status"], x["status"]), "Score %": x["score"], "Submitted": classes.fmt_local(x["submitted_at"]) if x["submitted_at"] else ""} for x in r["rows"]], hide_index=True, use_container_width=True)
    if r["missing"]:
        st.warning(("No solution from: " if pdf else "Not done yet: ") + ", ".join(x["name"] for x in r["missing"]))

    if pdf:
        _lazy_download("Show the assignment PDF", f"asg_sheet_{r['id']}", lambda: (pdf_assignments.get_assignment_file(tutor_id, r["id"]) or (None, None))[1], pdf_assignments.sanitize_filename(r["title"]))
        subs = [x for x in r["rows"] if x["submission_id"]]
        if subs:
            who = st.selectbox("Review a solution", [None] + subs, format_func=lambda x: "Choose a student..." if x is None else f"{x['name']} · {'marked ' + str(x['score']) if x['graded'] else 'not marked yet'}", key=f"asg_who_{r['id']}")
            if who:
                _review_panel(tutor_id, r, who)
            zkey = f"asg_zip_{r['id']}"
            if st.session_state.get(zkey):
                st.download_button("⬇ All solutions (ZIP)", st.session_state[zkey], file_name=f"{pdf_assignments.sanitize_filename(r['title'])[:-4]}_solutions.zip", mime="application/zip", key=zkey + "_dl")
            elif st.button("Prepare ZIP of all solutions", key=zkey + "_btn"):
                st.session_state[zkey] = pdf_assignments.submissions_zip(tutor_id, r["id"])
                st.rerun()

    b1, b2, b3 = st.columns(3)
    b1.download_button("Download CSV", report_csv(r), file_name=f"assignment_{r['quiz_code']}.csv", mime="text/csv", key=f"cls_csv_{r['id']}")
    with b2.popover("Change deadline"):
        cur = classes.utc_iso_to_local(r["due_at"])
        nd = st.date_input("Date", value=(cur.date() if cur else _local_now().date() + timedelta(days=1)), key=f"cls_nd_{r['id']}")
        nt = st.time_input("Time", value=(cur.time() if cur else time(23, 59)), key=f"cls_nt_{r['id']}")
        nl = st.checkbox("Allow late start" + (" / upload" if pdf else ""), value=bool(r["allow_late"]), key=f"cls_nl_{r['id']}")
        if st.button("Save", key=f"cls_ns_{r['id']}"):
            classes.update_deadline(tutor_id, r["id"], classes.local_to_utc_iso(nd, nt), nl)
            st.rerun()
    if b3.button("Remove assignment", key=f"cls_del_{r['id']}"):
        classes.delete_assignment(tutor_id, r["id"])
        st.rerun()


def _review_panel(tutor_id: str, r: dict[str, Any], row: dict[str, Any]) -> None:
    sid = row["submission_id"]
    with st.container(border=True):
        late = " · ⚠ uploaded late" if row["status"] == "done_late" else ""
        st.markdown(f"**{row['name']}** · uploaded {classes.fmt_local(row['submitted_at'])}{late} · {round((row['size_bytes'] or 0) / 1024)} KB")
        if row.get("note"):
            st.caption(f"Student's note: {row['note']}")
        _lazy_download("Open this solution", f"asg_sub_{sid}", lambda: (pdf_assignments.get_submission_file(tutor_id, sid) or (None, None))[1], pdf_assignments.sanitize_filename(f"{row['name']}_{r['title']}"))
        mx = r.get("max_marks")
        with st.form(f"asg_grade_{sid}"):
            marks = st.number_input(f"Marks{f' (out of {mx})' if mx else ''}", 0.0, float(mx or 1000), float(row["score"] or 0.0), step=0.5)
            feedback = st.text_area("Feedback for the student", value=row.get("feedback") or "", max_chars=2000)
            if st.form_submit_button("Save marks", type="primary"):
                ok, msg = pdf_assignments.grade(tutor_id, sid, float(marks), feedback)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
        if row["graded"] and st.button("Re-open for the student to re-upload", key=f"asg_reopen_{sid}"):
            pdf_assignments.reopen(tutor_id, sid)
            st.rerun()


def _student_pdf_assignment(student_id: str, a: dict[str, Any]) -> None:
    aid = a["id"]
    st.markdown(f"**📄 {a['title']}** · {a['class_name']}")
    st.caption(f"{PDF_LABEL.get(a['status'], a['status'])} · due {classes.fmt_local(a['due_at'])}" + (f" · total marks {a['max_marks']}" if a.get("max_marks") else ""))
    if a.get("instructions"):
        st.write(a["instructions"])
    _lazy_download("Show the assignment PDF", f"asg_sheet_{aid}", lambda: (pdf_assignments.get_assignment_file(student_id, aid) or (None, None))[1], pdf_assignments.sanitize_filename(a["title"]))
    sub = a.get("submission")
    if sub:
        late = " (late)" if a["status"] == "done_late" else ""
        st.success(f"Your solution: **{sub['filename']}** · uploaded {classes.fmt_local(sub['submitted_at'])}{late}")
        if sub.get("graded_at") and (sub.get("marks") is not None or sub.get("feedback")):
            st.info((f"**Marks: {sub['marks']:g}" + (f" / {a['max_marks']}" if a.get("max_marks") else "") + "**  \n" if sub.get("marks") is not None else "") + (f"Feedback: {sub['feedback']}" if sub.get("feedback") else ""))
        _lazy_download("Show my uploaded solution", f"asg_mine_{sub['id']}", lambda: (pdf_assignments.get_submission_file(student_id, sub["id"]) or (None, None))[1], sub["filename"] or "solution.pdf")
    locked = bool(sub and sub.get("graded_at"))
    if a["status"] == "missed" and not sub:
        st.error("The deadline has passed and uploads are closed. Contact your tutor.")
    elif locked:
        st.caption("Your solution has been marked, so it is locked. Ask your tutor if it needs to be re-opened.")
    else:
        if a["status"] == "overdue":
            st.warning("The deadline has passed. A late upload is allowed but will be marked late.")
        up = st.file_uploader(("Replace your solution (PDF)" if sub else "Upload your solution (PDF)") + f", up to {config.ASSIGNMENT_MAX_MB} MB", type=["pdf"], key=f"asg_up_{aid}")
        note = st.text_input("Note to your tutor (optional)", key=f"asg_note_{aid}", max_chars=500)
        if st.button("Submit solution" if not sub else "Replace solution", type="primary", key=f"asg_submit_{aid}", disabled=up is None):
            ok, msg, _ = pdf_assignments.submit_solution(student_id, aid, up.name, up.getvalue(), note)
            (st.success if ok else st.error)(msg)
            if ok:
                st.session_state.pop(f"asg_mine_{(sub or {}).get('id')}", None)
                st.rerun()
