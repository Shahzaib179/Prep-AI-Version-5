"""Streamlit pages for the two CrewAI agents: Merit Advisor and Path Finder."""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from config import get_secret
from crew_agents.llm import crewai_available
from crew_agents.merit import formulas as F
from crew_agents.merit import history
from crew_agents.merit.crew import run_merit
from crew_agents.pathfinder.crew import STAGES, StudentProfile, run_pathfinder_stage
from crew_agents.tools import web_tools
from db import save_agent_session
from pdf_export import text_to_pdf
from ui import hero

BAND_ICON = {"Safe": "🟢 Safe", "Target": "🟡 Target", "Reach": "🟠 Reach", "Unlikely": "🔴 Unlikely"}
EV_ICON = {"verified": "✅ verified from fetched page", "unverified": "⚠️ unverified - treat as a lead only"}


def _need_crewai(feature: str) -> bool:
    ok, info = crewai_available()
    if not ok:
        st.warning(f"{feature} needs CrewAI. {info}. Install with `pip install -r requirements.txt`. "
                   "The deterministic calculator still works without it.")
    return ok


def _key_ok() -> bool:
    return bool((get_secret("GROQ_API_KEY") or "").strip())


# =============================================================================
# Merit Advisor
# =============================================================================
def render_merit_advisor(student_id: str, student_name: str, model: str) -> None:
    hero("🧮 Merit Aggregate & Admission Advisor",
         "Calculates merit for MDCAT, NUMS, ECAT, NUST NET, FAST, NTS-based universities and more, then compares it with previous-year closing merits.")
    history.init_tables()

    keys = list(F.PROFILES)
    labels = {k: f"{F.PROFILES[k].exam} - {F.PROFILES[k].institution}" for k in keys}
    key = st.selectbox("Exam / institution formula", keys, format_func=lambda k: labels[k])
    prof = F.PROFILES[key]

    with st.expander(f"Formula details - status: {prof.status}", expanded=prof.status != "verified-official"):
        st.write(f"**Weights (SSC / HSSC / Test):** {prof.weights[F.SSC]:g}% / {prof.weights[F.HSSC]:g}% / {prof.weights[F.TEST]:g}%  ·  "
                 f"**Test total:** {prof.test_total:g}  ·  **Effective:** {prof.effective}")
        if prof.notes:
            st.info(prof.notes)
        for v in prof.variants:
            st.caption(f"Alternative reported by some sources - {v.label}: {v.weights}. {v.note}")
        for s in prof.sources:
            st.caption(f"Source hint: {s}")
        st.caption("Admission formulas change by year, programme and seat type. Confirm in the official prospectus.")
        override = st.checkbox("Override weights manually (use numbers from the official prospectus)")
        weights = None
        if override:
            c1, c2, c3 = st.columns(3)
            w1 = c1.number_input("Matric %", 0.0, 100.0, float(prof.weights[F.SSC]), 0.5)
            w2 = c2.number_input("FSc/HSSC %", 0.0, 100.0, float(prof.weights[F.HSSC]), 0.5)
            w3 = c3.number_input("Test %", 0.0, 100.0, float(prof.weights[F.TEST]), 0.5)
            if round(w1 + w2 + w3, 6) != 100:
                st.error(f"Weights must add up to 100 (now {w1 + w2 + w3:g}).")
            else:
                weights = {F.SSC: w1, F.HSSC: w2, F.TEST: w3}

    st.subheader("Your marks")
    c1, c2, c3 = st.columns(3)
    ssc_o = c1.number_input("Matric obtained", 0.0, 5000.0, 900.0, 1.0)
    ssc_t = c1.number_input("Matric total", 1.0, 5000.0, 1100.0, 1.0)
    hs_o = c2.number_input("FSc/HSSC obtained", 0.0, 5000.0, 900.0, 1.0,
                           help="Use the part(s) the institution counts (e.g. NUST/UET may use Part-I only). Enter that total below.")
    hs_t = c2.number_input("FSc/HSSC total", 1.0, 5000.0, 1100.0, 1.0)
    t_o = c3.number_input(f"{prof.exam} test obtained", 0.0, 5000.0, 0.0, 1.0)
    t_t = c3.number_input(f"{prof.exam} test total", 1.0, 5000.0, float(prof.test_total), 1.0)

    st.subheader("Where do you want to apply?")
    c1, c2 = st.columns(2)
    inst = c1.text_input("Institution (optional)", placeholder="e.g. King Edward Medical University / UET Lahore / NUST")
    prog = c2.text_input("Programme (optional)", placeholder="e.g. MBBS / BS Computer Science")
    goal = st.text_input("Your goal (optional)", placeholder="e.g. government medical college, open merit, near Rawalpindi")

    st.subheader("Research options (free sources)")
    o1, o2, o3 = st.columns(3)
    verify = o1.checkbox("Verify formula online", True, help="Agent 1 reads the official prospectus and compares it with the stored formula.")
    hist_web = o2.checkbox("Search previous-year closing merits online", True, help="Agent 2. Only numbers found AND quoted on a real page are kept.")
    ai_advice = o3.checkbox("AI advice (Agent 3)", True)
    use_verified = st.checkbox("If a verified official formula differs from the stored one, use the verified one", False)

    with st.expander("Previous-year closing merits you already have (CSV)"):
        st.caption("Columns: exam, year, institution, program, closing_merit (optional: category, source_url). Official merit lists are best.")
        tmpl = "exam,year,institution,program,category,closing_merit,source_url\nMDCAT,2024,Example Medical College,MBBS,open merit,90.12,https://example.org/merit.pdf\n"
        st.download_button("Download CSV template", tmpl, "closing_merit_template.csv", "text/csv")
        up = st.file_uploader("Upload closing-merit CSV", type=["csv"], key="merit_csv")
        if up is not None and st.button("Import CSV"):
            rows, errs = history.parse_csv(up.getvalue())
            if errs:
                st.error(errs[0])
            else:
                st.success(f"Imported {history.add_rows(student_id, rows, 'upload')} rows.")
        stored = history.get_rows(student_id)
        if stored:
            df = pd.DataFrame(stored)[["exam", "year", "institution", "program", "category", "closing_merit", "origin", "verified"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("Delete my stored closing-merit rows"):
                history.clear_rows(student_id)
                st.rerun()

    if st.button("Calculate merit & recommend", type="primary"):
        if t_o > t_t:
            st.error("Test obtained marks exceed the total."); return
        if (verify or hist_web or ai_advice) and not (_need_crewai("The research agents") and _key_ok()):
            if not _key_ok():
                st.warning("GROQ_API_KEY is missing - running the calculator only.")
            verify = hist_web = ai_advice = False
        data = F.MeritInput(ssc_o, ssc_t, hs_o, hs_t, t_o, t_t)
        box = st.status("Running Merit Crew...", expanded=True)
        try:
            rep = run_merit(profile_key=key, data=data, student_id=student_id, institution=inst, program=prog, student_goal=goal,
                            verify_online=verify, history_online=hist_web, use_verified_weights=use_verified, custom_weights=weights, model=model,
                            use_ai_advice=ai_advice, progress=lambda m: box.write(m))
            box.update(label="Done", state="complete")
        except F.MeritError as exc:
            box.update(label="Input problem", state="error"); st.error(str(exc)); return
        except Exception as exc:  # noqa: BLE001
            box.update(label="Failed", state="error"); st.error("The merit run failed."); 
            if st.session_state.get("debug_mode"): st.exception(exc)
            return
        st.session_state.merit_report = rep
        summary = f"{prof.exam} aggregate {rep.primary.aggregate:.2f}% ({inst or 'no institution'} {prog})"
        save_agent_session(student_id, "Merit Advisor", summary, rep.advice or summary)

    rep = st.session_state.get("merit_report")
    if rep:
        _show_merit_report(rep, student_name)


def _show_merit_report(rep, student_name: str) -> None:
    p = rep.primary
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Aggregate", f"{p.aggregate:.2f}%")
    lo, hi = F.sensitivity_spread(rep.variants)
    m2.metric("Range across reported formulas", f"{lo:.2f}% - {hi:.2f}%")
    m3.metric("Formula status", rep.profile.status)
    for issue in p.eligibility_issues:
        st.error(issue)

    st.markdown("**How it was calculated**")
    st.dataframe(pd.DataFrame([{"Component": c, "Percentage": round(p.percentages[k], 2), "Weight %": p.weights.get(k, 0),
                                "Contribution": round(p.contributions[k], 2)}
                               for k, c in ((F.SSC, "Matric/SSC"), (F.HSSC, "FSc/HSSC"), (F.TEST, f"{rep.profile.exam} test"))]),
                 hide_index=True, use_container_width=True)
    if len(rep.variants) > 1:
        st.markdown("**Same marks under other formulas quoted by sources**")
        st.dataframe(pd.DataFrame([{"Formula": n, "Weights": str(r.weights), "Aggregate": round(r.aggregate, 2)} for n, r in rep.variants]),
                     hide_index=True, use_container_width=True)

    fc = rep.formula_check
    if fc:
        with st.expander("Agent 1 - Formula verification", expanded=True):
            if fc.get("verified"):
                tag = "matches the stored formula" if fc.get("matches_profile") else "DIFFERS from the stored formula"
                st.success(f"Official source found; it {tag}: {fc['proposed_weights']}")
                st.caption(f"{fc.get('tier')} - {fc.get('source_url')}")
                st.caption(f"Quote: “{fc.get('quote', '')[:300]}”")
            else:
                st.warning("Could not verify the formula on an official page - the stored (provisional) weights were used. "
                           "Check the prospectus yourself.")
            if fc.get("conflicts"):
                st.caption(f"Conflicts noted: {fc['conflicts']}")

    st.subheader("Admission recommendations (from previous-year closing merit)")
    if rep.recommendations:
        st.caption("Band rule: Safe = 2+ points above last year's cut-off · Target = within -1 · Reach = within -4 · otherwise Unlikely. "
                   "Past cut-offs do not guarantee this year's result.")
        st.dataframe(pd.DataFrame([{"Fit": BAND_ICON[r.band], "Institution": r.institution, "Programme": r.program, "Seat": r.category,
                                    "Year": r.year, "Closing merit": r.closing_merit, "Your margin": r.margin,
                                    "Evidence": "✅" if r.verified else "⚠️", "Source": r.source_url, "Trend": r.trend_note}
                                   for r in rep.recommendations]), hide_index=True, use_container_width=True)
    else:
        st.info("No verified previous-year closing merits were found. Upload an official merit list CSV above, or try a more specific "
                "institution/programme name. The app will not guess closing merits.")
    if rep.rejected_rows:
        with st.expander(f"{len(rep.rejected_rows)} agent-found number(s) rejected (quote not found on the fetched page)"):
            st.json(rep.rejected_rows)
    if rep.advice:
        st.subheader("Agent 3 - Advisor")
        st.markdown(rep.advice)
    for line in rep.log:
        st.caption(line)

    body = (f"Aggregate: {p.aggregate:.2f}% ({rep.profile.exam})\nWeights: {p.weights}\n" +
            "\n".join(f"{BAND_ICON[r.band]} | {r.institution} | {r.program} | {r.year} | closing {r.closing_merit} | margin {r.margin}"
                      for r in rep.recommendations) + "\n\n" + rep.advice)
    st.download_button("📥 Download merit report (PDF)", text_to_pdf("Prep AI - Merit & Admission Report", body, f"Student: {student_name}"),
                       "prep_ai_merit_report.pdf", "application/pdf", key="dl_merit")


# =============================================================================
# Path Finder
# =============================================================================
STAGE_LABEL = {"careers": "1 · Career Explorer", "programs": "2 · Program & University Scout",
               "verification": "3 · Eligibility & Accreditation Verifier", "scholarships": "4 · Scholarship Scout",
               "roadmap": "5 · Roadmap Planner"}


def render_pathfinder(student_id: str, student_name: str, model: str) -> None:
    hero("🧭 Path Finder",
         "Career paths, programmes, eligibility checks, accreditation, scholarships and a personalised roadmap - grounded in official sources, never invented.")
    st.session_state.setdefault("pf_state", {})
    state = st.session_state.pf_state

    with st.form("pf_profile"):
        c1, c2 = st.columns(2)
        education = c1.selectbox("Current education", ["Matric / O-Level", "Intermediate (FSc Pre-Medical)", "Intermediate (FSc Pre-Engineering)",
                                                      "Intermediate (ICS / FA / Commerce)", "A-Level", "Bachelor's (in progress / done)", "Other"], index=1)
        stream = c2.text_input("Stream / group", placeholder="Pre-Medical, Pre-Engineering, ICS...")
        subjects = c1.text_input("Subjects (comma separated)", placeholder="Physics, Chemistry, Biology, English")
        tests = c2.text_input("Entry tests taken / planned", placeholder="MDCAT, ECAT, NET")
        ssc = c1.number_input("Matric / O-Level % (0 = unknown)", 0.0, 100.0, 0.0, 0.5)
        hssc = c2.number_input("Intermediate / A-Level % (0 = unknown)", 0.0, 100.0, 0.0, 0.5)
        age = c1.number_input("Age (0 = skip)", 0, 60, 0)
        domicile = c2.text_input("Domicile / province", placeholder="Punjab")
        interests = st.text_area("Interests", placeholder="What do you enjoy? Subjects, activities, problems you like solving.")
        strengths = st.text_area("Strengths / skills", placeholder="e.g. strong in maths, good at communication")
        goals = st.text_area("Goals / constraints", placeholder="e.g. want a stable career, family can afford ~PKR 300k/year, prefer Islamabad/Rawalpindi")
        d1, d2, d3 = st.columns(3)
        abroad = d1.checkbox("Open to studying abroad")
        need_sch = d2.checkbox("I need financial aid")
        reloc = d3.checkbox("Willing to relocate", True)
        budget = st.number_input("Budget PKR/year (0 = not stated)", 0, 20_000_000, 0, 10_000)
        saved = st.form_submit_button("Save profile", type="primary")
    if saved:
        st.session_state.pathfinder_profile = StudentProfile(
            name=student_name, education=education, stream=stream.strip(), subjects=[s.strip() for s in subjects.split(",") if s.strip()],
            ssc_pct=ssc or None, hssc_pct=hssc or None, tests_taken=[s.strip() for s in tests.split(",") if s.strip()], age=age or None,
            domicile=domicile.strip(), interests=interests.strip(), strengths=strengths.strip(), goals=goals.strip(),
            budget_pkr_per_year=budget or None, willing_to_relocate=reloc, study_abroad=abroad, need_scholarship=need_sch)
        st.session_state.pf_state = state = {}
        web_tools.reset_cache()
        st.success("Profile saved. Run the agents below.")

    profile: StudentProfile | None = st.session_state.get("pathfinder_profile")
    if not profile:
        st.info("Fill in and save your profile to start.")
        return
    if not (_need_crewai("Path Finder") and _key_ok()):
        if not _key_ok():
            st.warning("GROQ_API_KEY is missing.")
        return

    st.caption("Each agent runs as its own small crew so it stays inside Groq's token limits. You can re-run any single stage.")
    cols = st.columns(len(STAGES) + 1)
    run_all = cols[0].button("▶ Run all", type="primary")
    clicked = [s for i, s in enumerate(STAGES) if cols[i + 1].button(STAGE_LABEL[s].split("·")[1].strip().split(" ")[0], key=f"pf_{s}")]
    todo = STAGES if run_all else clicked
    if todo:
        box = st.status("Running Path Finder agents...", expanded=True)
        for stg in todo:
            box.write(f"**{STAGE_LABEL[stg]}** - working...")
            try:
                state.update(run_pathfinder_stage(stg, profile, state, model, on_wait=lambda m: box.write(m)))
            except Exception as exc:  # noqa: BLE001
                box.write(f"⚠️ {STAGE_LABEL[stg]} failed: {str(exc)[:200]}")
                if st.session_state.get("debug_mode"): st.exception(exc)
                break
        box.update(label="Finished", state="complete")
        if "roadmap" in state:
            save_agent_session(student_id, "Path Finder", profile.brief(), json.dumps(state.get("roadmap"), ensure_ascii=False)[:4000])

    t = st.tabs(["Careers", "Programmes", "Eligibility & accreditation", "Scholarships", "Roadmap"])
    with t[0]:
        for c in state.get("careers", []):
            _evidence_card(c.get("title"), c, [("Why it fits", c.get("why_fit")), ("Typical route", c.get("typical_route")),
                                               ("Fields of study", ", ".join(c.get("fields_of_study") or []))])
    with t[1]:
        for pr in state.get("programs", []):
            _evidence_card(f"{pr.get('program')} - {pr.get('institution')}", pr,
                           [("City", pr.get("city")), ("Admission test", pr.get("admission_test")), ("Career link", pr.get("career_link")),
                            ("Requirements found", json.dumps({k: v for k, v in (pr.get("requirements") or {}).items() if v})) ])
    with t[2]:
        for v in state.get("verification", []):
            acc, el = v["accreditation"], v["eligibility"]
            with st.expander(f"{v['program']} - {v['institution']}  ·  {el['overall']}", expanded=False):
                if el["criteria"]:
                    st.dataframe(pd.DataFrame(el["criteria"]), hide_index=True, use_container_width=True)
                st.markdown(f"**HEC recognition:** {acc.get('hec_recognised', 'not_found')}  ·  **Council:** {acc.get('council', '-')} → {acc.get('council_status', 'not_found')}")
                st.caption(EV_ICON.get(acc.get("evidence", "unverified")) + f" · {acc.get('source_tier', '')} · {acc.get('source_url', '')}")
                if acc.get("downgraded"):
                    st.warning(acc["downgraded"])
                st.caption(el.get("disclaimer", ""))
    with t[3]:
        for s in state.get("scholarships", []):
            _evidence_card(s.get("name"), s, [("Provider", s.get("provider")), ("Level", s.get("level")), ("Eligibility", s.get("eligibility_summary")),
                                              ("Deadline", s.get("deadline") or "not stated on source - check official page"), ("Covers", s.get("covers") or "not stated")])
    with t[4]:
        rm = state.get("roadmap")
        if rm:
            st.markdown(rm.get("summary", ""))
            lines = [rm.get("summary", "")]
            for ph in rm.get("phases", []):
                st.markdown(f"### {ph.get('phase')}")
                lines.append(f"\n{ph.get('phase')}")
                for sp in ph.get("steps", []):
                    flag = "✅" if sp.get("status") == "ready" else "🔎"
                    st.markdown(f"{flag} **{sp.get('action')}** - {sp.get('why', '')}  \n<span class='small-muted'>Verify at: {sp.get('verify_at', '-')}</span>", unsafe_allow_html=True)
                    lines.append(f"- {sp.get('action')} ({sp.get('why', '')}) verify at {sp.get('verify_at', '-')}")
            if rm.get("backup_plan"):
                st.info(f"Backup plan: {rm['backup_plan']}")
                lines.append(f"\nBackup: {rm['backup_plan']}")
            st.download_button("📥 Download roadmap (PDF)", text_to_pdf("Prep AI - Personalised Roadmap", "\n".join(lines), f"Student: {student_name}"),
                               "prep_ai_roadmap.pdf", "application/pdf", key="dl_roadmap")


def _evidence_card(title: str | None, item: dict, fields: list[tuple[str, str | None]]) -> None:
    ev = item.get("evidence", "unverified")
    with st.expander(f"{'✅' if ev == 'verified' else '⚠️'} {title}"):
        for k, v in fields:
            if v:
                st.write(f"**{k}:** {v}")
        st.caption(f"{EV_ICON[ev]} · {item.get('source_tier', '')}")
        if item.get("source_url"):
            st.caption(item["source_url"])
        if item.get("quote"):
            st.caption(f"“{item['quote'][:300]}”")
