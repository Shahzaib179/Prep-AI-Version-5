"""Streamlit pages for the Merit Aggregate and Path Finder agents."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from career_db import SCHEMAS, count, fetch, import_csv, template_csv
from config import get_secret
from db import save_agent_session
from merit_engine import FORMULAS, calculate_aggregate, recommend
from pathfinder import NA, PathFinderAgent, StudentProfile
from ui import hero

BAND_ICON = {"Safe": "🟢", "Target": "🟡", "Reach": "🟠", "Unlikely": "🔴"}


def _data_manager(table: str, label: str) -> None:
    """Tutor-protected CSV import so only trusted staff change the shared data."""
    st.caption(f"{count(table)} records stored. Every row needs a `source_url` so results stay traceable.")
    st.download_button(f"⬇️ {label} CSV template", template_csv(table), f"{table}_template.csv", "text/csv", key=f"tpl_{table}")
    code = st.text_input("Tutor access code", type="password", key=f"code_{table}")
    up = st.file_uploader("Upload CSV", type="csv", key=f"up_{table}")
    replace = st.checkbox("Replace existing records", key=f"rep_{table}")
    if st.button("Import", key=f"imp_{table}") and up:
        expected = get_secret("TUTOR_ACCESS_CODE")
        if not expected or code != expected:
            st.error("Invalid tutor access code.")
            return
        n, problems = import_csv(table, up.getvalue(), replace)
        st.success(f"Imported {n} rows.") if n else st.warning("No rows imported.")
        for p in problems[:10]:
            st.caption(f"⚠️ {p}")


def render_merit(student_id: str) -> None:
    hero("🧮 Merit Aggregate Agent", "Calculate your aggregate for MDCAT, ECAT, NUMS, NTS and more, then compare it with previous years' closing merit.")
    tabs = st.tabs(["Calculate & Recommend", "Merit Data (admin)"])
    with tabs[1]:
        _data_manager("merit_history", "Merit history")
    with tabs[0]:
        name = st.selectbox("Exam / formula", list(FORMULAS))
        f = FORMULAS[name]
        (st.success if f["verified"] else st.warning)(("✅ " if f["verified"] else "⚠️ Verify weights: ") + f["note"])
        c1, c2, c3 = st.columns(3)
        w = (c1.number_input("Matric weight %", 0.0, 100.0, float(f["w"][0]), key=f"wm_{name}"),
             c2.number_input("Inter weight %", 0.0, 100.0, float(f["w"][1]), key=f"wi_{name}"),
             c3.number_input("Test weight %", 0.0, 100.0, float(f["w"][2]), key=f"wt_{name}"))
        a1, a2, a3 = st.columns(3)
        m, mt = a1.number_input("Matric obtained", 0.0, value=900.0), a1.number_input("Matric total", 1.0, value=1100.0)
        i, it = a2.number_input("Inter obtained", 0.0, value=900.0), a2.number_input("Inter total", 1.0, value=1100.0)
        t, tt = a3.number_input("Test obtained", 0.0, value=0.0), a3.number_input("Test total", 1.0, value=float(f["test_total"]), key=f"tt_{name}")
        if st.button("Calculate merit", type="primary"):
            res = calculate_aggregate(m, mt, i, it, t, tt, w)
            if not res["ok"]:
                for e in res["errors"]:
                    st.error(e)
            else:
                st.session_state.merit_result = {**res, "exam": name}
                save_agent_session(student_id, "Merit Aggregate Agent", f"{name} m={m}/{mt} i={i}/{it} t={t}/{tt} w={w}", f"Aggregate {res['aggregate']}%")
        res = st.session_state.get("merit_result")
        if not res:
            return
        st.metric("Your aggregate", f"{res['aggregate']:.3f}%")
        st.bar_chart(pd.DataFrame({"Contribution": res["parts"]}))
        hist = fetch("merit_history")
        if not hist:
            st.info(f"{NA}: no previous-year merit data has been imported yet, so no recommendations can be made. Use the Merit Data tab.")
            return
        st.subheader("Recommendations from previous years")
        inst = sorted({h["institution"] for h in hist})
        pick = st.multiselect("Filter institutions", inst)
        unlikely = st.checkbox("Show unlikely options")
        rows = recommend(res["aggregate"], [h for h in hist if not pick or h["institution"] in pick], include_unlikely=unlikely)
        for band in ("Safe", "Target", "Reach", "Unlikely"):
            sub = [r for r in rows if r["band"] == band]
            if sub:
                st.markdown(f"#### {BAND_ICON[band]} {band}")
                st.dataframe(pd.DataFrame(sub)[["institution", "program", "category", "latest_year", "latest_closing", "margin", "trend", "source_url"]], hide_index=True)
        if not rows:
            st.info("No matching programs for this aggregate in the imported data.")
        st.caption("Past closing merit does not guarantee future results; seats, quotas and competition change every year.")


def render_pathfinder(student_id: str) -> None:
    hero("🧭 Path Finder", "Grounded career paths, programs, eligibility, scholarships, accreditation and a personalized roadmap.")
    tabs = st.tabs(["Find My Path", "Programs data", "Scholarships data", "Careers data"])
    for tab, table, label in zip(tabs[1:], ("pf_programs", "pf_scholarships", "pf_careers"), ("Programs", "Scholarships", "Careers")):
        with tab:
            _data_manager(table, label)
    with tabs[0]:
        c1, c2 = st.columns(2)
        stream = c1.selectbox("Stream", ["Pre-Medical", "Pre-Engineering", "ICS / Computer Science", "Commerce", "Arts / Humanities", "A-Levels", "Other"])
        inter = c1.number_input("Intermediate % (leave 0 if unknown)", 0.0, 100.0, 0.0)
        matric = c1.number_input("Matric % (leave 0 if unknown)", 0.0, 100.0, 0.0)
        tests = c1.multiselect("Entry tests taken", ["MDCAT", "ECAT", "NUMS", "NTS NAT", "NUST NET", "FAST", "SAT", "Other"])
        interests = c2.text_area("Interests / strengths")
        goal = c2.text_input("Career goal (optional)")
        city = c2.text_input("Preferred city (optional)")
        budget = c2.number_input("Max tuition per year, PKR (0 = no limit)", 0, value=0, step=50000)
        income = c2.number_input("Family income per month, PKR (0 = prefer not to say)", 0, value=0, step=10000)
        web = st.checkbox("Allow trusted web search when verified data is thin (results marked unverified)")
        if st.button("Find my path", type="primary"):
            p = StudentProfile(stream, inter or None, matric or None, interests, goal, city, budget or None, income or None, tests)
            with st.spinner("Matching verified records..."):
                try:
                    st.session_state.pf_result = PathFinderAgent().run(p, use_web=web)
                    save_agent_session(student_id, "Path Finder Agent", f"{stream} | {interests} | {goal}", str(st.session_state.pf_result["ai"].get("summary", "")))
                except Exception:
                    st.error("Path Finder could not complete the request. Check your Groq configuration.")
        r = st.session_state.get("pf_result")
        if not r:
            return
        ai = r["ai"]
        if ai.get("summary"):
            st.info(ai["summary"])
        if ai.get("career_paths"):
            st.subheader("Career paths")
            for c in ai["career_paths"]:
                st.markdown(f"**{c.get('career', '')}** — {c.get('why', '')}  \n`{', '.join(c.get('source_ids', []))}`")
        fit = {x["id"]: x.get("reason", "") for x in ai.get("program_fit", [])}
        if r["programs"]:
            st.subheader("Programs & eligibility")
            for pr in r["programs"]:
                e = pr["eligibility"]
                icon = {"Eligible": "✅", "Needs confirmation": "❓", "Not eligible": "⛔"}[e["status"]]
                with st.expander(f"{icon} {pr['institution']} — {pr['program']} ({e['status']})"):
                    st.write(fit.get(f"P{pr['id']}", ""))
                    st.write(f"**Accreditation:** {pr.get('accreditation_body') or NA} — {pr.get('accreditation_status') or NA}")
                    st.write(f"**Tuition/year (PKR):** {pr.get('tuition_per_year_pkr') or NA} · **City:** {pr.get('city') or NA} · **Last verified:** {pr.get('last_verified') or NA}")
                    for x in e["fails"]:
                        st.error(x)
                    for x in e["notes"]:
                        st.caption(f"• {x}")
                    st.markdown(f"[Official source]({pr['source_url']})")
        if r["scholarships"]:
            st.subheader("Scholarships you may qualify for")
            for s in r["scholarships"]:
                st.markdown(f"**{s['name']}** ({s.get('provider') or NA}) · Covers: {s.get('covers') or NA} · Deadline: {s.get('deadline') or NA} · "
                            f"{' '.join(s['flags'])} [source]({s['source_url']})")
        if r["web"]:
            st.subheader("Unverified web results")
            for w in r["web"]:
                st.caption(f"[{w['title']}]({w['url']}) — {w['snippet'][:200]}")
        if ai.get("roadmap"):
            st.subheader("Your roadmap")
            for ph in ai["roadmap"]:
                st.markdown(f"**{ph.get('phase', '')}** · {ph.get('timeframe', '')}")
                for a in ph.get("actions", []):
                    st.write(f"• {a}")
        for g in (r["gaps"] + (ai.get("gaps") or [])):
            st.warning(g)
        st.caption(r["disclaimer"])
