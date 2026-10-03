from __future__ import annotations

import streamlit as st


def apply_theme(primary_color: str = "#2563EB", theme: str = "System", font_scale: int = 100) -> None:
    st.markdown(
        f"""
        <style>
        :root {{ --prep-primary: {primary_color}; }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
            font-size: {font_scale}%;
        }}
        {"[data-testid='stAppViewContainer'] {background:#0e1117;color:#f3f4f6;} [data-testid='stSidebar'] {background:#111827;color:#f3f4f6;}" if theme == "Dark" else ""}
        {"[data-testid='stAppViewContainer'] {background:#ffffff;color:#111827;} [data-testid='stSidebar'] {background:#f8fafc;color:#111827;}" if theme == "Light" else ""}

        .prep-hero {{padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(135deg,#111827,#1f2937); color:white; margin-bottom:1rem; border-left: 5px solid var(--prep-primary);}}
        .prep-card {{padding:1rem; border:1px solid rgba(128,128,128,.25); border-radius:16px; background:rgba(128,128,128,.05); margin-bottom:.7rem;}}
        .small-muted {{color:#6b7280; font-size:.9rem;}}
        div.stButton > button[kind="primary"] {{background-color: var(--prep-primary); border-color: var(--prep-primary);}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="prep-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def source_cards(sources: list[dict]) -> None:
    if not sources:
        return
    st.subheader("Retrieved Sources")
    for i, s in enumerate(sources, 1):
        with st.expander(f"Source {i}: {s.get('filename', 'Source')} — Page {s.get('page') or 'N/A'}"):
            st.caption(f"Chunk {s.get('chunk_id', 'N/A')} · hybrid score {float(s.get('hybrid_score', 0)):.3f}")
            st.write(s.get("text", ""))
