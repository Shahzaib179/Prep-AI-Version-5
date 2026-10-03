# Prep AI V5 — Student Quiz & Exam Preparation Assistant

Prep AI is a Streamlit-based multi-agent learning assistant using Groq, FAISS/RAG,
SQLite, adaptive mastery tracking, long-term semantic memory, timed quizzes,
published multi-user quizzes, charts, and user interface customization.

## New features in this update

- Clear semantic long-term memory without deleting quiz history.
- Reset all student learning data while retaining the student profile.
- Sidebar long-term-memory usage indicator.
- Dashboard progress charts instead of percentage-only progress.
- Timed quizzes with a live decrementing countdown and automatic submission.
- Publish a quiz with a shareable code.
- Multiple students can take the same published quiz.
- Each student receives a different deterministic question sequence.
- Student-specific published quiz results.
- Tutor-only published quiz result dashboard protected by `TUTOR_ACCESS_CODE`.
- Application theme: System / Light / Dark.
- Accent color selection.
- Font-size selection.
- Settings are stored per student.

## Streamlit Cloud deployment

1. Upload the project files to GitHub.
2. Create a Streamlit Cloud app pointing to `app.py`.
3. In **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
TUTOR_ACCESS_CODE = "use_a_strong_private_code"
```

Never commit `.streamlit/secrets.toml` or API keys to GitHub.

## Published quiz workflow

1. A student/tutor creates a normal quiz from **Learn**.
2. Open **Published Quizzes → Publish**.
3. Set the expected number of students and time limit.
4. Publish and copy the generated quiz code.
5. Students open **Published Quizzes → Join Quiz** and enter the code.
6. The same questions are used for everyone, but the question sequence is
   randomized per Student ID.
7. Each submission is saved with Student ID, student name, score, answers and
   submission time.
8. A tutor enters the private `TUTOR_ACCESS_CODE`, then enters the quiz code to
   see the results.

Students should use unique Student IDs.

## Important deployment note about long-term storage

The application uses SQLite and local FAISS memory files as the fallback storage
layer. This works for multiple simultaneous sessions while the Streamlit app
instance is running, but Streamlit Community Cloud can restart/rebuild an app,
so local filesystem data should not be treated as guaranteed permanent storage.

For a production university deployment where quiz results and long-term memory
must survive restarts, connect the database layer to a persistent external
database such as PostgreSQL/Supabase. The current update keeps the existing
SQLite architecture so it remains compatible with the supplied V4 project.

## Existing RAG index

Keep the existing `faiss_index/` directory and its index/metadata files in the
repository when the application depends on the prebuilt knowledge base.

## Main files

- `app.py` — Streamlit UI and application flow
- `db.py` — SQLite schema, student data, quiz history and published quizzes
- `memory.py` — per-student semantic long-term memory
- `ui.py` — theme and font-size styling
- `agent_system.py` — multi-agent learning workflows
- `groq_service.py` — Groq API integration
- `rag.py` — FAISS retrieval
- `adaptive.py` / `mastery_model.py` — adaptive learning and mastery
- `faiss_index/` — prebuilt knowledge index
- `requirements.txt` — deployment dependencies
- `secrets.example.toml` — Streamlit Secrets template

## New agents (V5.1)

- **Merit Aggregate Agent** (`merit_engine.py`) — weighted aggregate for MDCAT, NUMS, ECAT, NUST NET, NTS/university tests or a custom formula. Weights are editable; formulas not confirmed from official sources are flagged "verify". Compares the aggregate with the last 3 years of imported closing merit and labels each program Safe / Target / Reach.
- **Path Finder Agent** (`pathfinder.py`) — matches the student profile to verified programs, scholarships and careers; runs deterministic eligibility checks; the LLM may only explain and draft the roadmap using supplied record IDs, and says "Not available in verified data" when information is missing.
- `career_db.py` — SQLite tables + CSV import. Every row requires a `source_url`. Imports are protected by `TUTOR_ACCESS_CODE`.
- `agents_ui.py` — Streamlit pages. No new pip dependencies.

No merit, fee or scholarship data is bundled. Import official data from the CSV templates in each agent's data tab.
