"""Run with:  python -m unittest discover -s tests -v
No network, no Groq key and no CrewAI install needed: LLM stages are replaced by canned answers,
which lets us test the part that matters most - the grounding / verification logic around them."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "streamlit" not in sys.modules:   # minimal stub so config/db import without Streamlit installed
    st = types.ModuleType("streamlit")
    st.secrets = {}
    st.session_state = {}
    st.cache_resource = lambda *a, **k: (lambda f: f)
    sys.modules["streamlit"] = st

import db  # noqa: E402
from crew_agents import evidence  # noqa: E402
from crew_agents.merit import crew as merit_crew, formulas as F, history  # noqa: E402
from crew_agents.pathfinder import crew as pf_crew, eligibility  # noqa: E402
from crew_agents.tools import web_tools  # noqa: E402
from crew_agents.util import extract_json, is_rate_limit, with_backoff  # noqa: E402


class FormulaTests(unittest.TestCase):
    def test_nust_worked_example_from_public_calculator(self):
        r = F.calculate(F.PROFILES["nust_net"], F.MeritInput(800, 1000, 900, 1000, 150, 200))
        self.assertAlmostEqual(r.aggregate, 77.75, 2)

    def test_mdcat_worked_example(self):
        r = F.calculate(F.PROFILES["mdcat_pmdc"], F.MeritInput(1000, 1100, 1000, 1100, 160, 180))
        self.assertAlmostEqual(r.aggregate, 89.90, 2)

    def test_all_profiles_sum_to_100_and_variants_too(self):
        for p in F.PROFILES.values():
            self.assertEqual(round(sum(p.weights.values())), 100)
            for v in p.variants:
                self.assertEqual(round(sum(v.weights.values())), 100, p.key)

    def test_validation(self):
        p = F.PROFILES["mdcat_pmdc"]
        with self.assertRaises(F.MeritError):
            F.calculate(p, F.MeritInput(1200, 1100, 900, 1100, 100, 180))
        with self.assertRaises(F.MeritError):
            F.calculate(p, F.MeritInput(900, 0, 900, 1100, 100, 180))
        with self.assertRaises(F.MeritError):
            F.calculate(p, F.MeritInput(900, 1100, 900, 1100, 100, 180), {F.SSC: 10, F.HSSC: 10, F.TEST: 10})

    def test_eligibility_floor_and_variant_spread(self):
        res = F.calculate(F.PROFILES["nust_net"], F.MeritInput(500, 1000, 550, 1000, 100, 200))
        self.assertEqual(len(res.eligibility_issues), 2)
        allv = F.calculate_all_variants(F.PROFILES["nums_mbbs"], F.MeritInput(1000, 1100, 800, 1100, 150, 200))
        lo, hi = F.sensitivity_spread(allv)
        self.assertGreater(len(allv), 1)
        self.assertNotEqual(lo, hi)


class EvidenceTests(unittest.TestCase):
    PAGE = "UHS merit 2024: closing merit for MBBS open seats was 93.5% at the last list.\nOther text."

    def test_verify(self):
        q = "closing merit for MBBS open seats was 93.5%"
        self.assertTrue(evidence.verify_evidence(93.5, q, self.PAGE))
        self.assertFalse(evidence.verify_evidence(94.5, q, self.PAGE))                      # number not in quote
        self.assertFalse(evidence.verify_evidence(93.5, "closing merit was 93.5%", self.PAGE))  # quote not on page
        self.assertFalse(evidence.verify_evidence(93.5, "", self.PAGE))

    def test_tiers(self):
        self.assertEqual(evidence.trust_tier("https://admission.uet.edu.pk/x"), 1)
        self.assertEqual(evidence.trust_tier("https://www.hec.gov.pk/a"), 1)
        self.assertEqual(evidence.trust_tier("https://someblog.com/mdcat"), 3)

    def test_json_and_backoff(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Here you go: {"a": [1,2]} thanks'), {"a": [1, 2]})
        self.assertIsNone(extract_json("no json"))
        self.assertTrue(is_rate_limit(Exception("Error 429 rate limit")))
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise Exception("429 Too Many Requests")
            return "ok"
        with mock.patch("time.sleep"):
            self.assertEqual(with_backoff(flaky, wait=0.01), "ok")


class WebToolTests(unittest.TestCase):
    def test_ssrf_guard_and_parsing(self):
        self.assertEqual(web_tools.fetch_page("file:///etc/passwd")[1], "Only http(s) URLs are allowed.")
        self.assertIn("Refused", web_tools.fetch_page("http://127.0.0.1:8000/")[1])
        self.assertIn("Refused", web_tools.fetch_page("http://169.254.169.254/latest/meta-data")[1])
        text = web_tools._html_to_text("<table><tr><td>MBBS</td><td>94.5</td></tr></table><script>bad()</script>")
        self.assertIn("MBBS | 94.5", text)
        self.assertNotIn("bad()", text)


class _TmpDbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._p = mock.patch.object(db, "DB_PATH", Path(self.tmp.name) / "t.db")
        self._p.start()
        history.init_tables()

    def tearDown(self):
        self._p.stop()
        self.tmp.cleanup()


class HistoryTests(_TmpDbCase):
    def test_csv_import_and_bands(self):
        csv_bytes = (b"exam,year,institution,program,closing_merit\n"
                     b"MDCAT,2024,Alpha MC,MBBS,90.0\nMDCAT,2023,Alpha MC,MBBS,88.5\n"
                     b"MDCAT,2024,Beta MC,MBBS,86.0\nMDCAT,2024,Gamma MC,MBBS,99.9\nMDCAT,2024,Bad,MBBS,150\n")
        rows, errs = history.parse_csv(csv_bytes)
        self.assertEqual(errs, [])
        self.assertEqual(history.add_rows("s1", rows), 4)          # 150 rejected
        self.assertEqual(history.parse_csv(b"a,b\n1,2\n")[1][0][:7], "Missing")
        recs = history.recommend(89.0, history.get_rows("s1", "MDCAT"))
        bands = {r.institution: r.band for r in recs}
        self.assertEqual(bands, {"Beta MC": "Safe", "Alpha MC": "Target", "Gamma MC": "Unlikely"})
        alpha = next(r for r in recs if r.institution == "Alpha MC")
        self.assertIn("rose 1.50", alpha.trend_note)
        self.assertEqual(history.get_rows("other_student"), [])    # per-student isolation


class MeritCrewFlowTests(_TmpDbCase):
    URL = "https://www.uhs.edu.pk/merit2024"
    PAGE = "UHS 2024. Closing merit for MBBS at Alpha MC was 91.2% on open merit.  Weightage: MDCAT 50%, FSc 40%, Matric 10%."

    def fake_run_stage(self, **kw):
        web_tools.PAGE_CACHE[self.URL] = self.PAGE           # what read_webpage would have cached
        role = kw["role"]
        if role == "Admission Formula Verifier":
            return json.dumps({"found": True, "weights_percent": {"ssc": 10, "hssc": 40, "test": 50}, "test_total": 180,
                               "source_url": self.URL, "quote": "Weightage: MDCAT 50%, FSc 40%, Matric 10%", "conflicts": ""})
        if role == "Admission Merit History Scout":
            return json.dumps({"rows": [
                {"year": 2024, "institution": "Alpha MC", "program": "MBBS", "category": "open merit", "closing_merit": 91.2,
                 "source_url": self.URL, "quote": "Closing merit for MBBS at Alpha MC was 91.2%"},
                {"year": 2024, "institution": "Alpha MC", "program": "MBBS", "category": "open merit", "closing_merit": 95.0,   # hallucinated
                 "source_url": self.URL, "quote": "Closing merit for MBBS at Alpha MC was 95.0%"}]})
        return "Advice text."

    def test_end_to_end_with_hallucination_rejected(self):
        with mock.patch("crew_agents.runner.run_stage", side_effect=lambda **kw: self.fake_run_stage(**kw)), \
             mock.patch.object(merit_crew, "_tools", return_value=(None, None, None, None)):
            rep = merit_crew.run_merit(profile_key="mdcat_pmdc", data=F.MeritInput(1000, 1100, 1000, 1100, 160, 180), student_id="s1",
                                       institution="Alpha MC", program="MBBS")
        self.assertAlmostEqual(rep.primary.aggregate, 89.90, 2)
        self.assertTrue(rep.formula_check["verified"] and rep.formula_check["matches_profile"])
        self.assertEqual(len(rep.recommendations), 1)
        self.assertEqual(rep.recommendations[0].closing_merit, 91.2)
        self.assertEqual(len(rep.rejected_rows), 1)                    # the invented 95.0 never reaches the student
        self.assertEqual(rep.recommendations[0].band, "Reach")
        self.assertEqual(rep.advice, "Advice text.")

    def test_calculator_survives_agent_failure(self):
        with mock.patch("crew_agents.runner.run_stage", side_effect=RuntimeError("boom")), \
             mock.patch.object(merit_crew, "_tools", return_value=(None, None, None, None)):
            rep = merit_crew.run_merit(profile_key="nust_net", data=F.MeritInput(800, 1000, 900, 1000, 150, 200), student_id="s1",
                                       institution="NUST")
        self.assertAlmostEqual(rep.primary.aggregate, 77.75, 2)
        self.assertTrue(rep.log)


class PathfinderTests(unittest.TestCase):
    URL = "https://www.pec.org.pk/accredited"
    PAGE = "Accredited programs: BS Software Engineering at Real University is accredited by PEC."

    def test_requirements_unverified_means_not_checked_and_accreditation_downgraded(self):
        profile = pf_crew.StudentProfile(hssc_pct=80, ssc_pct=85, stream="Pre-Engineering", subjects=["Maths", "Physics"])

        def fake(**kw):
            web_tools.PAGE_CACHE[self.URL] = self.PAGE
            if kw["role"] == "Accreditation Verifier":
                good = "Fake University" not in kw["description"]
                return json.dumps({"hec_recognised": "yes", "council": "PEC", "council_status": "accredited", "source_url": self.URL,
                                   "quote": ("BS Software Engineering at Real University is accredited by PEC" if good else "Fake University is accredited by PEC")})
            return "{}"
        state = {"programs": {"programs": [
            {"institution": "Real University", "program": "BS Software Engineering", "evidence": "verified",
             "requirements": {"min_hssc_pct": 60, "required_subjects": ["Physics"]}},
            {"institution": "Fake University", "program": "BS Fake", "evidence": "unverified", "requirements": {"min_hssc_pct": 50}}]}}
        with mock.patch("crew_agents.runner.run_stage", side_effect=fake), mock.patch.object(pf_crew, "_tools", return_value=[]):
            out = pf_crew.run_pathfinder_stage("verification", profile, state)["verification"]
        real, fake_u = out
        self.assertTrue(real["eligibility"]["overall"].startswith("ELIGIBLE"))
        self.assertEqual(real["accreditation"]["council_status"], "accredited")
        self.assertTrue(fake_u["eligibility"]["overall"].startswith("NOT CHECKED"))
        self.assertEqual(fake_u["accreditation"]["council_status"], "not_found")      # claimed 'accredited' but quote not on page
        self.assertEqual(fake_u["accreditation"]["hec_recognised"], "not_found")

    def test_eligibility_fail_and_unknown(self):
        r = eligibility.check({"hssc_pct": 55}, {"min_hssc_pct": 60, "min_ssc_pct": 60})
        self.assertEqual(r["overall"], "NOT ELIGIBLE")
        r = eligibility.check({}, {"min_hssc_pct": 60})
        self.assertTrue(r["overall"].startswith("INCOMPLETE"))
        self.assertEqual(eligibility.check({}, {})["overall"], "NO REQUIREMENTS PROVIDED")


if __name__ == "__main__":
    unittest.main()
