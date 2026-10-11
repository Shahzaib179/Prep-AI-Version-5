"""Study plans from chapters: parsing, searching the material, validating the AI plan, and the no-AI fallback."""
import pathlib, sys, unittest
from datetime import date
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import study_plan_sources as sps

def passage(text, score=0.8, filename="bio.pdf", page=3): return {"text": text, "hybrid_score": score, "filename": filename, "page": page}
DB = {"Genetics": [passage("Genes are inherited in Mendelian ratios. " * 6, 0.9, page=12), passage("Alleles segregate during meiosis. " * 5, 0.7, page=13)],
      "Cell cycle": [passage("Mitosis has prophase metaphase anaphase telophase. " * 6, 0.8, page=40)],
      "Quantum": [passage("unrelated text", 0.05)]}
def search(q, k): return DB.get(q, [])[:k]

class Parse(unittest.TestCase):
    def test_parse_chapters(self):
        self.assertEqual(sps.parse_chapters("Genetics; Cell cycle\n - Biomolecules , genetics ,, "), ["Genetics", "Cell cycle", "Biomolecules"])
        self.assertEqual(sps.parse_chapters(""), []); self.assertEqual(len(sps.parse_chapters(",".join(f"c{i}" for i in range(20)))), sps.MAX_CHAPTERS)
        self.assertEqual(len(sps.parse_chapters("x" * 300)[0]), 80)
    def test_plan_days(self):
        t = date(2026, 10, 1)
        self.assertEqual(sps.plan_days(7, "2026-12-01", t), 7); self.assertEqual(sps.plan_days(7, "2026-10-04", t), 3)
        self.assertEqual(sps.plan_days(7, "2026-10-01", t), 7); self.assertEqual(sps.plan_days(7, "2026-09-01", t), 7)   # exam today/past: keep the request
        self.assertEqual(sps.plan_days(99, "", t), 14); self.assertEqual(sps.plan_days(0, "bad", t), 1)

class Gather(unittest.TestCase):
    def test_found_missing_and_labels(self):
        m = sps.gather(["Genetics", "Cell cycle", "Quantum", "Nothing"], search)
        self.assertEqual([b["chapter"] for b in m["blocks"]], ["Genetics", "Cell cycle"]); self.assertEqual(m["missing"], ["Quantum", "Nothing"])
        self.assertIn("### CHAPTER: Genetics", m["context"]); self.assertIn("[bio.pdf p.12]", m["context"]); self.assertEqual(m["labels"], ["bio.pdf p.12", "bio.pdf p.13", "bio.pdf p.40"])
    def test_budget_is_respected(self):
        big = {"A": [passage("word " * 2000, 0.9, page=i) for i in range(5)]}
        m = sps.gather(["A"], lambda q, k: big["A"][:k], char_budget=1500)
        self.assertLess(len(m["context"]), 2200)

class Normalize(unittest.TestCase):
    LABELS = ["bio.pdf p.12"]
    def test_fits_time_budget_and_drops_unknown_sources(self):
        data = {"days": [{"day": "Day 1", "sessions": [{"subject": "Biology", "topic": "Genetics - alleles", "minutes": 100, "activity": "Read", "source": "bio.pdf p.12"},
                                                        {"topic": "Genetics - ratios", "minutes": "x", "source": "invented.pdf p.1"}, {"topic": "  "}, "junk"]}, {"sessions": []}, "junk"]}
        p = sps.normalize_plan(data, 7, 1.0, self.LABELS, "Biology")
        self.assertEqual(len(p), 1); s = p[0]["sessions"]; self.assertEqual(len(s), 2)
        self.assertEqual(s[0]["source"], "bio.pdf p.12"); self.assertEqual(s[1]["source"], ""); self.assertEqual(s[1]["subject"], "Biology")
        self.assertTrue(55 <= sum(x["minutes"] for x in s) <= 65 and all(x["minutes"] % 5 == 0 for x in s))
    def test_accepts_list_and_limits_days(self):
        one = {"day": "D", "sessions": [{"topic": "t", "minutes": 30}]}
        self.assertEqual(len(sps.normalize_plan([one] * 10, 3, 1, [], "S")), 3)
        self.assertEqual(sps.normalize_plan("garbage", 3, 1, [], "S"), []); self.assertEqual(sps.normalize_plan(None, 3, 1, [], "S"), [])

class Fallback(unittest.TestCase):
    def test_every_chapter_is_learned_practised_and_revised(self):
        ch = ["Genetics", "Cell cycle", "Biomolecules"]
        p = sps.fallback_plan(ch, 7, 1.5, "Biology", {"Genetics": "bio.pdf p.12"})
        self.assertEqual(len(p), 7)
        learn = {s["topic"] for d in p for s in d["sessions"] if s["activity"] == sps.ACTIVITIES[0]}; self.assertEqual(learn, set(ch))
        self.assertTrue(any(s["activity"] == sps.ACTIVITIES[1] for d in p for s in d["sessions"]))
        self.assertEqual({s["topic"] for s in p[-1]["sessions"]}, set(ch)); self.assertTrue(all(s["activity"] == sps.ACTIVITIES[2] for s in p[-1]["sessions"]))
        for d in p: self.assertTrue(80 <= sum(s["minutes"] for s in d["sessions"]) <= 100, d)
        self.assertEqual(p[0]["sessions"][0]["source"], "bio.pdf p.12")
    def test_tiny_plans(self):
        for days in (1, 2, 3):
            p = sps.fallback_plan(["A", "B"], days, 1, "S"); self.assertEqual(len(p), days); self.assertTrue(all(d["sessions"] for d in p))
        self.assertEqual(len(sps.fallback_plan([f"c{i}" for i in range(8)], 14, 1, "S")[-1]["sessions"]), sps.MAX_SESSIONS_PER_DAY)

class Make(unittest.TestCase):
    def mat(self): return sps.gather(["Genetics", "Cell cycle", "Quantum"], search)
    def good(self, prompt):
        self.prompt = prompt
        return {"days": [{"day": f"Day {i}", "sessions": [{"subject": "Biology", "topic": "Genetics - alleles", "minutes": 30, "activity": "Read", "source": "bio.pdf p.12"}]} for i in range(1, 8)]}
    def test_ai_plan_is_used_and_prompt_is_grounded(self):
        plan, used = sps.make_plan(["Genetics", "Cell cycle", "Quantum"], self.mat(), 7, 1.0, "MDCAT", "2026-12-01", "Biology", "score 90", "Biology - Genetics", self.good)
        self.assertTrue(used and len(plan) == 7)
        for must in ("Genes are inherited", "MDCAT", "score 90", "Biology - Genetics", "7-day", "Genetics, Cell cycle"): self.assertIn(must, self.prompt)
        self.assertNotIn("Quantum, ", self.prompt); self.assertNotIn("unrelated text", self.prompt)
    def test_falls_back_when_ai_fails_or_returns_junk(self):
        def boom(_): raise RuntimeError("rate limit")
        for ai in (boom, lambda _: {"days": []}, lambda _: "nonsense", lambda _: {"days": [{"sessions": [{"topic": "x"}]}]}):
            plan, used = sps.make_plan(["Genetics", "Cell cycle"], self.mat(), 7, 1.0, "E", "d", "Biology", "", "", ai)
            self.assertFalse(used); self.assertEqual(len(plan), 7); self.assertNotIn("Quantum", str(plan))
    def test_nothing_found(self):
        with self.assertRaises(ValueError): sps.make_plan(["Quantum"], sps.gather(["Quantum"], search), 7, 1, "E", "d", "S", "", "", self.good)

if __name__ == "__main__": unittest.main()
