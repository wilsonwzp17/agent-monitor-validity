"""Tests for the ARB data path on a synthetic fixture (ARB's terms forbid committing its data).

The fixture plants canary strings where leaks would come from: the rule-based reward
(summary_info), the agent's own prompts (chat_messages), the agent and model names, and a
test-split label row. Run with ``python3 -m unittest discover -s tests`` or ``pytest``.
"""

import csv
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import struct
import unittest
import urllib.request
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datapath import constants as C  # noqa: E402
from datapath import fetch  # noqa: E402
from datapath import labels as labels_mod  # noqa: E402
from datapath import leakage, validate  # noqa: E402
from datapath.ids import Splits, normalize_task_id, trace_id, trace_key  # noqa: E402
from datapath.normalize import ending_type, normalize, serialize  # noqa: E402
from datapath.png import inspect_png, make_png  # noqa: E402

AGENT = "GenericAgent-fake-model-2025"
MODEL = "fake-model-2025"
CANARY_REWARD = "CANARY_REWARD_19c"
CANARY_CHAT = "CANARY_CHAT_55a"
CANARY_TEST_LABEL = "CANARY_TEST_LABEL_7f3"
HEADER = ["annotator_name", "benchmark", "task_id", "model_name", "exp_name",
          "trajectory_success", "trajectory_side_effect", "trajectory_optimality",
          "trajectory_looping"]


def _trajectory(bench, task_id, n_actions=2, last_action="click('12')"):
    steps = []
    for i in range(n_actions + 1):
        action = None if i == n_actions else (last_action if i == n_actions - 1 else f"click('{i}')")
        steps.append({
            "num": i,
            "reasoning": None if action is None else f"step {i} reasoning",
            "action": action,
            "screenshot_path": f"trajectories/screenshots/{bench}/{AGENT}/{task_id}/screenshot_step_{i}.png",
            "url": f"https://wa-shopping-xl-2.mcgill-nlp.org/page{i}",
            "open_pages_urls": [f"https://wa-shopping-xl-2.mcgill-nlp.org/page{i}"],
            "focused_element": "12",
            "last_action_error": "",
            "stats": {"n_token_agent_messages": 10},
            "axtree": f"RootWebArea 'Shop page {i}'",
            "axtree_obj": {"nodes": []},
            "chat_messages": [{"role": "system", "content": f"You are {MODEL}. {CANARY_CHAT}"}],
            "bounding_boxes": {"12": [0, 0, 10, 10]},
            "extra_element_properties": {"12": {"visibility": 1.0, "bbox": [0, 0, 10, 10],
                                                "clickable": True, "set_of_marks": False}},
            "axtree_pruned": f"[12] link 'Item {i}'",
        })
    return {
        "benchmark": bench, "agent": AGENT, "model": MODEL, "valid": True,
        "experiment": f"{AGENT}_on_{bench}", "goal": "Find the cheapest red mug.",
        "seed": 0, "model_args": {"model_name": MODEL}, "flags": {},
        "summary_info": {"cum_reward": 1.0, "note": CANARY_REWARD, "terminated": True},
        "package_version": "browsergym==0.13.3", "steps": steps,
    }


def listing_entry(raw: Path, rel: str) -> dict:
    """A listing entry as Hugging Face reports it: LFS SHA-256, or Git blob SHA-1 for data/."""
    data = (raw / rel).read_bytes()
    entry = {"type": "file", "path": rel, "size": len(data),
             "oid": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
    if not rel.startswith("data/"):
        entry["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
    return entry


def build_fixture(root: Path) -> dict:
    """Write a miniature ARB layout under root/raw plus pinned listings under root/listing."""
    raw = root / "raw"
    files = {}

    def put(rel, data: bytes):
        f = raw / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        files[rel] = len(data)

    put("data/splits.csv", b"task_id,benchmark,split\nwebarena.1,webarena,dev\n"
                           b"webarena.2,webarena,test\nvisualwebarena.3,visualwebarena,dev\n"
                           b"visualwebarena.4,visualwebarena,test\n")
    rows = [
        ["a1", "webarena", "webarena.1", AGENT, "x", "Successful", "No", "4. Completely Optimal", "No"],
        ["a2", "webarena", "webarena.1", AGENT, "x", "Unsuccessful", "Yes", "2. Suboptimal", "Yes"],
        ["a1", "webarena", "webarena.2", AGENT, "x", CANARY_TEST_LABEL, CANARY_TEST_LABEL,
         CANARY_TEST_LABEL, CANARY_TEST_LABEL],
        ["a3", "visualwebarena", "visualwebarena.resized.3", AGENT, "x", "Unsure", "No", "Unsure", "No"],
        ["a2", "visualwebarena", "visualwebarena.resized.4", AGENT, "x", CANARY_TEST_LABEL,
         CANARY_TEST_LABEL, CANARY_TEST_LABEL, CANARY_TEST_LABEL],
    ]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(rows)
    put(C.LABEL_FILE, buf.getvalue().encode())
    for meta in C.META_FILES:
        if meta not in files:
            put(meta, b"[]\n" if meta.endswith(".json") else b"task_name\n")

    trajs = {}
    for bench, tid, kw in (("webarena", "webarena.1", {}),
                           ("webarena", "webarena.2", {"last_action": "send_msg_to_user('3')"}),
                           ("visualwebarena", "visualwebarena.resized.3", {"n_actions": 1})):
        t = _trajectory(bench, tid, **kw)
        rel = f"cleaned/{bench}/{AGENT}/{AGENT}_on_{bench}/{tid}.json"
        put(rel, json.dumps(t).encode())
        trajs[tid] = (bench, rel, t)
        for step in t["steps"]:
            put(step["screenshot_path"].split("trajectories/", 1)[1], make_png(4, 3))

    listing = root / "listing" / C.REVISION
    listing.mkdir(parents=True, exist_ok=True)
    for top in ("data", "cleaned", "screenshots"):
        (listing / f"{top}.json").write_text(json.dumps(
            [listing_entry(raw, p) for p in files if p.startswith(top + "/")]))
    return {"raw": raw, "trajs": trajs}


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = build_fixture(self.tmp)
        self.raw = self.fx["raw"]
        self.splits = Splits(self.raw / "data/splits.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    TEST_RUNS = (("webarena", "webarena.2"), ("visualwebarena", "visualwebarena.resized.4"))

    def make_lock(self, runs=TEST_RUNS) -> Path:
        pred = self.tmp / "pred.jsonl"
        pred.write_text("".join(json.dumps({"trace_key": trace_key(trace_id(b, AGENT, t)),
                                            "verdict": "fail"}) + "\n" for b, t in runs))
        lock = self.tmp / "lock.json"
        lock.write_text(json.dumps({"predictions_path": "pred.jsonl",
                                    "predictions_sha256": hashlib.sha256(pred.read_bytes()).hexdigest()}))
        return lock

    def norm(self, tid="webarena.1"):
        bench, rel, t = self.fx["trajs"][tid]
        return normalize(json.loads((self.raw / rel).read_text()), self.raw, tid,
                         self.splits.split(tid), self.splits.group(tid),
                         source_record={"path": rel})


class TestNormalize(FixtureCase):
    def test_monitor_view_has_no_canaries_or_identity(self):
        view, prov = self.norm()
        text = serialize(view)
        for s in (CANARY_REWARD, CANARY_CHAT, AGENT, MODEL, "webarena.1", "summary_info",
                  "cum_reward", "chat_messages", "screenshots/", "-xl-2", "sha256"):
            self.assertNotIn(s, text)
        self.assertIn("https://wa-shopping.mcgill-nlp.org/page0", text)
        self.assertEqual(prov["hostname_suffixes_rewritten"], 3 * 2)   # url + open page, 3 steps
        self.assertEqual(view["trace_key"], trace_key(prov["trace_id"]))
        self.assertEqual(prov["agent"], AGENT)

    def test_event_order_observations_and_fields_kept(self):
        view, prov = self.norm()
        self.assertEqual([e["step"] for e in view["events"]], [0, 1, 2])
        self.assertEqual(view["events"][0]["observation"]["axtree"], "RootWebArea 'Shop page 0'")
        shot = view["events"][0]["observation"]["screenshot"]
        self.assertEqual(shot, {"width": 4, "height": 3, "png_ok": True, "png_error": None})
        self.assertEqual(prov["screenshots_present"], 3)
        self.assertEqual(len(prov["screenshot_files"][0]["sha256"]), 64)
        self.assertEqual(prov["source_url"], None)
        self.assertIn("summary_info", prov["dropped_top_level_fields"])
        self.assertIn("chat_messages", prov["dropped_step_fields"])
        self.assertEqual(prov["unrecognized_top_level_fields"], [])

    def test_unknown_source_field_is_dropped_and_reported(self):
        bench, rel, t = self.fx["trajs"]["webarena.1"]
        t = json.loads(json.dumps(t))
        t["new_field"] = "x"
        t["steps"][0]["new_step_field"] = "y"
        view, prov = normalize(t, self.raw, "webarena.1", "dev", "webarena")
        self.assertNotIn("new_field", serialize(view))
        self.assertNotIn("new_step_field", serialize(view))
        self.assertEqual(prov["unrecognized_top_level_fields"], ["new_field"])
        self.assertEqual(prov["unrecognized_step_fields"], ["new_step_field"])

    def test_unverified_screenshot_is_treated_as_missing(self):
        bench, rel, t = self.fx["trajs"]["webarena.1"]
        view, prov = normalize(t, self.raw, "webarena.1", "dev", "webarena", verified_paths=set())
        self.assertEqual(prov["screenshots_present"], 0)
        self.assertEqual(view["events"][0]["observation"]["screenshot"], {"missing": True})

    def test_task_id_must_match_screenshot_directory(self):
        bench, rel, t = self.fx["trajs"]["webarena.1"]
        with self.assertRaises(ValueError):
            normalize(t, self.raw, "webarena.9", "dev", "webarena")

    def test_deterministic_bytes(self):
        a, pa = self.norm()
        b, pb = self.norm()
        self.assertEqual(serialize(a), serialize(b))
        self.assertEqual(serialize(pa), serialize(pb))

    def test_ending_type(self):
        self.assertEqual(self.norm("webarena.1")[1]["ending"], "environment")
        self.assertIsNone(self.norm("webarena.2")[1]["ending"])      # test split: withheld
        msg = [{"action": "click('1')"}, {"action": "send_msg_to_user('3')"}, {"action": None}]
        self.assertEqual(ending_type(msg), "agent_message")
        capped = [{"action": f"click('{i}')"} for i in range(C.STEP_CAP)] + [{"action": None}]
        self.assertEqual(ending_type(capped), "step_cap")


class TestLeakage(FixtureCase):
    def test_clean_record_passes(self):
        view, prov = self.norm()
        bench, rel, t = self.fx["trajs"]["webarena.1"]
        self.assertTrue(leakage.check_record(view, leakage.identity_strings(t, prov))["ok"])

    def test_injections_are_caught(self):
        view, prov = self.norm()
        ids = [AGENT, MODEL, prov["trace_id"], prov["task_id"]]
        cases = {
            "L0_unexpected_keys": lambda v: v["events"][0]["observation"].update(foo=1),
            "L1_forbidden_keys": lambda v: v["events"][1].update(cum_reward=1.0),
            "L2_forbidden_substrings": lambda v: v["task"].update(goal="trajectory_success: yes"),
            "L3_identifiers": lambda v: v["events"][0].update(reasoning=f"I am {MODEL}"),
        }
        cases_l3 = [lambda v: v["events"][0].update(url="https://vwa-forum-xl-1.mcgill-nlp.org/"),
                    lambda v: v["events"][0].update(url="https://h/?r=https%3A%2F%2Fwa%2Dshopping%2Dxl%2D2.mcgill-nlp.org"),
                    lambda v: v["events"][0].update(reasoning="WA-FORUM-XL-2.MCGILL-NLP.ORG"),
                    lambda v: v["task"].update(goal="see https://h/static/input_images/task_1/input_0.png")]
        for inject in cases_l3:
            v = json.loads(serialize(view))
            inject(v)
            self.assertTrue(leakage.check_record(v, ids + ["task_1/"])["L3_identifiers"])
        self.assertIn("task_3/", leakage.identity_strings({}, {"task_id": "visualwebarena.resized.3"}))
        self.assertIn("visualwebarena.3", leakage.identity_strings({}, {"task_id": "visualwebarena.resized.3"}))
        for check, inject in cases.items():
            v = json.loads(serialize(view))
            inject(v)
            result = leakage.check_record(v, ids)
            self.assertFalse(result["ok"], check)
            self.assertTrue(result[check], check)

    def test_element_ids_are_data_not_field_names(self):
        view, _ = self.norm()
        view["events"][0]["elements"]["label"] = {"visibility": 1.0}
        self.assertTrue(leakage.check_record(view)["ok"])
        view["events"][0]["elements"]["label"]["reward"] = 1
        self.assertFalse(leakage.check_record(view)["ok"])

    def test_runtime(self):
        r = leakage.check_runtime()
        self.assertTrue(r["ok"], r)


class TestLabels(FixtureCase):
    def test_dev_only_and_no_test_canary(self):
        store = labels_mod.load_labels(self.raw / C.LABEL_FILE, self.splits)
        self.assertEqual({r["split"] for r in store.values()}, {"dev"})
        self.assertNotIn(CANARY_TEST_LABEL, json.dumps(store))
        self.assertEqual(len(store), 2)

    def test_test_rows_skipped_before_label_columns_are_read(self):
        accessed = []

        class Spy(dict):
            def __getitem__(self, k):
                accessed.append((dict.get(self, "task_id"), k))
                return dict.__getitem__(self, k)

            def get(self, k, default=None):
                accessed.append((dict.get(self, "task_id"), k))
                return dict.get(self, k, default)

            def values(self):
                accessed.append((dict.get(self, "task_id"), "*values"))
                return dict.values(self)

            def items(self):
                accessed.append((dict.get(self, "task_id"), "*items"))
                return dict.items(self)

        real = csv.DictReader

        def spy_reader(*a, **kw):
            return (Spy(r) for r in real(*a, **kw))

        with mock.patch.object(labels_mod.csv, "DictReader", spy_reader):
            labels_mod.load_labels(self.raw / C.LABEL_FILE, self.splits)
        for test_task in ("webarena.2", "visualwebarena.resized.4"):
            test_keys = {k for tid, k in accessed if tid == test_task}
            self.assertEqual(test_keys, {"task_id"}, test_task)

    def test_test_split_needs_lock(self):
        f = self.raw / C.LABEL_FILE
        with self.assertRaises(labels_mod.LockedSplitError):
            labels_mod.load_labels(f, self.splits, allowed_splits=("dev", "test"))
        lock = self.tmp / "lock.json"
        lock.write_text(json.dumps({"predictions_sha256": "abc"}))
        with self.assertRaises(labels_mod.LockedSplitError):          # no predictions file named
            labels_mod.load_labels(f, self.splits, allowed_splits=("test",), lock_file=lock)
        (self.tmp / "pred.jsonl").write_text("{}\n")
        lock.write_text(json.dumps({"predictions_path": "pred.jsonl", "predictions_sha256": "abc"}))
        with self.assertRaises(labels_mod.LockedSplitError):          # hash does not match
            labels_mod.load_labels(f, self.splits, allowed_splits=("test",), lock_file=lock)
        (self.tmp / "pred.jsonl").write_text("{}\n")
        lock.write_text(json.dumps({"predictions_path": "pred.jsonl", "predictions_sha256":
                                    hashlib.sha256(b"{}\n").hexdigest()}))
        with self.assertRaises(labels_mod.LockedSplitError):          # prediction without a key
            labels_mod.load_labels(f, self.splits, allowed_splits=("test",), lock_file=lock)
        self.make_lock()
        store = labels_mod.load_labels(f, self.splits, allowed_splits=("test",), lock_file=lock)
        self.assertEqual(len(store), 2)

    def test_lock_unlocks_only_predicted_runs(self):
        lock = self.make_lock(runs=(("webarena", "webarena.2"),))
        store = labels_mod.load_labels(self.raw / C.LABEL_FILE, self.splits,
                                       allowed_splits=("test",), lock_file=lock)
        self.assertEqual([r["task_id_normalized"] for r in store.values()], ["webarena.2"])

    def test_primary_row_is_first_in_file_order(self):
        store = labels_mod.load_labels(self.raw / C.LABEL_FILE, self.splits)
        rec = store[trace_key(trace_id("webarena", AGENT, "webarena.1"))]
        self.assertEqual(rec["trajectory_success"], "Successful")
        self.assertEqual(rec["n_annotation_rows"], 2)
        self.assertEqual(rec["success_binary"], 1)
        self.assertEqual(rec["primary_annotator_name"], "a1")
        self.assertEqual([(r["annotator_name"], r["trajectory_success"]) for r in rec["secondary_labels"]],
                         [("a2", "Unsuccessful")])

    def test_label_values_validated(self):
        store = labels_mod.load_labels(self.raw / C.LABEL_FILE, self.splits,
                                       allowed_splits=("test",), lock_file=self.make_lock())
        result = validate.validate_labels(store, set(store))
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["bad_values"]), 8)

    def test_annotation_structure_is_label_free(self):
        s = labels_mod.annotation_structure(self.raw / C.LABEL_FILE, self.splits)
        self.assertEqual((s["rows"], s["trajectories"], s["trajectories_with_extra_rows"]), (5, 4, 1))
        self.assertNotIn(CANARY_TEST_LABEL, json.dumps(s))


class TestValidate(FixtureCase):
    def test_good_record(self):
        view, _ = self.norm()
        self.assertTrue(validate.validate_record(view)["ok"])

    def test_corrupt_png(self):
        bench, rel, t = self.fx["trajs"]["webarena.1"]
        png = self.raw / t["steps"][1]["screenshot_path"].split("trajectories/", 1)[1]
        data = bytearray(png.read_bytes())
        data[20] ^= 0xFF
        png.write_bytes(bytes(data))
        r = validate.validate_record(self.norm()[0])
        self.assertFalse(r["checks"]["V4_screenshots"]["ok"])
        self.assertIn("CRC", r["checks"]["V4_screenshots"]["problems"][0][1])

    def test_missing_screenshot(self):
        bench, rel, t = self.fx["trajs"]["webarena.1"]
        (self.raw / t["steps"][0]["screenshot_path"].split("trajectories/", 1)[1]).unlink()
        view, prov = self.norm()
        self.assertFalse(validate.validate_record(view)["checks"]["V4_screenshots"]["ok"])
        self.assertEqual(prov["screenshots_present"], 2)

    def test_event_order_and_actions(self):
        view, _ = self.norm()
        view["events"][1]["step"] = 5
        self.assertFalse(validate.validate_record(view)["checks"]["V2_event_order"]["ok"])
        view, _ = self.norm()
        view["events"][0]["action"] = None
        self.assertFalse(validate.validate_record(view)["checks"]["V3_actions"]["ok"])

    def test_empty_goal(self):
        view, _ = self.norm()
        view["task"]["goal"] = " "
        self.assertFalse(validate.validate_record(view)["checks"]["V1_required"]["ok"])


class TestIdsAndPng(unittest.TestCase):
    def test_normalize_task_id(self):
        self.assertEqual(normalize_task_id("visualwebarena.resized.118"), "visualwebarena.118")
        self.assertEqual(normalize_task_id("assistantbench.improved.validation.3"),
                         "assistantbench.validation.3")

    def test_split_conflict_raises(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.csv"
            f.write_text("task_id,benchmark,split\nvisualwebarena.1,v,dev\n"
                         "visualwebarena.resized.1,v,test\n")
            with self.assertRaises(ValueError):
                Splits(f)

    def test_png(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.png"
            f.write_bytes(make_png(7, 5))
            r = inspect_png(f)
            self.assertEqual((r["ok"], r["width"], r["height"]), (True, 7, 5))
            f.write_bytes(make_png(7, 5)[:-6])
            self.assertFalse(inspect_png(f)["ok"])
            f.write_bytes(b"not a png")
            self.assertEqual(inspect_png(f)["error"], "bad signature")
            f.write_bytes(make_png(7, 5) + b"junk")
            self.assertEqual(inspect_png(f)["error"], "data after IEND")
            self.assertEqual(inspect_png(self._png(ihdr=b"\x00" * 4))["error"], "IHDR is not 13 bytes")
            self.assertEqual(inspect_png(self._png(w=0, h=0))["error"], "zero width or height")
            self.assertIn("does not decompress", inspect_png(self._png(idat=b"not zlib"))["error"])
            self.assertIn("expected", inspect_png(self._png(idat=zlib.compress(b"\x00" * 3)))["error"])

    def _png(self, w=2, h=2, ihdr=None, idat=None):
        def chunk(t, b):
            return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
        ihdr = ihdr if ihdr is not None else struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        idat = idat if idat is not None else zlib.compress(b"\x00" + b"\x01" * 3 * w)
        f = Path(tempfile.mkdtemp()) / "x.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))
        return f


class TestRunnerOffline(FixtureCase):
    def _load_runner(self):
        spec = importlib.util.spec_from_file_location("arb_datapath", REPO / "scripts/arb_datapath.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_end_to_end_and_rerun_identical(self):
        runner = self._load_runner()
        specs = [f"webarena/{AGENT}/webarena.1", f"webarena/{AGENT}/webarena.2",
                 f"visualwebarena/{AGENT}/visualwebarena.resized.3"]
        outs = []
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")):
            for name in ("o1", "o2"):
                out = self.tmp / name
                args = SimpleNamespace(data_root=str(self.tmp), out=str(out), trajectory=specs)
                with mock.patch("sys.stdout", new=io.StringIO()):
                    self.assertEqual(runner.run(args), 0)
                outs.append(out)
        for f in ("monitor_view.jsonl", "provenance.jsonl", "labels/labels_dev.jsonl"):
            self.assertEqual((outs[0] / f).read_bytes(), (outs[1] / f).read_bytes(), f)
        mv = (outs[0] / "monitor_view.jsonl").read_text()
        self.assertEqual(len(mv.splitlines()), 3)
        for s in (CANARY_REWARD, CANARY_CHAT, CANARY_TEST_LABEL, AGENT, MODEL, "-xl-2"):
            self.assertNotIn(s, mv)
        for f in ("provenance.jsonl", "labels/labels_dev.jsonl", "report.json"):
            text = (outs[0] / f).read_text()
            for s in (CANARY_REWARD, CANARY_CHAT, CANARY_TEST_LABEL):
                self.assertNotIn(s, text, f)
        labels = [json.loads(x) for x in (outs[0] / "labels/labels_dev.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(r["split"] for r in labels), ["dev", "dev"])
        provs = [json.loads(x) for x in (outs[0] / "provenance.jsonl").read_text().splitlines()]
        for prov in provs:      # run events stay out of provenance, so reruns are identical
            self.assertNotIn("downloaded", prov["file"])
            self.assertNotIn("redownloaded_after_failed_check", prov["file"])
        self.assertEqual({p["split"]: p["ending"] is None for p in provs}, {"dev": False, "test": True})
        report = json.loads((outs[0] / "report.json").read_text())
        self.assertTrue(report["ok"])
        test_rows = [r for r in report["summary"] if r["split"] == "test"]
        self.assertEqual(len(test_rows), 1)
        for k in ("n_events", "n_actions", "ending", "screenshots"):
            self.assertNotIn(k, test_rows[0])
        self.assertEqual(report["inputs"]["files_redownloaded_after_failed_check"], 0)
        self.assertEqual(report["inputs"]["files_downloaded"], 0)
        self.assertEqual(report["inputs"]["listings_read_from_cache"], ["data", "cleaned", "screenshots"])

    def test_refuses_paths_inside_repo(self):
        runner = self._load_runner()
        args = SimpleNamespace(data_root=str(self.tmp), out=str(REPO / "out_x"),
                               trajectory=[f"webarena/{AGENT}/webarena.1"])
        with self.assertRaises(SystemExit):
            runner.run(args)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestFetch(FixtureCase):
    def setUp(self):
        super().setUp()
        self.rel = self.fx["trajs"]["webarena.1"][1]
        self.good = (self.raw / self.rel).read_bytes()
        self.entry = listing_entry(self.raw, self.rel)

    def test_good_cached_file_needs_no_network(self):
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")):
            rec = fetch.fetch_file(self.rel, self.raw, self.entry)
        self.assertEqual(rec["checked_against_listing"], "lfs-sha256")
        self.assertFalse(rec["redownloaded_after_failed_check"])
        meta = fetch.fetch_file("data/splits.csv", self.raw, listing_entry(self.raw, "data/splits.csv"))
        self.assertEqual(meta["checked_against_listing"], "git-blob-sha1")

    def test_same_size_corruption_is_redownloaded(self):
        bad = self.good.replace(b"cheapest", b"priciest")
        self.assertEqual(len(bad), len(self.good))
        (self.raw / self.rel).write_bytes(bad)
        with mock.patch.object(urllib.request, "urlopen", return_value=_FakeResponse(self.good)):
            rec = fetch.fetch_file(self.rel, self.raw, self.entry)
        self.assertTrue(rec["redownloaded_after_failed_check"])
        self.assertEqual((self.raw / self.rel).read_bytes(), self.good)

    def test_same_size_corruption_of_a_git_blob_file_is_redownloaded(self):
        rel = "data/splits.csv"
        good = (self.raw / rel).read_bytes()
        entry = listing_entry(self.raw, rel)
        self.assertNotIn("lfs", entry)
        bad = good.replace(b"dev", b"tst", 1)
        self.assertEqual(len(bad), len(good))
        (self.raw / rel).write_bytes(bad)
        with mock.patch.object(urllib.request, "urlopen", return_value=_FakeResponse(good)):
            rec = fetch.fetch_file(rel, self.raw, entry)
        self.assertEqual(rec["checked_against_listing"], "git-blob-sha1")
        self.assertTrue(rec["redownloaded_after_failed_check"] and rec["downloaded"])
        self.assertEqual((self.raw / rel).read_bytes(), good)

    def test_bad_download_is_never_moved_into_place(self):
        (self.raw / self.rel).unlink()
        with mock.patch.object(urllib.request, "urlopen", return_value=_FakeResponse(self.good[:100])):
            with self.assertRaises(IOError):
                fetch.fetch_file(self.rel, self.raw, self.entry)
        self.assertFalse((self.raw / self.rel).exists())
        self.assertFalse((self.raw / (self.rel + ".part")).exists())
        with mock.patch.object(urllib.request, "urlopen", return_value=_FakeResponse(self.good)):
            rec = fetch.fetch_file(self.rel, self.raw, self.entry)
        self.assertTrue(rec["downloaded"])
        self.assertFalse(rec["redownloaded_after_failed_check"])
        self.assertEqual((self.raw / self.rel).read_bytes(), self.good)


if __name__ == "__main__":
    unittest.main()
