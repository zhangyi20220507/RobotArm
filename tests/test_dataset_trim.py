"""Episode-specific trimming, review persistence, and real RGB export regression."""
import json
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from robotarm.modeling.dataset_browser import DatasetStore, make_handler
from robotarm.modeling.dataset_trim import export_dataset, export_plan, frame_bounds, trim_from_seconds


class TrimRangeTests(unittest.TestCase):
    def test_seconds_are_half_open_and_frame_aligned(self):
        trim = trim_from_seconds(5, 14, 400, 20)
        self.assertEqual(frame_bounds(trim, 400, 20), (100, 280))
        self.assertEqual(frame_bounds(trim_from_seconds(.051, .201, 10, 20), 10, 20), (2, 5))
        self.assertIsNone(trim_from_seconds(0, 20, 400, 20))

    def test_invalid_ranges(self):
        for start, end in [(-1, 2), (2, 1), (1, 1), (0, 21), (float("nan"), 1),
                           (0, float("inf")), (.001, .002), (True, 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                trim_from_seconds(start, end, 400, 20)
        trim = trim_from_seconds(5, 14, 400, 20)
        with self.assertRaisesRegex(ValueError, "长度/FPS"):
            frame_bounds(trim, 401, 20)

    def test_each_episode_has_its_own_interval(self):
        info = {"fps": 20}
        rows = [{"episode_index": i, "length": 400} for i in range(4)]
        reviews = {"0": {"trim": trim_from_seconds(5, 14, 400, 20)},
                   "1": {"trim": trim_from_seconds(2, 8, 400, 20)},
                   "2": {"status": "reject"}}
        plan = export_plan(info, rows, reviews)
        self.assertEqual([x["frames"] for x in plan], [180, 120, 400])
        self.assertEqual([x["source_episode"] for x in plan], [0, 1, 3])
        self.assertEqual([x["output_episode"] for x in plan], [0, 1, 2])


class BrowserTrimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        (self.root / "meta").mkdir(parents=True)
        (self.root / "meta/info.json").write_text(json.dumps({"total_episodes": 2, "fps": 20}))
        self.reviews = self.root.parent / "review.json"
        rows = [{"episode_index": i, "length": 400} for i in range(2)]
        with patch.object(DatasetStore, "_load_episode_rows", return_value=rows):
            self.store = DatasetStore(self.root, self.reviews)

    def test_save_reload_change_status_and_clear_trim(self):
        self.store.save_review(0, "keep", "first", {"start_s": 5, "end_s": 14})
        self.store.save_review(1, "", "", {"start_s": 2, "end_s": 8})
        self.store.save_review(0, "review", "status only")  # old clients must not erase trim
        reviews = self.store.reviews()
        self.assertEqual(reviews["0"]["trim"]["start_frame"], 100)
        self.assertEqual(reviews["1"]["trim"]["end_frame"], 160)
        self.store.save_review(0, "keep", "", None)
        self.assertNotIn("trim", self.store.reviews()["0"])
        self.store.save_review(1, "", "", {"start_s": 0, "end_s": 20})
        self.assertNotIn("1", self.store.reviews())

    def test_bad_range_does_not_change_saved_file(self):
        self.store.save_review(0, "keep", "", {"start_s": 5, "end_s": 14})
        original = self.reviews.read_bytes()
        with self.assertRaises(ValueError):
            self.store.save_review(0, "reject", "", {"start_s": 16, "end_s": 14})
        self.assertEqual(self.reviews.read_bytes(), original)

    def test_foreign_reviews_are_not_overwritten(self):
        self.reviews.write_text(json.dumps({"dataset_root": str(self.root.parent / "other"), "episodes": {}}))
        with self.assertRaisesRegex(ValueError, "another dataset"):
            self.store.save_review(0, "keep", "")

    def test_export_path_guards_and_single_job(self):
        for name in ("../other", "/tmp/other", ".", "..", "source", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.start_export(name)
        with patch("robotarm.modeling.dataset_browser.threading.Thread") as thread:
            job = self.store.start_export("trimmed")
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["total_episodes"], 2)
            self.assertEqual(job["episodes_done"], 0)
            self.assertIsNone(job["current_episode"])
            thread.return_value.start.assert_called_once()
            with self.assertRaisesRegex(ValueError, "正在运行"):
                self.store.start_export("trimmed2")

    def test_export_failure_is_reported(self):
        with patch("robotarm.modeling.dataset_browser.threading.Thread") as thread:
            self.store.start_export("trimmed")
            run = thread.call_args.kwargs["target"]
        with patch("robotarm.modeling.dataset_browser.export_dataset", side_effect=RuntimeError("test failure")):
            run()
        self.assertEqual(self.store.export_status()["status"], "failed")
        self.assertIn("test failure", self.store.export_status()["error"])

    def test_export_progress_counts_selected_episodes_and_preserves_failure(self):
        self.store.save_review(0, "reject", "")
        with patch("robotarm.modeling.dataset_browser.threading.Thread") as thread:
            initial = self.store.start_export("trimmed")
            run = thread.call_args.kwargs["target"]
        self.assertEqual(initial["total_episodes"], 1)
        self.assertEqual(initial["total_frames"], 400)

        def fail(source, output, info, plan, reviews, progress):
            progress({"episodes_done": 1, "frames_done": 400, "phase": "生成统计与来源清单"})
            self.assertEqual(self.store.export_status()["status"], "running")
            raise RuntimeError("finalize failed")

        with patch("robotarm.modeling.dataset_browser.export_dataset", side_effect=fail):
            run()
        job = self.store.export_status()
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["episodes_done"], 1)
        self.assertEqual(initial["episodes_done"], 0)  # independent snapshots

    def test_export_completion_and_new_job_reset(self):
        with patch("robotarm.modeling.dataset_browser.threading.Thread") as thread:
            self.store.start_export("trimmed")
            run = thread.call_args.kwargs["target"]
            with patch("robotarm.modeling.dataset_browser.export_dataset", return_value={"episodes": 2, "frames": 800}):
                run()
            job = self.store.export_status()
            self.assertEqual((job["status"], job["episodes_done"], job["frames_done"]), ("complete", 2, 800))
            new_job = self.store.start_export("next")
            self.assertEqual((new_job["status"], new_job["episodes_done"], new_job["frames_done"]), ("running", 0, 0))

    def test_frontend_javascript_syntax(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        html = Path(__file__).resolve().parents[1] / "src/robotarm/modeling/dataset_browser.html"
        script = html.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
        result = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frontend_range_controls_use_frame_boundaries(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        html = Path(__file__).resolve().parents[1] / "src/robotarm/modeling/dataset_browser.html"
        script = html.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
        harness = """
const vm=require('node:vm'), assert=require('node:assert/strict');
const elements=new Map();
function element(id){
  if(!elements.has(id))elements.set(id,{value:'',textContent:'',classList:{toggle(){}},matches(){return false},setAttribute(){}});
  return elements.get(id);
}
const sandbox={document:{documentElement:{},querySelector:element,querySelectorAll:()=>[],addEventListener(){}},
  window:{addEventListener(){}},fetch:()=>new Promise(()=>{}),requestAnimationFrame(){},
  setTimeout,clearTimeout,confirm:()=>true};
vm.createContext(sandbox);
vm.runInContext(SCRIPT,sandbox);
vm.runInContext("meta={fps:20};current={length:400,duration:20};"+
  "$('#trimStart').value='5';$('#trimEnd').value='14';",sandbox);
let range=JSON.parse(vm.runInContext("JSON.stringify(bounds())",sandbox));
assert.equal(range.first,100);assert.equal(range.stop,280);
vm.runInContext("time=13.95;$('#markEnd').onclick()",sandbox);
assert.equal(element('#trimEnd').value,14);
vm.runInContext("$('#trimReset').onclick()",sandbox);
assert.equal(element('#trimStart').value,0);assert.equal(element('#trimEnd').value,20);
vm.runInContext("$('#trimStart').value='15';$('#trimEnd').value='14'",sandbox);
assert.throws(()=>vm.runInContext('bounds()',sandbox));
"""
        result = subprocess.run(["node"], input=harness.replace("SCRIPT", json.dumps(script)),
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_http_save_and_reject_invalid_or_cross_origin_changes(self):
        # Exercise the real HTTP parser/handler over memory streams, no listening port.
        def request(method, path, data=None, origin=None):
            body = json.dumps(data).encode() if data is not None else b""
            headers = (f"{method} {path} HTTP/1.0\r\nHost: 127.0.0.1:8766\r\n"
                       f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n")
            if origin:
                headers += f"Origin: {origin}\r\n"
            sock, output = Mock(), bytearray()
            sock.makefile.return_value = io.BytesIO(headers.encode() + b"\r\n" + body)
            sock.sendall.side_effect = output.extend
            handler = make_handler(self.store)
            with patch.object(handler, "log_message"):
                handler(sock, ("127.0.0.1", 0), Mock())
            head, payload = bytes(output).split(b"\r\n\r\n", 1)
            return int(head.split()[1]), json.loads(payload)

        data = {"episode": 0, "status": "keep", "trim": {"start_s": 5, "end_s": 14}}
        status, result = request("POST", "/api/review", data)
        self.assertEqual(status, 200)
        self.assertEqual(result["0"]["trim"]["end_frame"], 280)
        self.assertEqual(request("GET", "/api/meta")[1]["reviews"]["0"]["trim"]["start_frame"], 100)
        self.assertEqual(request("GET", "/api/export")[1]["status"], "idle")
        before = self.reviews.read_bytes()
        self.assertEqual(request("POST", "/api/review", data, "http://untrusted.example")[0], 400)
        data["trim"]["end_s"] = 4
        self.assertEqual(request("POST", "/api/review", data)[0], 400)
        self.assertEqual(self.reviews.read_bytes(), before)


class RealTrimExportTests(unittest.TestCase):
    def test_two_camera_export_preserves_actions_and_rebuilds_timeline(self):
        try:
            import numpy as np
            from lerobot.configs.video import RGBEncoderConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            self.skipTest(f"LeRobot export dependencies unavailable: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            self.enterContext(patch("datasets.config.HF_DATASETS_CACHE", Path(directory) / "hf-cache"))
            root = Path(directory) / "source"
            features = {
                "observation.state": {"dtype": "float32", "shape": (7,), "names": {"motors": [f"j{i}" for i in range(7)]}},
                "action": {"dtype": "float32", "shape": (7,), "names": {"motors": [f"j{i}" for i in range(7)]}},
                **{f"observation.images.{role}": {
                    "dtype": "video", "shape": (24, 32, 3), "names": ["height", "width", "channel"]
                } for role in ("global", "wrist")},
            }
            source = LeRobotDataset.create(
                repo_id="local/source", root=root, fps=20, features=features, use_videos=True,
                encoder_threads=1, rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast"))
            for episode in range(2):
                for index in range(10):
                    value = episode * 100 + index
                    source.add_frame({
                        "observation.state": np.full(7, value, dtype=np.float32),
                        "action": np.full(7, value + .5, dtype=np.float32),
                        "observation.images.global": np.full((24, 32, 3), index * 20, dtype=np.uint8),
                        "observation.images.wrist": np.full((24, 32, 3), 240 - index * 20, dtype=np.uint8),
                        "task": f"task {episode}",
                    })
                source.save_episode(parallel_encoding=False)
            source.finalize()
            (root / "meta/piper_conversion.json").write_text(json.dumps({
                "representation": {"space": "joint", "frame": "absolute"}, "repo_id": "local/source"}))
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            store = DatasetStore(root, Path(directory) / "reviews.json")
            store.save_review(0, "keep", "", {"start_s": .1, "end_s": .3})
            store.save_review(1, "", "", {"start_s": .05, "end_s": .2})
            plan = export_plan(store.info, store.episodes, store.reviews())
            output = root.parent / "trimmed"
            updates = []
            result = export_dataset(root, output, store.info, plan, store.reviews(), updates.append)
            snapshot = {}
            encoding_counts = []
            for update in updates:
                snapshot.update(update)
                if update.get("phase") == "编码视频":
                    encoding_counts.append(snapshot["episodes_done"])
            self.assertEqual(encoding_counts, [0, 1])
            self.assertEqual(snapshot["episodes_done"], 2)
            self.assertEqual(snapshot["total_episodes"], 2)
            self.assertEqual(snapshot["frames_done"], 7)
            self.assertIsNone(snapshot["current_episode"])
            self.assertEqual([u["current_episode"] for u in updates if u.get("episode_position")], [0, 1])
            self.assertEqual(result["frames"], 7)
            self.assertFalse((output / "TRIM_INCOMPLETE.txt").exists())
            self.assertTrue((output / "meta/stats.json").exists())
            stats = json.loads((output / "meta/stats.json").read_text())
            self.assertAlmostEqual(stats["action"]["min"][0], 2.5)
            self.assertAlmostEqual(stats["action"]["max"][0], 103.5)
            derived = json.loads((output / "meta/piper_conversion.json").read_text())
            self.assertEqual(derived["representation"]["frame"], "absolute")
            for relative, original in before.items():
                self.assertEqual((root / relative).read_bytes(), original, relative)
            loaded = LeRobotDataset(repo_id="local/trimmed", root=output, return_uint8=True)
            for i, value in enumerate((2, 3, 4, 5, 101, 102, 103)):
                frame = loaded[i]
                self.assertAlmostEqual(float(frame["action"][0]), value + .5)
                self.assertAlmostEqual(float(frame["timestamp"]), (i if i < 4 else i - 4) / 20, places=6)
                self.assertEqual(frame["task"], f"task {0 if i < 4 else 1}")
                original_index = value % 100
                self.assertAlmostEqual(float(frame["observation.images.global"].float().mean()), original_index * 20, delta=6)
                self.assertAlmostEqual(float(frame["observation.images.wrist"].float().mean()), 240-original_index*20, delta=6)
            windowed = LeRobotDataset(repo_id="local/trimmed", root=output,
                                     delta_timestamps={"action": [i / 20 for i in range(5)]})
            last = windowed[3]
            self.assertTrue(bool(last["action_is_pad"][1:].all()))
            self.assertTrue(bool((last["action"][:, 0] == 5.5).all()))
            with self.assertRaises(ValueError):
                export_dataset(root, output, store.info, plan, store.reviews())
            incomplete = root.parent / "failed"
            with patch.object(LeRobotDataset, "add_frame", side_effect=RuntimeError("encoding failed")):
                with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                    export_dataset(root, incomplete, store.info, plan, store.reviews())
            self.assertTrue((incomplete / "TRIM_INCOMPLETE.txt").is_file())
            # Frames copied is not equivalent to an episode saved successfully.
            updates = []
            with patch.object(LeRobotDataset, "save_episode", side_effect=RuntimeError("save failed")):
                with self.assertRaisesRegex(RuntimeError, "save failed"):
                    export_dataset(root, root.parent / "save_failed", store.info, plan, store.reviews(), updates.append)
            self.assertEqual(max(u.get("episodes_done", 0) for u in updates), 0)
            self.assertEqual(updates[-1]["phase"], "编码视频")
            self.assertEqual(updates[-1]["frames_done"], 4)


if __name__ == "__main__":
    unittest.main()
