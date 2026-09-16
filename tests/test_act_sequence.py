"""Run with: python -m unittest discover -s tests -v (no robot dependencies)."""

import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_act_sequence import MockBackend, main
from sequence_core import SequenceController, Stage, checked_action, hold_action, validate_features
from sequence_lerobot import LeRobotBackend


def stages():
    return [Stage("08", "p08", "d08", "task08", 5), Stage("09", "p09", "d09", "task09", 5)]


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.backend = MockBackend(stages())
        self.engine = SequenceController(stages(), self.backend, trials=2)
        self.engine.connect()

    def test_handoff_preserves_pose_and_grasp_without_reconnect_or_teleop(self):
        self.engine.tick("s", 0)
        self.engine.tick(None, 0.1)
        last_pose = self.backend.read()
        self.engine.tick("n", 0.2)
        self.assertEqual(self.engine.phase, "handoff")
        self.assertEqual(self.backend.state["x.vel"], 0)
        self.engine.tick("t", 0.3)  # Cannot enter teleop between skills.
        self.engine.tick("s", 0.4)
        first09 = next(obs for i, obs in self.backend.inputs if i == 1)
        for name in self.backend.action_names:
            if name.endswith(".pos"):
                self.assertEqual(first09[name], last_pose[name])
        self.assertEqual(self.backend.connects, 1)
        self.assertEqual(self.backend.disconnects, 0)
        self.assertEqual(self.backend.teleop_calls, 0)
        self.assertEqual(self.backend.resets, [0, 1])

    def test_two_trials_keep_one_connection_and_one_episode_per_pair(self):
        for t in (0, 1):
            for cmd, offset in (("s", 0), ("n", .1), ("s", .2), ("n", .3)):
                self.engine.tick(cmd, t + offset)
        self.assertEqual(self.engine.phase, "done")
        self.assertEqual(len(self.backend.results), 2)
        self.assertTrue(all(len(r["stages"]) == 2 for r in self.backend.results))
        self.engine.close()
        self.assertEqual((self.backend.connects, self.backend.disconnects), (1, 1))

    def test_timeout_is_failure_not_automatic_next_stage(self):
        self.engine.tick("s", 0)
        self.engine.tick(None, 6)
        self.assertEqual(self.backend.results[0]["status"], "timeout")
        self.assertFalse(any(i == 1 for i, _ in self.backend.inputs))
        self.assertEqual(self.backend.state["x.vel"], 0)

    def test_failure_does_not_execute_second_skill(self):
        self.engine.tick("s", 0)
        self.engine.tick("f", .1)
        self.assertEqual(self.backend.results[0]["status"], "operator_failure")
        self.assertEqual(self.backend.resets, [0])

    def test_pause_discards_old_chunk_on_resume(self):
        self.engine.tick("s", 0)
        self.backend.queues[0].append("old action")
        self.engine.tick(" ", .1)
        count = len(self.backend.inputs)
        self.engine.tick(None, .2)
        self.assertEqual(len(self.backend.inputs), count)
        self.assertEqual(self.backend.state["x.vel"], 0)
        self.engine.tick("s", .3)
        self.assertFalse(self.backend.queues[0])
        self.assertEqual(self.backend.resets, [0, 0])

    def test_quit_holds_and_does_not_disconnect_until_close(self):
        self.engine.tick("s", 0)
        self.engine.tick("q", .1)
        self.assertEqual(self.engine.phase, "done")
        self.assertEqual(self.backend.results[0]["status"], "aborted")
        self.assertEqual(self.backend.disconnects, 0)
        self.assertEqual(self.backend.state["x.vel"], 0)

    def test_pause_does_not_restart_timeout_budget(self):
        self.engine.tick("s", 0)
        self.engine.tick(" ", 4)
        self.engine.tick("s", 100)
        self.engine.tick(None, 102)
        self.assertEqual(self.backend.results[0]["status"], "timeout")

    def test_shutdown_after_camera_failure_still_stops_base(self):
        self.engine.tick("s", 0)
        def failed_read():
            raise OSError("camera disconnected")
        self.backend.read = failed_read
        with self.assertRaises(OSError):
            self.engine.tick(None, .1)
        self.engine.abort("camera disconnected")
        self.engine.close()
        self.assertEqual(self.backend.state["x.vel"], 0)
        self.assertEqual(self.backend.disconnects, 1)

    def test_stop_during_inference_does_not_send_predicted_motion(self):
        original_predict = self.backend.predict
        def predict(*args):
            action = original_predict(*args)
            self.backend.stop_requested = lambda: True
            return action
        self.backend.predict = predict
        self.engine.tick("s", 0)
        self.assertEqual(self.engine.phase, "done")
        self.assertEqual(self.backend.state["left_joint_0.pos"], 0)

    def test_teleop_is_explicit_and_ready_only(self):
        self.engine.tick(None, 0)
        self.assertEqual(self.backend.teleop_calls, 0)
        self.engine.tick("t", .1)
        self.assertEqual(self.backend.teleop_calls, 1)
        self.engine.tick("s", .2)
        self.engine.tick("t", .3)
        self.assertEqual(self.backend.teleop_calls, 1)

    def test_hold_retains_gripper_target_not_measured_gap(self):
        obs = self.backend.read()
        name = "left_left_carriage_joint.pos"
        command = hold_action(obs, {name: .005}, self.backend.action_names)
        self.assertEqual(command[name], .005)

    def test_non_finite_action_rejected_before_send(self):
        self.backend.predict = lambda *_: {**self.backend.state, "x.vel": float("nan")}
        with self.assertRaises(ValueError):
            self.engine.tick("s", 0)
        self.engine.abort("nonfinite")
        self.assertEqual(self.backend.state["x.vel"], 0)
        self.assertEqual(self.engine.phase, "done")

    def test_incomplete_action_rejected(self):
        with self.assertRaises(ValueError):
            checked_action({"x.vel": 0}, self.backend.action_names)

    def test_reconnect_rejected(self):
        with self.assertRaises(RuntimeError):
            self.engine.connect()


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.features = {
            "observation.state": {"dtype": "float32", "shape": [14], "names": [f"j{i}" for i in range(14)]},
            "action": {"dtype": "float32", "shape": [16], "names": [f"j{i}" for i in range(14)] + ["x.vel", "theta.vel"]},
            "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
        }
        self.info = {"fps": 30, "features": copy.deepcopy(self.features)}
        self.config = {"type": "act", "input_features": {
            "observation.state": {"shape": [14]}, "observation.images.cam_high": {"shape": [3, 480, 640]},
        }, "output_features": {"action": {"shape": [16]}}}

    def test_asymmetric_14_16_accepted(self):
        validate_features(self.config, self.info, self.features, 30)

    def test_wrong_channel_order_rejected_even_when_shapes_match(self):
        self.info["features"]["action"]["names"].reverse()
        with self.assertRaises(ValueError):
            validate_features(self.config, self.info, self.features, 30)

    def test_wrong_state_dim_rejected(self):
        self.config["input_features"]["observation.state"]["shape"] = [16]
        with self.assertRaises(ValueError):
            validate_features(self.config, self.info, self.features, 30)

    def test_wrong_camera_resolution_rejected(self):
        self.info["features"]["observation.images.cam_high"]["shape"] = [240, 320, 3]
        with self.assertRaises(ValueError):
            validate_features(self.config, self.info, self.features, 30)

    def test_wrong_fps_rejected(self):
        with self.assertRaises(ValueError):
            validate_features(self.config, self.info, self.features, 20)


class DriverCleanupTests(unittest.TestCase):
    def test_real_disconnect_method_respects_parking_flag(self):
        # Execute the actual method body with a fake driver: no robotics imports.
        path = ROOT / "packages/lerobot_robot_trossen/src/lerobot_robot_trossen/widowxai_follower.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WidowXAIFollower")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "disconnect")
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        namespace = {"DeviceNotConnectedError": RuntimeError, "logger": SimpleNamespace(info=lambda *_: None)}
        exec(compile(module, str(path), "exec"), namespace)
        for park, expected_moves in ((False, 0), (True, 2)):
            calls = []
            obj = SimpleNamespace(
                is_connected=True,
                config=SimpleNamespace(park_on_disconnect=park, staged_positions=[1]*7, joint_names=list(range(7))),
                driver=SimpleNamespace(set_all_positions=lambda *a, **k: calls.append("move"), cleanup=lambda: calls.append("cleanup")),
                cameras={},
            )
            namespace["disconnect"](obj)
            self.assertEqual(calls.count("move"), expected_moves)
            self.assertEqual(calls[-1], "cleanup")

    def test_cli_mock_mode_needs_no_robot_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--dry-run", "--output", str(Path(tmp)/"run")]), 0)
            self.assertTrue((Path(tmp)/"run/dry_run.json").is_file())


class AdapterTests(unittest.TestCase):
    def test_each_stage_uses_its_own_policy_and_processors(self):
        backend = LeRobotBackend.__new__(LeRobotBackend)
        backend.stages = stages()
        backend.policies = [(object(), object(), object()), (object(), object(), object())]
        backend.device = "cuda"
        backend.robot = SimpleNamespace(robot_type="mobileai_robot")
        backend.features = {"action": {"names": ["x.vel"]}}
        backend.action_names = ["x.vel"]
        seen = []
        def predict_action(**kwargs):
            seen.append(kwargs)
            return {"x.vel": 0.0}
        # Substitute dependency boundaries, exercising the actual adapter method.
        modules = {
            "lerobot.datasets.utils": SimpleNamespace(build_dataset_frame=lambda *args: args[1]),
            "lerobot.policies.utils": SimpleNamespace(make_robot_action=lambda action, _: action),
            "lerobot.utils.control_utils": SimpleNamespace(predict_action=predict_action),
        }
        with patch.dict(sys.modules, modules):
            for index in (0, 1):
                backend.predict(index, {"state": "current physical observation"})
        for i, call in enumerate(seen):
            self.assertIs(call["policy"], backend.policies[i][0])
            self.assertIs(call["preprocessor"], backend.policies[i][1])
            self.assertIs(call["postprocessor"], backend.policies[i][2])
            self.assertEqual(call["task"], backend.stages[i].task)

    def test_reset_clears_only_selected_policy_and_processors(self):
        backend = LeRobotBackend.__new__(LeRobotBackend)
        calls = []
        backend.policies = [
            tuple(SimpleNamespace(reset=lambda i=i, j=j: calls.append((i, j))) for j in range(3))
            for i in range(2)
        ]
        backend.reset_policy(1)
        self.assertEqual(calls, [(1, 0), (1, 1), (1, 2)])

    def test_failed_episode_save_is_not_retried_during_cleanup(self):
        backend = LeRobotBackend.__new__(LeRobotBackend)
        calls = []
        def fail_save(**kwargs):
            calls.append("save")
            raise OSError("disk full")
        backend.recording_failed = False
        backend.active = True
        backend.frame_index = 4
        backend.dataset = SimpleNamespace(save_episode=fail_save)
        backend.event = lambda *args, **kwargs: None
        with self.assertRaises(OSError):
            backend.finish_trial({"status": "operator_success"})
        with self.assertRaises(RuntimeError):
            backend.finish_trial({"status": "aborted"})
        self.assertFalse(backend.active)
        self.assertEqual(calls, ["save"])


if __name__ == "__main__":
    unittest.main()
