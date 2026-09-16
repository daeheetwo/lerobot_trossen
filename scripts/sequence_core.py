"""Robot-independent controller for manually supervised ACT skill chaining."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Stage:
    name: str
    policy: str
    dataset: str
    task: str
    timeout_s: float = 120.0


def checked_action(action, names):
    if set(action) != set(names):
        raise ValueError("Action channel names do not match the robot.")
    result = {name: float(action[name]) for name in names}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite action; no command was sent.")
    return result


def hold_action(observation, previous, names):
    """Hold measured arm positions, retain gripper targets, stop the base."""
    action = {}
    for name in names:
        if name in ("x.vel", "theta.vel"):
            action[name] = 0.0
        elif "carriage_joint" in name or "gripper" in name:
            action[name] = previous.get(name, observation[name])
        else:
            action[name] = observation[name]
    return checked_action(action, names)


class SequenceController:
    """One connection; no teleoperation, parking or reconnect between stages.

    Backend methods perform hardware I/O. This class is also used by --dry-run
    and unit tests, so those exercise the actual transition logic.
    """

    def __init__(self, stages, backend, trials=10):
        if len(stages) < 2 or trials < 1:
            raise ValueError("At least two stages and one trial are required.")
        if len({s.name for s in stages}) != len(stages):
            raise ValueError("Stage names must be unique.")
        if any(not math.isfinite(s.timeout_s) or s.timeout_s <= 0 for s in stages):
            raise ValueError("Stage timeouts must be finite and positive.")
        self.stages = stages
        self.backend = backend
        self.trials = trials
        self.trial = 0
        self.index = 0
        self.phase = "ready"
        self.teleop_enabled = False
        self.last_action = {}
        self.hold = None
        self.started_at = None
        self.elapsed_s = 0.0
        self.active_trial = False
        self.results = []
        self.connected = False

    def connect(self):
        if self.connected:
            raise RuntimeError("Controller already connected.")
        self.backend.connect()
        self.connected = True
        self.backend.event("ready", trial=0, phase=self.phase)

    def _event(self, name, **extra):
        self.backend.event(
            name, trial=self.trial, stage=self.stages[self.index].name,
            phase=self.phase, **extra,
        )

    def _hold(self, observation):
        self.hold = hold_action(observation, self.last_action, self.backend.action_names)
        # Stop immediately, before policy reset, disk encoding or operator waits.
        self.last_action = self.backend.send(self.hold)

    def _start_stage(self, now, resume=False):
        self.backend.reset_policy(self.index)
        if not resume:
            self.elapsed_s = 0.0
        self.started_at = now
        self.phase = "running"
        self._event("stage_started")

    def _finish(self, status):
        result = {
            "trial": self.trial, "status": status,
            "stages": list(self.results), "manual_transition": True,
        }
        # A whole sequence (including pauses) is one recorded episode.
        self.backend.finish_trial(result)
        self.active_trial = False
        self.phase = "done" if status == "aborted" or self.trial >= self.trials else "ready"
        self.index = 0
        self.teleop_enabled = False
        self._event("trial_finished", result=result)

    def tick(self, command, now):
        if not self.connected:
            raise RuntimeError("Connect before ticking.")
        observation = self.backend.read()
        if self.hold is None:
            self.hold = hold_action(observation, self.last_action, self.backend.action_names)

        if command == "q":
            self._hold(observation)
            if self.active_trial:
                self.results.append({"stage": self.stages[self.index].name, "status": "aborted"})
                self._finish("aborted")
            self.phase = "done"
            self._event("stop_requested")
        elif command == "t" and self.phase == "ready":
            self.teleop_enabled = not self.teleop_enabled
            self._hold(observation)
            self._event("teleop_toggled", enabled=self.teleop_enabled)
        elif command == "s" and self.phase in ("ready", "handoff", "paused"):
            resume = self.phase == "paused"
            self._hold(observation)
            if self.phase == "ready":
                self.trial += 1
                self.index = 0
                self.results = []
                self.backend.begin_trial(self.trial)
                self.active_trial = True
            self.teleop_enabled = False
            self._start_stage(now, resume=resume)
        elif command == " " and self.phase == "running":
            self._hold(observation)
            self.elapsed_s += now - self.started_at
            self.phase = "paused"
            self._event("paused")
        elif command == "n" and self.phase == "running":
            self._hold(observation)
            self.results.append({"stage": self.stages[self.index].name, "status": "operator_success"})
            self._event("stage_completed", decision="operator")
            if self.index + 1 == len(self.stages):
                self._finish("operator_success")
            else:
                self.index += 1
                self.phase = "handoff"
                self._event("handoff_wait")
        elif command == "f" and self.active_trial:
            self._hold(observation)
            self.results.append({"stage": self.stages[self.index].name, "status": "operator_failure"})
            self._finish("operator_failure")

        if self.phase == "running":
            if self.elapsed_s + now - self.started_at >= self.stages[self.index].timeout_s:
                self._hold(observation)
                self.results.append({"stage": self.stages[self.index].name, "status": "timeout"})
                self._finish("timeout")
                return
            action = self.backend.predict(self.index, observation)
        elif self.phase == "ready" and self.teleop_enabled:
            action = self.backend.teleop(observation)
            # Leader base velocities are observed passive movement, not commands.
            action = {**action, "x.vel": 0.0, "theta.vel": 0.0}
        else:
            action = self.hold

        # Prediction can take time. Check a latched stop again before sending.
        if self.backend.stop_requested():
            self._hold(observation)
            self.abort("stop_during_inference")
            return
        action = checked_action(action, self.backend.action_names)
        self.last_action = self.backend.send(action)
        if self.active_trial:
            self.backend.record(observation, self.last_action, self.index, self.phase)
        self.backend.display(observation, self.last_action)

    def abort(self, reason):
        self.backend.stop(self.last_action)
        self._event("aborted", reason=reason)
        if self.active_trial:
            self.results.append({"stage": self.stages[self.index].name, "status": "aborted"})
            self._finish("aborted")
        self.phase = "done"

    def close(self, park=False):
        self.backend.stop(self.last_action)
        self.backend.close(park=park)
        self.connected = False


def validate_features(config, info, robot_features, fps):
    """Check dimensions AND channel order before any physical connection."""
    if config.get("type") != "act":
        raise ValueError("This runner currently supports ACT checkpoints only.")
    if info["fps"] != fps:
        raise ValueError(f"Dataset FPS {info['fps']} differs from runner FPS {fps}.")
    for name in ("observation.state", "action"):
        expected = robot_features[name]
        feature = info["features"][name]
        if list(feature["shape"]) != list(expected["shape"]) or feature["names"] != expected["names"]:
            raise ValueError(f"Dataset/robot channel shape or order mismatch: {name}")
    expected_inputs = {k for k in robot_features if k.startswith("observation.")}
    if set(config["input_features"]) != expected_inputs or set(config["output_features"]) != {"action"}:
        raise ValueError("Checkpoint input/output feature names differ from robot.")
    for key, feature in robot_features.items():
        if not (key.startswith("observation.") or key == "action"):
            continue
        shape = list(feature["shape"])
        if feature["dtype"] in ("video", "image"):
            if list(info["features"][key]["shape"]) != shape:
                raise ValueError(f"Dataset camera shape mismatch: {key}")
            shape = [shape[2], shape[0], shape[1]]
        group = "output_features" if key == "action" else "input_features"
        if list(config[group][key]["shape"]) != shape:
            raise ValueError(f"Checkpoint shape mismatch: {key}")
