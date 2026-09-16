#!/usr/bin/env python3
"""Manual ACT sequence evaluation; --dry-run needs only Python's standard library."""

import argparse
from collections import deque
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import signal
import sys
import threading
import time

from sequence_core import SequenceController, Stage


class TerminalKeys:
    """Read only the focused terminal, not global desktop hotkeys."""
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Real evaluation requires an interactive terminal.")
        self.saved = None
        if sys.platform != "win32":
            import termios
            import tty
            self.saved = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self):
        if sys.platform == "win32":
            import msvcrt
            return msvcrt.getwch().lower() if msvcrt.kbhit() else None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            # TextIO can read ahead, hiding queued keys from select(). Read the
            # terminal file descriptor directly so a queued stop isn't delayed.
            import os
            value = os.read(sys.stdin.fileno(), 1)
            if not value:
                raise EOFError("Control terminal disconnected.")
            return value.decode("ascii", errors="ignore").lower() or None
        return None

    def __exit__(self, *exc):
        if self.saved is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)


class MockBackend:
    """Small deterministic robot; it never imports robot/CUDA/network packages."""
    def __init__(self, stages):
        self.stages = stages
        self.action_names = [
            f"{side}_joint_{i}.pos" for side in ("left", "right") for i in range(6)
        ] + ["left_left_carriage_joint.pos", "right_left_carriage_joint.pos", "x.vel", "theta.vel"]
        self.state = dict.fromkeys(self.action_names, 0.0)
        self.state["left_left_carriage_joint.pos"] = 0.015
        self.state["right_left_carriage_joint.pos"] = 0.015
        self.connects = 0
        self.disconnects = 0
        self.teleop_calls = 0
        self.events = []
        self.frames = []
        self.results = []
        self.inputs = []
        self.resets = []
        self.queues = [deque() for _ in stages]

    def connect(self):
        self.connects += 1

    def read(self):
        return dict(self.state)

    def reset_policy(self, index):
        self.resets.append(index)
        self.queues[index].clear()

    def predict(self, index, observation):
        self.inputs.append((index, dict(observation)))
        action = dict(observation)
        action["left_joint_0.pos"] += 0.01 * (index + 1)
        action["x.vel"] = 0.02
        return action

    def send(self, action):
        self.state.update(action)
        return dict(action)

    def teleop(self, observation):
        self.teleop_calls += 1
        return dict(observation)

    def stop_requested(self):
        return False

    def begin_trial(self, trial):
        self.event("trial_started", trial=trial)

    def record(self, observation, action, stage, phase):
        self.frames.append({"observation": dict(observation), "action": dict(action), "stage": stage, "phase": phase})

    def finish_trial(self, result):
        self.results.append(result)

    def event(self, name, **data):
        self.events.append({"event": name, **data})

    def display(self, observation, action):
        pass

    def stop(self, previous):
        self.state["x.vel"] = self.state["theta.vel"] = 0.0

    def close(self, park=False):
        self.disconnects += 1


def dry_run(config, stages, output):
    backend = MockBackend(stages)
    engine = SequenceController(stages, backend, trials=config["trials"])
    engine.connect()
    now = 0.0
    for _ in range(config["trials"]):
        for _stage in stages:
            engine.tick("s", now)
            now += 0.01
            engine.tick(None, now)
            now += 0.01
            engine.tick("n", now)
            now += 0.01
    engine.close()
    if (engine.phase != "done" or len(backend.results) != config["trials"]
            or any(result["status"] != "operator_success" for result in backend.results)
            or (backend.connects, backend.disconnects, backend.teleop_calls) != (1, 1, 0)):
        raise RuntimeError("Mock sequence did not complete as expected.")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": "mock_only", "connections": backend.connects,
        "disconnections": backend.disconnects, "teleop_calls": backend.teleop_calls,
        "results": backend.results, "policy_resets": backend.resets,
        "events": backend.events, "frames": backend.frames,
    }
    (output / "dry_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MOCK PASS: {len(backend.results)} sequences; connect={backend.connects}, disconnect={backend.disconnects}, teleop={backend.teleop_calls}")
    print(f"Report: {output / 'dry_run.json'}")
    print("No model weights, GPU, network, cameras or robot were used.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/task08_09_sequence.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Run mock transitions with standard Python only.")
    mode.add_argument("--check", action="store_true", help="Load/validate real models without connecting the robot (Linux/CUDA).")
    mode.add_argument("--run", action="store_true", help="Connect and operate the physical robot (Linux/CUDA).")
    parser.add_argument("--output", help="New local output directory; existing directories are never overwritten.")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stages = [Stage(**stage) for stage in config["stages"]]
    if len(stages) < 2 or config["trials"] < 1 or not 1 <= config["fps"] <= 60:
        parser.error("Need >=2 stages, >=1 trial and FPS in [1,60].")
    for stage in stages:
        if not math.isfinite(stage.timeout_s) or stage.timeout_s <= 0:
            parser.error("Timeout must be finite and positive.")
    if len({stage.name for stage in stages}) != len(stages):
        parser.error("Stage names must be unique.")
    output = Path(args.output).expanduser().resolve() if args.output else (
        Path(__file__).resolve().parents[1] / "output" / f"act_sequence_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if args.dry_run:
        dry_run(config, stages, output)
        return 0
    if sys.platform != "linux":
        parser.error("Edit and --dry-run on Windows; --check/--run use the Linux robot PC.")
    if args.run and not sys.stdin.isatty():
        parser.error("Use an interactive terminal for --run.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sequence_lerobot import LeRobotBackend
    stop_event = threading.Event()
    backend = LeRobotBackend(config, stages, output, stop_event, check_only=args.check)
    if args.check:
        print("CHECK PASS: both real models loaded and inferred; no robot connection.")
        return 0
    engine = SequenceController(stages, backend, config["trials"])
    old_handler = signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    park = False
    code = 0
    print("\nS: start/resume | N: mark current stage successful | SPACE: pause")
    print("F: fail this trial | T: toggle leader control in READY only")
    print("Q/Ctrl+C: stop & hold | P: park and exit in DONE | X: release without parking in DONE")
    print("At the 08->09 HANDOFF, press S to start 09 from the current pose.")
    print("Startup moves arms to the usual staged pose. P also moves arms; X releases connections/torque.")
    try:
        with TerminalKeys() as keys:
            engine.connect()
            while True:
                started = time.monotonic()
                command = keys.read()
                if stop_event.is_set():
                    stop_event.clear()
                    command = "q"
                if engine.phase == "done" and command in ("p", "x"):
                    park = command == "p"
                    break
                old_phase = engine.phase
                engine.tick(command, started)
                if engine.phase != old_phase:
                    print(f"\n[{engine.phase.upper()}] trial {engine.trial}/{engine.trials}; stage {stages[engine.index].name}")
                time.sleep(max(0, 1 / config["fps"] - (time.monotonic() - started)))
    except Exception as exc:
        code = 1
        logging.exception("Sequence stopped: %s", exc)
        try:
            engine.abort(str(exc))
        except Exception:
            logging.exception("Could not finish partial recording.")
    finally:
        signal.signal(signal.SIGINT, old_handler)
        engine.close(park=park)
    print(f"Saved locally: {output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
