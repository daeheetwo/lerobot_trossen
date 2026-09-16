"""LeRobot 0.4.4 / Mobile AI adapter. Imported only by real/check modes."""

import json
import logging
from pathlib import Path
import time

from sequence_core import checked_action, hold_action, validate_features


def resolve_model(value):
    from huggingface_hub import snapshot_download

    path = Path(value).expanduser()
    if not path.is_dir():
        path = Path(snapshot_download(value, allow_patterns=["*.json", "*.safetensors"]))
    if not (path / "config.json").is_file():
        path = path / "pretrained_model"
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (path / name).is_file():
            raise ValueError(f"Checkpoint file missing: {path / name}")
    return path


def load_dataset_info(value):
    from huggingface_hub import hf_hub_download

    local = Path(value).expanduser()
    path = local / "meta/info.json" if local.is_dir() else Path(
        hf_hub_download(value, "meta/info.json", repo_type="dataset", revision="main")
    )
    return json.loads(path.read_text(encoding="utf-8"))


class LeRobotBackend:
    def __init__(self, config, stages, output, stop_event, check_only=False):
        import importlib.metadata
        if importlib.metadata.version("lerobot") != "0.4.4":
            raise RuntimeError("This runner requires lerobot==0.4.4 (the repository's pinned environment).")
        import torch
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.datasets.utils import hw_to_dataset_features
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot_robot_trossen import MobileAIRobot, MobileAIRobotConfig
        from lerobot_teleoperator_trossen.config_mobileai_leader import MobileAILeaderTeleopConfig
        from lerobot_teleoperator_trossen.mobileai_leader import MobileAILeaderTeleop

        self.config, self.stages, self.output = config, stages, Path(output)
        self.stop_event = stop_event
        self.device = torch.device(config.get("device", "cuda"))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Real ACT evaluation requires a CUDA GPU; use --dry-run on this PC.")
        raw_robot = dict(config["robot"])
        raw_robot["cameras"] = {
            name: RealSenseCameraConfig(**camera) for name, camera in raw_robot["cameras"].items()
        }
        # Existing lerobot-record retains its default parking behaviour.
        # This runner parks only on an explicit P command after objects are secured.
        raw_robot["park_on_disconnect"] = False
        self.robot = MobileAIRobot(MobileAIRobotConfig(**raw_robot))
        self.teleoperator = MobileAILeaderTeleop(MobileAILeaderTeleopConfig(**config["teleop"]))
        self.action_names = list(self.robot.action_features)
        self.features = {
            **hw_to_dataset_features(self.robot.observation_features, "observation"),
            **hw_to_dataset_features(self.robot.action_features, "action"),
        }
        self.policies = []
        self.model_paths = []
        self.dataset = None
        self.events_file = None
        self.results_file = None
        self.frame_index = 0
        self.episode_index = 0
        self.active = False
        self.recording_failed = False
        self.closed = False
        self.trial_start = None
        self.display_enabled = config.get("display_data", True) and not check_only

        for stage in stages:
            path = resolve_model(stage.policy)
            raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
            info = load_dataset_info(stage.dataset)
            validate_features(raw, info, self.features, config["fps"])
            policy_config = ACTConfig.from_pretrained(str(path))
            policy_config.device = str(self.device)
            # Load the complete ACT checkpoint strictly; no extra ImageNet download.
            policy_config.pretrained_backbone_weights = None
            policy_config.use_amp = False
            policy = ACTPolicy.from_pretrained(str(path), config=policy_config, strict=True)
            pre, post = make_pre_post_processors(
                policy_cfg=policy_config, pretrained_path=str(path),
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
            for step in [*pre.steps, *post.steps]:
                stats = getattr(step, "stats", None)
                if stats:
                    for feature, values in stats.items():
                        for statistic, value in values.items():
                            if not torch.isfinite(torch.as_tensor(value)).all():
                                raise ValueError(f"Non-finite checkpoint statistics: {feature}/{statistic}")
            self.policies.append((policy, pre, post))
            self.model_paths.append(str(path))
            # Warm up and validate inference BEFORE connecting the physical robot.
            import numpy as np
            dummy = {name: 0.0 for name in self.action_names if name.endswith(".pos")}
            for name, shape in self.robot.observation_features.items():
                if isinstance(shape, tuple):
                    dummy[name] = np.zeros(shape, dtype=np.uint8)
                elif name not in dummy:
                    dummy[name] = 0.0
            self.predict(len(self.policies) - 1, dummy)
            self.reset_policy(len(self.policies) - 1)
            logging.info("Checked %s: state %s / action %s", stage.name,
                         info["features"]["observation.state"]["shape"], info["features"]["action"]["shape"])
            logging.info("%s: n_action_steps=%s, temporal_ensemble_coeff=%s", stage.name,
                         policy_config.n_action_steps, policy_config.temporal_ensemble_coeff)

        if check_only:
            return
        self.output.mkdir(parents=True, exist_ok=False)
        self.events_file = (self.output / "events.jsonl").open("w", encoding="utf-8")
        self.results_file = (self.output / "results.jsonl").open("w", encoding="utf-8")
        (self.output / "run_config.json").write_text(
            json.dumps({**config, "resolved_model_paths": self.model_paths}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.dataset = LeRobotDataset.create(
            repo_id="local/eval_act_sequence", root=self.output / "dataset",
            fps=config["fps"], features=self.features, robot_type=self.robot.name,
            use_videos=True, image_writer_threads=4, batch_encoding_size=1,
            vcodec="libx264", metadata_buffer_size=1,
        )
        if self.display_enabled:
            from lerobot.utils.visualization_utils import init_rerun
            init_rerun(session_name="act_sequence")

    def connect(self):
        self.robot.connect()
        self.teleoperator.connect()

    def read(self):
        return self.robot.get_observation()

    def reset_policy(self, index):
        for item in self.policies[index]:
            item.reset()

    def predict(self, index, observation):
        from lerobot.datasets.utils import build_dataset_frame
        from lerobot.policies.utils import make_robot_action
        from lerobot.utils.control_utils import predict_action
        policy, pre, post = self.policies[index]
        action = predict_action(
            observation=build_dataset_frame(self.features, observation, "observation"),
            policy=policy, device=self.device, preprocessor=pre, postprocessor=post,
            use_amp=False, task=self.stages[index].task, robot_type=self.robot.robot_type,
        )
        return checked_action(make_robot_action(action, self.features), self.action_names)

    def send(self, action):
        return self.robot.send_action(action)

    def teleop(self, observation):
        return self.teleoperator.get_action()

    def stop_requested(self):
        return self.stop_event.is_set()

    def begin_trial(self, trial):
        self.active = True
        self.frame_index = 0
        self.trial_start = time.monotonic()
        self.event("trial_started", trial=trial)

    def record(self, observation, action, stage, phase):
        from lerobot.datasets.utils import build_dataset_frame
        now = time.monotonic()
        # Dataset timestamp remains nominal frame/fps for LeRobot compatibility.
        # Actual wall time and phase are recorded separately, including pauses.
        self.dataset.add_frame({
            **build_dataset_frame(self.features, observation, "observation"),
            **build_dataset_frame(self.features, action, "action"),
            "task": self.stages[stage].task,
        })
        self.event("frame", frame=self.frame_index, stage=self.stages[stage].name,
                   phase=phase, elapsed_s=now - self.trial_start)
        self.frame_index += 1

    def finish_trial(self, result):
        if self.recording_failed:
            raise RuntimeError("Recording already failed; partial files retained for inspection.")
        if self.dataset is not None and self.frame_index:
            # Base has been stopped before any potentially slow video encoding.
            try:
                self.dataset.save_episode(parallel_encoding=False)
            except Exception:
                # save_episode is not transactional: retrying can duplicate rows.
                self.recording_failed = True
                self.active = False
                self.event("recording_failed", result=result)
                raise
            result = {**result, "episode_index": self.episode_index, "frames": self.frame_index}
            self.episode_index += 1
        else:
            result = {**result, "episode_index": None, "frames": 0}
        self.results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        self.results_file.flush()
        self.active = False

    def event(self, name, **data):
        if self.events_file is not None:
            self.events_file.write(json.dumps({
                "event": name, "monotonic_s": time.monotonic(),
                "episode_index": self.episode_index, "frame_index": self.frame_index, **data,
            }, ensure_ascii=False) + "\n")
            if name != "frame":
                self.events_file.flush()
        if name != "frame":
            logging.info("%s %s", name, data)

    def display(self, observation, action):
        if self.display_enabled:
            from lerobot.utils.visualization_utils import log_rerun_data
            log_rerun_data(observation=observation, action=action)

    def stop(self, previous):
        # Do not depend on camera availability to stop a moving base.
        try:
            self.robot.base.set_cmd_vel(0.0, 0.0)
        except Exception:
            logging.exception("Could not send base stop.")
        try:
            if self.robot.arms.is_connected:
                obs = self.robot.arms.get_observation()
                action = hold_action(obs, previous, self.action_names)
                self.robot.arms.send_action({k: v for k, v in action.items() if k.endswith(".pos")})
        except Exception:
            logging.exception("Could not hold arms; hardware stop may be required.")

    def close(self, park=False):
        if self.closed:
            return
        self.closed = True
        # Also clean partial connections; top-level is_connected can be False
        # when one camera/arm failed during startup.
        for arm in (self.robot.arms.left_arm, self.robot.arms.right_arm):
            try:
                arm.config.park_on_disconnect = park
                if arm.driver.get_is_configured():
                    arm.disconnect()
            except Exception:
                logging.exception("Arm cleanup failed.")
        for camera in self.robot.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                logging.exception("Camera cleanup failed.")
        for arm in (self.teleoperator.left_arm, self.teleoperator.right_arm):
            try:
                if arm.driver.get_is_configured():
                    if park:
                        arm.disconnect()
                    else:
                        # Leader disconnect normally parks too. X/error cleanup
                        # must not move the leader handles unexpectedly either.
                        arm.driver.cleanup()
            except Exception:
                logging.exception("Leader cleanup failed.")
        try:
            if self.dataset is not None:
                try:
                    if self.active and self.frame_index and not self.recording_failed:
                        self.finish_trial({"status": "aborted", "manual_transition": True})
                finally:
                    self.dataset.finalize()
        finally:
            if self.dataset is not None:
                self.dataset.stop_image_writer()
            for handle in (self.events_file, self.results_file):
                if handle is not None:
                    handle.close()
