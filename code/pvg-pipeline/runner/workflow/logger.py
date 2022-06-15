# System imports
import sys
import os
import csv
from datetime import datetime
from glob import glob

import comet_ml
from ignite.engine import Engine, Events
from ignite.handlers import Timer

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)


class MetricCSVLogger:

    def __init__(self,
                 csv_path: str,
                 period: int,
                 epoch_length: int):
        self.csv_path = csv_path
        self.period = period
        self.epoch_length = epoch_length
        self.metric_buffer = []

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        if not engine.has_event_handler(self.on_epoch):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch)

    def on_epoch(self, engine):
        self.metric_buffer.append((engine.state.epoch, engine.state.metrics.copy()))
        if engine.state.epoch % self.period == 0:
            self.save_buffer()
            self.metric_buffer = []

    def save_buffer(self):
        for epoch, metrics in self.metric_buffer:
            self.save_metrics(epoch, metrics)

    def save_metrics(self, epoch, metrics):
        csv_exists = os.path.isfile(self.csv_path)
        open_mode = "a" if csv_exists else "w"
        with open(self.csv_path, open_mode) as f:
            fieldnames = ["epoch", "iteration"] + list(metrics.keys())

            writer = csv.DictWriter(f, fieldnames)

            if not csv_exists:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "iteration": self.epoch_length * epoch, **metrics})


class MetricConsoleLogger:

    def __init__(self, header="METRICS", footer="-"*50):
        self.header = header
        self.footer = footer

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        if not engine.has_event_handler(self.on_epoch):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch)

    def on_epoch(self, engine):
        self.print_metrics(engine.state.epoch, engine.state.metrics)

    def print_metrics(self, epoch, metrics):
        print(self.header)
        print(f"Epoch: {epoch}")
        for metric_name, metric in metrics.items():
            print(f"{metric_name}: {metric}")
        print(self.footer)

class EpochProgressLogger:

    def __init__(self, progress_freq=5, header="PROGRESS", footer="-"*50):
        assert progress_freq > 0, "progress_freq must be greater than 0"
        self.header = header
        self.footer = footer
        self.progress_freq = max(progress_freq, 1)

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        if not engine.has_event_handler(self.on_iteration):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.on_iteration)

    def on_iteration(self, engine):
        if engine.state.iteration % engine.state.epoch_length == 1:
            print(self.header)
            print(f"Epoch: {engine.state.epoch}")

        progress_interval = max(engine.state.epoch_length // self.progress_freq, 1)
        if engine.state.iteration % engine.state.epoch_length == 1 or \
           (engine.state.iteration % engine.state.epoch_length) % progress_interval == 0 or \
           engine.state.iteration % engine.state.epoch_length == 0:
            epoch_progress = engine.state.iteration % engine.state.epoch_length
            if epoch_progress == 0: epoch_progress += engine.state.epoch_length
            current_time_str = datetime.now().isoformat()
            print(f"{current_time_str}: {epoch_progress}/{engine.state.epoch_length}")

        if engine.state.iteration % engine.state.epoch_length == 0:
            print(self.footer)


class StartPausedTimer(Timer):
    def __init__(self, average: bool = False, start_paused: bool = True):
        self.start_paused = start_paused
        super().__init__(average)

    def reset(self, *args):
        super().reset()

        if self.start_paused:
            super().pause()

class TimeConsoleLogger:
    def __init__(self, header="TIMES", footer="-"*50):
        self.header = header
        self.footer = footer
        self.batch_timer = StartPausedTimer(average=True)
        self.iteration_timer = StartPausedTimer(average=True)

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        if not engine.has_event_handler(self.on_epoch):
            self.batch_timer.attach(engine,
                                    start=Events.EPOCH_STARTED,
                                    resume=Events.GET_BATCH_STARTED,
                                    pause=Events.GET_BATCH_COMPLETED,
                                    step=Events.GET_BATCH_COMPLETED)
            self.iteration_timer.attach(engine,
                                        start=Events.EPOCH_STARTED,
                                        resume=Events.ITERATION_STARTED,
                                        pause=Events.ITERATION_COMPLETED,
                                        step=Events.ITERATION_COMPLETED)

            engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch)

    def on_epoch(self, engine):
        times = {"AVERAGE BATCH_FETCH": self.batch_timer.value(),
                 "AVERAGE ITERATION": self.iteration_timer.value(),
                 **engine.state.times
                }
        self.print_times(engine.state.epoch, times)

    def print_times(self, epoch, times):
        print(self.header)
        print(f"Epoch: {epoch}")
        for time_name, time in times.items():
            if time_name != Events.COMPLETED.name:
                print(f"{time_name}: {time}")
        print(self.footer)


class CometMLLogger:
    def __init__(self,
                 project_name: str,
                 exp_name: str,
                 exp_config: dict,
                 exp_dir: str,
                 checkpoint_dir: str,
                 is_resuming: bool,
                 epoch_length: int,
                 workspace: str = None):
        self.epoch_length = epoch_length
        self.exp_name = exp_name

        if is_resuming:
            with open(os.path.join(checkpoint_dir, "comet.exp_key"), "r") as f:
                prev_exp_key = f.readline().strip()
            self.experiment = comet_ml.ExistingExperiment(previous_experiment=prev_exp_key,
                                                          auto_metric_logging=False,
                                                          log_env_details=True,
                                                          log_env_host=True,
                                                          log_code=False)
        else:
            self.experiment = comet_ml.Experiment(project_name=project_name,
                                                  workspace=workspace,
                                                  auto_metric_logging=False,
                                                  log_env_details=True,
                                                  log_env_host=True,
                                                  log_code=False)
            self.experiment.log_other('experiment_name', exp_name)
            for f in glob(f'{exp_dir}/*.py'):
                self.experiment.log_code(file_name=f)
            self._log_config(exp_config)

            with open(os.path.join(checkpoint_dir, "comet.exp_key"), "w") as f:
                f.write(self.experiment.get_key())


    def _log_config(self, config):
        def recurse(config_prefix, config):
            for key, value in config.items():
                key = "->".join(filter(None, [config_prefix, key]))
                if isinstance(value, dict):
                    recurse(key, value)
                else:
                    self.experiment.log_parameter(key, value)
        recurse("", config)

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        if not engine.has_event_handler(self.on_epoch):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch)

        if not engine.has_event_handler(self.on_terminate):
            engine.add_event_handler(Events.TERMINATE, self.on_terminate)

    def on_epoch(self, engine):
        self.experiment.log_metrics(engine.state.metrics, step=engine.state.epoch * self.epoch_length, epoch=engine.state.epoch)

    def on_terminate(self, engine):
        self.experiment.add_tag("TERMINATED")
        self.experiment.send_notification(title=f"TERMINATED EXPERIMENT - {self.exp_name}",
                                          status="aborted")