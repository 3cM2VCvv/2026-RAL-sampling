"""Extension for sacred experiment tracking.

When loaded, this extension automatically:
- Captures the full XML config as a sacred config dict
- Collects runtime attributes from extensions and controllers
- Registers output files as sacred artifacts
- Logs summary metrics from sibling extensions

Usage in XML config (must be listed LAST among extensions):

    <extension handle='sacred_experiment' observer_path='sacred_runs'>
      SacredExperiment
    </extension>
"""
import os
from threading import Thread, Event
from time import time

from sacred import Experiment
from sacred.observers import FileStorageObserver

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension
from buzzracer.sacred_utils import xml_config_to_dict


_MAX_COLLECTION_LEN = 20


def _is_simple_value(val) -> bool:
    """Check if a value is a simple JSON-serializable primitive or a
    small, shallow collection of primitives (suitable for sacred config)."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return True
    if isinstance(val, (list, tuple)):
        return len(val) <= _MAX_COLLECTION_LEN and all(
            _is_simple_value(v) for v in val)
    if isinstance(val, dict):
        return len(val) <= _MAX_COLLECTION_LEN and all(
            isinstance(k, str) and _is_simple_value(v)
            for k, v in val.items())
    return False


def _collect_config_attrs(obj) -> dict:
    """Collect simple config-like attributes from an object, skipping
    private/internal attributes and complex objects."""
    result = {}
    for key in sorted(vars(obj)):
        if key.startswith('_'):
            continue
        val = getattr(obj, key)
        if callable(val):
            continue
        if _is_simple_value(val):
            result[key] = val
    return result


class SacredExperiment(Extension):
    """Extension for automatic sacred experiment tracking."""

    def __init__(self):
        Extension.__init__(self, 'sacred_experiment')
        self.observer_path = 'sacred_runs'
        self.ex = None
        self._run = None
        self._start_time = None
        self._run_started = Event()
        self._run_complete = Event()
        self._sacred_thread = None

    def pre_init(self):
        # Parse XML config into a dict
        config_dict = xml_config_to_dict(self.main.config_filename)
        config_dict['config_filename'] = self.main.config_filename

        # Create sacred Experiment
        self.ex = Experiment(self.main.experiment_name)
        observer_dir = os.path.join(BASEDIR, self.observer_path)
        self.ex.observers.append(FileStorageObserver(observer_dir))
        self.ex.add_config(config_dict)

        # Replace Main's stub methods with real implementations
        self.main.log_metric = self._log_metric
        self.main.register_artifact = self._register_artifact

    def post_init(self):
        # Auto-collect runtime attributes from all extensions
        runtime_config = {'extension_runtime': {}}
        for ext in Extension.extensions:
            if ext is self:
                continue
            attrs = _collect_config_attrs(ext)
            if attrs:
                runtime_config['extension_runtime'][ext.name] = attrs

        # Auto-collect runtime attributes from all controllers
        for i, car in enumerate(self.main.cars):
            if car.controller is not None:
                attrs = _collect_config_attrs(car.controller)
                if attrs:
                    runtime_config[f'car_{i}_controller_runtime'] = attrs

        self.ex.add_config(runtime_config)

        # Start sacred run in a background thread.
        # The decorated main function blocks on _run_complete, keeping the
        # sacred run alive for the duration of the experiment.
        @self.ex.main
        def _sacred_main(_run):
            self._run = _run
            self._run_started.set()
            self._run_complete.wait()

        self._sacred_thread = Thread(target=self.ex.run, daemon=True)
        self._sacred_thread.start()
        self._run_started.wait()
        self._start_time = time()
        self.print_ok('Sacred run started (ID: %s)' % self._run._id)

    def post_final(self):
        if self._run is None:
            return

        # Collect summary metrics from sibling extensions
        self._collect_summary_metrics()

        # Register all accumulated artifacts with sacred
        registered = set()
        for filepath in self.main._sacred_artifacts:
            if os.path.isfile(filepath) and filepath not in registered:
                try:
                    self._run.add_artifact(filepath)
                    registered.add(filepath)
                except Exception as e:
                    self.print_warning(f'Failed to add artifact {filepath}: {e}')

        # Fallback: scan known output dirs for files modified during the run
        self._scan_output_dirs(registered)

        # Signal the sacred run to complete
        self._run_complete.set()
        self._sacred_thread.join(timeout=10)
        self.print_ok('Sacred run completed (ID: %s)' % self._run._id)

    def _log_metric(self, name, value, step=None):
        """Log a scalar metric to the active sacred run."""
        if self._run is not None:
            self._run.log_scalar(name, value, step)

    def _register_artifact(self, filepath, name=None):
        """Register a file as a sacred artifact."""
        self.main._sacred_artifacts.append(filepath)

    def _collect_summary_metrics(self):
        """Collect summary metrics not already logged by sibling extensions.

        Metrics like laptime_mean, laptime_stddev, boundary_violations, and
        oob_fraction are logged directly by their respective extensions via
        self.main.log_metric(). This method only collects metrics that are
        not logged elsewhere.
        """
        # Total laps (not logged by laptimer's own log_metric calls)
        if hasattr(self.main, 'laptimer'):
            lt = self.main.laptimer
            for car in self.main.cars:
                if car in lt.laptime_vec_by_car:
                    self._run.log_scalar(
                        f'total_laps_car{car.id}',
                        len(lt.laptime_vec_by_car[car]))

        # Step counter total
        if hasattr(self.main, 'step_counter'):
            self._run.log_scalar('total_steps', self.main.step_counter.count)

        # Simulation duration
        if self.main.simulator is not None and hasattr(self.main.simulator, 'sim_t'):
            self._run.log_scalar('sim_duration', self.main.simulator.sim_t)

    def _scan_output_dirs(self, already_registered: set):
        """Scan known output directories for files modified during the run."""
        scan_dirs = [
            os.path.join(BASEDIR, 'gifs'),
        ]
        # Add experiment-specific log dir if logger is loaded
        if hasattr(self.main, 'logger') and hasattr(self.main.logger, 'log_folder'):
            scan_dirs.append(self.main.logger.log_folder)

        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for fname in os.listdir(scan_dir):
                fpath = os.path.join(scan_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if fpath in already_registered:
                    continue
                if os.path.getmtime(fpath) >= self._start_time:
                    try:
                        self._run.add_artifact(fpath)
                        already_registered.add(fpath)
                    except Exception:
                        pass
