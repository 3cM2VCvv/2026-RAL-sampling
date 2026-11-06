"""Buzzracer XML config construction for the reachset sweep trial."""

import os

from _buzzracer_config import write_temp_config
from buzzracer.common import BASEDIR

BASE_XML = os.path.join(BASEDIR, "buzzracer", "configs", "reachset_sweep_base.xml")


def build_xml_config(overrides):
    """Create a temporary XML config file with the given overrides applied.

    Returns the path to the temporary file (caller must delete it).
    """

    def mutate(config):
        # --- Settings ---
        settings_nodes = config.getElementsByTagName("settings")
        if settings_nodes and "dt" in overrides:
            settings_nodes[0].setAttribute("dt", str(overrides["dt"]))

        # --- Extensions (simulator + auto_stopper) ---
        for ext_node in config.getElementsByTagName("extension"):
            handle = ext_node.getAttribute("handle")
            if handle == "simulator":
                for key in (
                    "state_noise_direction",
                    "state_noise_enabled",
                    "state_noise_magnitude",
                    "state_noise_type",
                ):
                    if key in overrides:
                        ext_node.setAttribute(key, str(overrides[key]))
            if handle == "auto_stopper":
                for key in (
                    "wall_clock",
                    "crash_stop",
                    "stall_timeout",
                    "lap_count",
                ):
                    if key in overrides:
                        ext_node.setAttribute(key, str(overrides[key]))

        # --- Controller attributes ---
        ctrl_node = config.getElementsByTagName("controller")[0]
        for key in (
            "use_reachsets",
            "prng_seed",
            "centering_weight_factor",
            "velocity_scale_factor",
            "horizon",
            "max_speed",
        ):
            if key in overrides:
                ctrl_node.setAttribute(key, str(overrides[key]))

        # --- Initial state ---
        car_node = config.getElementsByTagName("car")[0]
        init_nodes = car_node.getElementsByTagName("init_states")
        init_text = repr(overrides["init_states"])
        if init_nodes:
            init_nodes[0].firstChild.data = init_text
        else:
            elem = config.createElement("init_states")
            elem.appendChild(config.createTextNode(init_text))
            car_node.appendChild(elem)

    return write_temp_config(BASE_XML, mutate)
