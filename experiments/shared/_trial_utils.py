"""Shared helpers for the controller sweep trial."""

import zlib

import numpy as np
from _buzzracer_config import write_temp_config
from scipy.interpolate import splev


def trial_seed(base_seed, track, disturbance):
    """Per-(track, disturbance) seed, excluding the controller.

    Every controller therefore faces the same plant-noise stream on a given
    cell (common random numbers), which is what makes the comparison paired.
    """
    return trial_seed_from_combo(base_seed, f"{track}-{disturbance}")


def trial_seed_from_combo(base_seed, combo):
    """Deterministic seed from a base seed and a DVC combination name."""
    return (int(base_seed) * 1_000_003 + zlib.crc32(combo.encode())) % (2**31)


def compute_initial_state(track, point_index=0, offset_name="center"):
    """Return ``(x, y, heading, v_forward)`` on the raceline at ``s = index/10``."""
    track_len = track.raceline_len_m
    s = (point_index / 10) * track_len

    rr = splev(s % track_len, track.raceline_s, der=0)
    drr = splev(s % track_len, track.raceline_s, der=1)
    heading = float(np.arctan2(drr[1], drr[0]))
    v_ref = float(track.sToV(np.array([s]))[0])

    normal = np.array([-np.sin(heading), np.cos(heading)])
    offsets = {"center": 0.0, "inside": -0.05, "outside": 0.05}
    d = offsets[offset_name]

    x = float(rr[0] + d * normal[0])
    y = float(rr[1] + d * normal[1])
    return x, y, heading, v_ref


def build_xml_config(
    template_path, *, track, controller_type, init_states, sim, controller
):
    """Write a temp XML config derived from ``template_path``.

    ``sim`` supplies the settings/simulator/auto-stopper values; ``controller``
    supplies the controller attributes.  The auto-stopper class is set to
    ``SimTimeAutoStopper`` and the track name is written into ``<track>``.
    """

    def mutate(config):
        settings_nodes = config.getElementsByTagName("settings")
        if settings_nodes:
            settings_nodes[0].setAttribute("dt", str(sim["dt"]))

        for ext_node in config.getElementsByTagName("extension"):
            handle = ext_node.getAttribute("handle")
            if handle == "simulator":
                for key in (
                    "state_noise_direction",
                    "state_noise_enabled",
                    "state_noise_magnitude",
                    "state_noise_type",
                ):
                    ext_node.setAttribute(key, str(sim[key]))
            elif handle == "auto_stopper":
                for key in ("wall_clock", "crash_stop", "stall_timeout", "lap_count"):
                    ext_node.setAttribute(key, str(sim[key]))
                ext_node.firstChild.data = "SimTimeAutoStopper"

        ctrl_node = config.getElementsByTagName("controller")[0]
        for key, value in controller.items():
            ctrl_node.setAttribute(key, str(value))
        ctrl_node.getElementsByTagName("type")[0].firstChild.data = controller_type

        config.getElementsByTagName("track")[0].firstChild.data = track

        car_node = config.getElementsByTagName("car")[0]
        init_nodes = car_node.getElementsByTagName("init_states")
        init_text = repr(tuple(init_states))
        if init_nodes:
            init_nodes[0].firstChild.data = init_text
        else:
            elem = config.createElement("init_states")
            elem.appendChild(config.createTextNode(init_text))
            car_node.appendChild(elem)

    return write_temp_config(template_path, mutate)
