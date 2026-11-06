"""Registry of controllers available to the controller sweep.

Add a new controller by registering one entry:
    register("stanley", "StanleyCarController")              # in Buzzracer
    register("mine", "MyController", MyController)            # in this repo

``type_name`` is written into the XML ``<controller><type>`` node.  A parent
repo class is injected into the ``buzzracer.controllers`` namespace at runtime,
so no submodule file is edited.
"""

from dataclasses import dataclass

# Import the package (not immrax_controller directly) so its __init__ finishes
# before ImmraxController is resolved; importing immrax_controller first creates
# a circular import through buzzracer.controllers.
import buzzracer.controllers as _controllers


@dataclass(frozen=True)
class ControllerSpec:
    type_name: str
    cls: type | None = None


_CONTROLLERS: dict[str, ControllerSpec] = {}


def register(name, type_name, cls=None):
    _CONTROLLERS[name] = ControllerSpec(type_name, cls)


def available():
    return sorted(_CONTROLLERS)


def resolve(name):
    try:
        return _CONTROLLERS[name]
    except KeyError:
        raise SystemExit(
            f"unknown controller {name!r}; registered: {sorted(_CONTROLLERS)}"
        )


def ensure_registered(spec):
    """Make ``spec.type_name`` importable from ``buzzracer.controllers``."""
    if spec.cls is None:
        return
    from buzzracer import controllers

    if not hasattr(controllers, spec.type_name):
        setattr(controllers, spec.type_name, spec.cls)


register("immrax", "ImmraxController", _controllers.ImmraxController)

from shield_mppi_controller import ShieldMPPIController

register("shieldmppi", "ShieldMPPIController", ShieldMPPIController)

from mppi_controller import MPPI

register("mppi", "MPPI", MPPI)

from tube_mppi_controller import TubeMPPI

register("tubemppi", "TubeMPPI", TubeMPPI)
