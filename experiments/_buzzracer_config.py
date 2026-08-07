"""Shared helper for building temporary Buzzracer XML configs.

Several experiment scripts need to run Buzzracer's ``Main`` against a config
derived from a committed template XML with a few attributes overridden, without
editing the committed file (the whole ``Buzzracer`` dir is a dependency of the
reachset_trial sweep, so its content hash must stay stable). This centralizes
the parse -> mutate -> write-temp boilerplate; callers supply a ``mutate``
callback that edits the parsed DOM in place.

Like ``_env``, this is an ``experiments/``-local helper: ``import
_buzzracer_config`` resolves because each script runs as
``python experiments/<script>.py`` (so ``experiments/`` is ``sys.path[0]``).
"""

import os
import tempfile
from xml.dom import minidom


def write_temp_config(template_path, mutate):
    """Return the path to a temp XML config derived from ``template_path``.

    ``mutate`` is called with the parsed ``minidom`` document and should apply
    the desired overrides in place. The caller is responsible for deleting the
    returned file.
    """
    config = minidom.parse(template_path)
    mutate(config)
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="buzzracer_config_")
    with os.fdopen(fd, "w") as f:
        config.writexml(f)
    return path
