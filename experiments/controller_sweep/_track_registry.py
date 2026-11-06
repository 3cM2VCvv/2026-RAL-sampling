"""Register fixture-backed tracks with Buzzracer's ``TrackFactory`` at runtime.

This overrides the factory entry for each named track only in the current
process; no submodule file is edited.  ``Main`` then builds the track from the
plain XML ``<track>name</track>`` text.
"""

import os

from raceline_track import FIXTURE_DIR, RacelineTrack

_ORIGINAL_GET_MAPPING = None


def register_fixture_tracks(names):
    global _ORIGINAL_GET_MAPPING
    from buzzracer.tracks.track_factory import TrackFactory

    if _ORIGINAL_GET_MAPPING is None:
        _ORIGINAL_GET_MAPPING = TrackFactory.get_mapping

    mapping = dict(_ORIGINAL_GET_MAPPING())
    for name in names:
        path = os.path.join(FIXTURE_DIR, f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing track fixture {path}; run build_track_fixtures first"
            )

        def builder(main, config, track=None, _path=path):
            return RacelineTrack(main, config, fixture_path=_path)

        mapping[name] = builder

    TrackFactory.get_mapping = staticmethod(lambda: mapping)
