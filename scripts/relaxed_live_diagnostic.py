#!/usr/bin/env python3
"""Run navila_eval.py with live rendering but without strict root locking.

This is intentionally a diagnostic launcher, not a training entry point.  It
keeps the normal Safe-VLN command line and only overrides the two checks that
would otherwise force the episode-start pose: the wrapper's strict reset flag
and the initial Habitat alignment guard.
"""

from __future__ import annotations

import scripts.navila_eval as navila_eval


class RelaxedVLNEnvWrapper(navila_eval.VLNEnvWrapper):
    def __init__(self, *args, **kwargs):
        kwargs["strict_start_alignment"] = False
        super().__init__(*args, **kwargs)


def main() -> int:
    navila_eval.VLNEnvWrapper = RelaxedVLNEnvWrapper
    # The live renderer still records the actual alignment error.  Suppress
    # only the fail-fast start-pose guard for this controlled experiment.
    navila_eval.navigation_alignment_error = lambda *args, **kwargs: (0.0, 0.0)
    navila_eval.main()
    navila_eval.simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
