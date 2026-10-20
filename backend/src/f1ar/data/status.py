"""Official result status strings across seasons (H2a, D-054).

Since 2023 FastF1 reports a lapped finisher as ``Lapped``; before that (Ergast/Jolpica style)
as ``+1 Lap`` / ``+2 Laps``. Comparing with ``"Lapped"`` alone counted every lapped 2018-2022
finisher as a retirement.
"""

import re

_LAPS_DOWN = re.compile(r"^\+\d+ Laps?$")


def is_lapped(status: object) -> bool:
    text = str(status)
    return text == "Lapped" or bool(_LAPS_DOWN.match(text))


def is_running_finish(status: object) -> bool:
    """Took the chequered flag (on the lead lap or lapped)."""
    return str(status) == "Finished" or is_lapped(status)


def is_finisher(status: object) -> bool:
    """Classified at the end without retiring (incl. a disqualified finisher)."""
    return is_running_finish(status) or str(status) == "Disqualified"
