"""
The MV Wakashio casualty — documented facts.

Background the AIS feed cannot supply on its own: who the vessel was, what she
was carrying, where she was going, and what happened after she stopped moving.
Everything here comes from the public record of the casualty.

The kinematics are *not* here. Position, timing and behaviour are read from the
real AIS feed in `real_ais.py`, which contains the vessel's own broadcasts —
including the moment she lost way and the six days of status 6 (aground) that
followed. Where the two could disagree, the feed wins: `real_ais.build_scenario`
overwrites `position` and `grounding_utc` with what was actually observed.

Sources: Panama Maritime Authority casualty investigation, Mauritius National
Crisis Committee statements, and ITOPF / IMO incident reporting.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

# Documented grounding site, Pointe d'Esny reef. The AIS feed puts the vessel's
# final resting position ~50 m from this, which is well inside the precision of
# a position quoted to four decimal places in an incident report.
GROUNDING_LAT = -20.4442
GROUNDING_LON = 57.7433

# 19:25 local (UTC+4) per the casualty record. The feed's own last-moving ping
# is 15:27:22 UTC — a two-minute agreement with the official time.
GROUNDING_UTC = datetime(2020, 7, 25, 15, 25, tzinfo=timezone.utc)

INCIDENT: Dict[str, Any] = {
    "name": "MV Wakashio grounding",
    "location": "Pointe d'Esny reef, south-east Mauritius",
    "grounding_utc": GROUNDING_UTC,
    "grounding_local": "25 July 2020, 19:25 (UTC+4)",
    "position": (GROUNDING_LAT, GROUNDING_LON),
    "vessel": {
        "name": "MV WAKASHIO",
        "mmsi": 372711000,          # as broadcast in the AIS feed
        "imo": 9337119,
        "flag": "Panama",
        "type": "Bulk carrier (Capesize)",
        "length_m": 203,
        "dwt": 101932,
        "owner": "Nagashiki Shipping (Japan)",
    },
    "voyage": "Lianyungang, China -> Tubarao, Brazil, in ballast",
    "bunkers_t": 3894,
    "diesel_t": 207,
    "oil_released_t": 1000,
    "leak_began": "6 August 2020",
    "hull_failure": "Vessel broke in two, 15 August 2020",
    "cause_summary": (
        "The vessel altered course toward the coast to obtain mobile telephone "
        "signal, closed the reef at full service speed with no effective "
        "navigational watch, and struck it at approximately 11 knots without "
        "any avoiding action being taken."
    ),
    "why_it_matters": (
        "The behavioural signature -- a sustained inshore approach at service speed "
        "followed by an abrupt loss of way -- is exactly what an AIS anomaly detector "
        "is meant to surface, and it precedes the visible spill by twelve days."
    ),
}


def grounding_point() -> Tuple[float, float]:
    return GROUNDING_LAT, GROUNDING_LON
