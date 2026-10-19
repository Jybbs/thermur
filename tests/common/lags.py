"""
Holds the lag a test lifts the `uv` mise pins through to `LIFTED`, a major
lift from any release before it, and `BREAKING`, the lines of a stand-in
`mise` whose `mise ci` fails wherever that lift has landed.
"""

from pathlib import Path

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Lag

LIFTED   = "9.0.0"
BREAKING = (
    f'[ "$*" = "run ci" ] && grep -q \'"{LIFTED}"\' .mise/config.toml && exit 1\nexit 0'
)


def uv_lag(checkout: Checkout) -> Lag:
    """
    Builds the lag lifting the `uv` pin `.mise/config.toml` holds in the
    copy `checkout` roots to `LIFTED`.
    """
    return Lag.declared(
        {Path(".mise/config.toml"): r'^"?uv"?\s*=\s*"'},
        current = checkout.config.uv,
        newest  = LIFTED,
        pin     = "uv"
    )
