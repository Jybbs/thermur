#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "lift"
# MISE description = "Report each lagging pin, then try or write the lift"
# USAGE flag "--hold <pin>" var=#true help="Hold back the pin a row names <pin>"
# USAGE flag "--try" help="Try the lift in a scratch clone through `mise ci`"
# USAGE flag "--write" help="Write the lift into the checkout once any trial passes"
"""
Prints every pin the checkout declares that lags its newest release as one
Markdown table, as the `lock:lift` task, then tries the lift in a scratch
clone under `--try` and writes it under `--write`, leaving each pin a
`--hold` names where it stands. mise passes each flag the usage spec above
declares through a `usage_` variable.
"""

from datetime import UTC, datetime
from os       import environ
from sys      import stderr

from thermur.repo.checkout import Checkout
from thermur.repo.lift     import Lift
from thermur.repo.schemas  import join_names


def main():
    """
    Prints the table, then exits with status `2` where a hold names no row
    of it. Under `--try`, it exits with status `1` where the lift fails
    `mise ci`, naming the pin a major lift moves whose hold lets the rest
    pass. Under `--write`, it then writes the lift into the checkout.
    """
    lift = Lift(checkout=Checkout(), holds=tuple(environ.get("usage_hold", "").split()))
    print(lift.table(datetime.now(UTC)), flush=True)
    if unmatched := lift.unmatched:
        print(
            "No row of the table names "
            + join_names([f"`{hold}`" for hold in unmatched]),
            file = stderr
        )
        raise SystemExit(2)

    if environ.get("usage_try") == "true" and not lift.passes():
        held = lift.isolate()
        raise SystemExit(
            "The lift fails `mise ci`, "
            + (
                f"and holding back `{held}` lets the rest pass it"
                if held
                else "and holding back no single major lift lets the rest pass it"
            )
        )

    if environ.get("usage_write") == "true":
        lift.write(lift.checkout)


if __name__ == "__main__":
    main()
