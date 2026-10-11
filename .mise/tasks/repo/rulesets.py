#!/usr/bin/env -S uv run --exact --locked
# MISE alias       = "rulesets"
# MISE confirm     = "Apply the rulesets, settings, description, and topics to GitHub?"
# MISE description = "Apply the rulesets and settings, listing live rulesets left out"
"""
Applies every ruleset under `.github/rulesets/` and sends every setting
`.github/settings.toml` declares, beside the description and the topics
`pyproject.toml` declares, to the repository the checkout's remote names,
as the `repo:rulesets` task, once mise's prompt is answered. It then prints
each live ruleset no file names, which stays on GitHub until a deletion
by hand.
"""

from json       import loads as from_json
from pydantic   import ValidationError
from subprocess import check_output, run

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import join_names, repeated


def main():
    """
    Validates every file the task reads before anything reaches GitHub,
    exiting with the validation error where one fails, or with each name
    two ruleset files share. It then reads every live ruleset, updates the
    one sharing each file's name through `PUT` or creates it through `POST`,
    and sends each setting, stopping on the first call `gh` fails, before
    printing each live ruleset no file names beneath a line naming what
    they are.
    """
    checkout = Checkout()
    try:
        rulesets = checkout.rulesets.values()
        requests = [*checkout.github_settings.requests, *checkout.manifest.requests]
    except ValidationError as error:
        raise SystemExit(error) from None

    if shared := sorted(repeated(ruleset.name for ruleset in rulesets)):
        raise SystemExit(
            "More than one file under `.github/rulesets/` names each of these "
            "rulesets, which the task matches to a live ruleset by name: "
            + join_names(shared)
        )

    live = dict(
        map(
            from_json,
            check_output(
                [
                    "gh", "api",
                    "--jq", ".[] | [.name, .id]",
                    "--paginate", "repos/{owner}/{repo}/rulesets?includes_parents=false"
                ],
                text = True
            ).splitlines()
        )
    )
    for request in [
        *(ruleset.request(live.get(ruleset.name)) for ruleset in rulesets),
        *requests
    ]:
        run(request.command, check=True, input=request.payload)

    if omitted := sorted(live.keys() - {ruleset.name for ruleset in rulesets}):
        print(
            "No file under `.github/rulesets/` names these live rulesets, left for "
            "a deletion by hand:",
            *omitted,
            sep = "\n"
        )


if __name__ == "__main__":
    main()
