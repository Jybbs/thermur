#!/usr/bin/env python3
# MISE description = "Delete each cache entry a newer one of its generation replaced"
"""
Deletes the Actions cache entries newer ones replaced, as the `gha:prune`
task the prune job of `🐦‍⬛ Warm` runs on the runner's own interpreter and
the `gh` it carries. `Caches.prune` groups the entries by generation, keeps
the newest entry of each, and deletes the rest through `gh cache delete`,
which needs the `actions: write` the prune job holds, printing a warning for
a deletion GitHub refuses rather than stopping.
"""

from dataclasses import dataclass, fields
from itertools   import groupby
from json        import loads
from operator    import attrgetter
from subprocess  import check_output, run
from typing      import Self


@dataclass(frozen=True, kw_only=True)
class Entry:
    """
    One Actions cache entry, holding the fields GitHub's REST API returns
    for it under the names the API gives them.
    """

    created_at    : str
    id            : int
    key           : str
    ref           : str
    size_in_bytes : int

    @property
    def generation(self) -> tuple[str, str]:
        """
        Pairs the branch that saved the entry with its key once the hash
        ending it is dropped, which every newer entry of the same archive
        shares, since each save ends the key on a hash of the files that
        decide what the archive holds.
        """
        return self.ref, self.key.rpartition("-")[0]

    def delete(self) -> bool:
        """
        Deletes the entry, printing its key and size where GitHub accepts
        the request and a warning naming the error `gh` reports where it
        refuses.

        Returns:
            Whether GitHub accepted the request.
        """
        deletion = run(
            ["gh", "cache", "delete", str(self.id)],
            capture_output = True,
            text           = True
        )
        if deletion.returncode:
            print(
                f"::warning::{self.key} was not deleted, since `gh` reports "
                f"`{deletion.stderr.strip()}`"
            )
            return False

        print(f"Deleted {self.key}, freeing {self.size_in_bytes} bytes")
        return True


@dataclass(frozen=True)
class Caches:
    """
    The Actions cache entries the repository holds.
    """

    entries: list[Entry]

    @property
    def replaced(self) -> list[Entry]:
        """
        Sets aside every entry a newer entry of its generation replaced,
        leaving out an entry whose key holds no `-`, since no hash ends it.
        """
        ordered = sorted(
            (entry for entry in self.entries if "-" in entry.key),
            key = attrgetter("generation", "created_at")
        )
        return [
            entry
            for _, generation in groupby(ordered, key=attrgetter("generation"))
            for entry in [*generation][:-1]
        ]

    @classmethod
    def listed(cls) -> Self:
        """
        Reads every entry the repository holds, paging through the whole
        listing.
        """
        listing = check_output(
            [
                "gh", "api", "--jq",
                ".actions_caches[]", "--paginate", "repos/{owner}/{repo}/actions/caches"
            ],
            text = True
        )
        names = [field.name for field in fields(Entry)]
        return cls(
            [
                Entry(**{name: row[name] for name in names})
                for row in map(loads, listing.splitlines())
            ]
        )

    def prune(self):
        """
        Deletes every replaced entry, then prints how many of them GitHub
        deleted and how many bytes that freed.
        """
        stale   = self.replaced
        deleted = [entry for entry in stale if entry.delete()]
        print(
            f"Deleted {len(deleted)} of {len(stale)} replaced entries, freeing "
            f"{sum(entry.size_in_bytes for entry in deleted)} bytes"
        )


if __name__ == "__main__":
    Caches.listed().prune()
