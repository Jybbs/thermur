"""
Pins the entries `gha:prune` deletes, meaning every entry a newer one of its
generation replaced and none other, the commands it sends GitHub, and the
warning it prints for a deletion GitHub refuses before carrying on.
"""

from collections.abc  import Callable
from common.stand_ins import StandIn
from json             import dumps
from pytest           import CaptureFixture, MonkeyPatch, fixture
from syrupy.assertion import SnapshotAssertion
from types            import ModuleType


def entry(created_at: str, id: int, key: str, ref: str = "refs/heads/main") -> str:
    """
    Writes one cache entry as the line the listing's `--jq` filter prints
    for it, beside a field the task leaves unread.
    """
    return dumps(
        {
            "created_at"       : created_at,
            "id"               : id,
            "key"              : key,
            "last_accessed_at" : created_at,
            "ref"              : ref,
            "size_in_bytes"    : 100 * id
        }
    )


@fixture
def gh(stand_in: Callable[[str, str], StandIn]) -> StandIn:
    """
    Puts a stand-in `gh` on the path that prints the entries `CACHES` holds
    for an `api` call and refuses the deletion of the entry `REFUSED` names.
    """
    return stand_in(
        "gh",
        '[ "$1" = api ] && printf "%s" "$CACHES" && exit 0\n'
        '[ "$3" = "${REFUSED:-}" ] && echo "HTTP 403" >&2 && exit 1\n'
        "exit 0"
    )


@fixture
def prune(
    gh          : StandIn,
    load_task   : Callable[[str], ModuleType],
    monkeypatch : MonkeyPatch
) -> ModuleType:
    """
    Sets the entries `CACHES` holds for the stand-in `gh` to list.

    Returns:
        The task script, imported as a module.
    """
    monkeypatch.setenv(
        "CACHES",
        "\n".join(
            [
                entry(
                    created_at = "2026-10-01T00:00:00Z",
                    id         = 1,
                    key        = "mise-linux-x64-ubuntu26-aaa-old"
                ),
                entry(
                    created_at = "2026-10-03T00:00:00Z",
                    id         = 2,
                    key        = "mise-linux-x64-ubuntu26-aaa-new"
                ),
                entry(
                    created_at = "2026-10-02T00:00:00Z",
                    id         = 3,
                    key        = "mise-linux-x64-ubuntu26-aaa-mid"
                ),
                entry(
                    created_at = "2026-10-01T00:00:00Z",
                    id         = 4,
                    key        = "mise-linux-x64-ubuntu26-bbb-only"
                ),
                entry(
                    created_at = "2026-10-01T00:00:00Z",
                    id         = 5,
                    key        = "mise-linux-x64-ubuntu26-aaa-pr",
                    ref        = "refs/pull/7/merge"
                ),
                entry("2026-09-01T00:00:00Z", 6, "unhashed"),
                entry("2026-09-02T00:00:00Z", 7, "unhashed")
            ]
        )
    )
    return load_task(".mise/tasks/gha/prune.py")


def test_a_refused_deletion_warns_and_the_rest_carry_on(
    capsys      : CaptureFixture[str],
    monkeypatch : MonkeyPatch,
    prune       : ModuleType
):
    """
    Asserts that a deletion GitHub refuses prints a warning naming the entry
    and the error, while the task deletes the remaining entries and counts
    only those GitHub accepted.
    """
    monkeypatch.setenv("REFUSED", "1")

    prune.Caches.listed().prune()

    assert capsys.readouterr().out.splitlines() == [
        "::warning::mise-linux-x64-ubuntu26-aaa-old was not deleted, since "
        "`gh` reports `HTTP 403`",
        "Deleted mise-linux-x64-ubuntu26-aaa-mid, freeing 300 bytes",
        "Deleted 1 of 2 replaced entries, freeing 300 bytes"
    ]


def test_only_the_entries_a_newer_one_replaced_are_deleted(
    capsys   : CaptureFixture[str],
    gh       : StandIn,
    prune    : ModuleType,
    snapshot : SnapshotAssertion
):
    """
    Asserts that the task deletes every entry older than the newest of its
    generation, a generation spanning one branch and one key once its hash
    is dropped, and leaves the newest, an entry another branch saved, and an
    entry whose key ends on no hash.
    """
    prune.Caches.listed().prune()

    assert "\n".join(gh.commands) == snapshot
    assert capsys.readouterr().out.splitlines() == [
        "Deleted mise-linux-x64-ubuntu26-aaa-old, freeing 100 bytes",
        "Deleted mise-linux-x64-ubuntu26-aaa-mid, freeing 300 bytes",
        "Deleted 2 of 2 replaced entries, freeing 400 bytes"
    ]
