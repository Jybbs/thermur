"""
Pins what the render task writes into `docs/` and how it reads the copies
the repository holds, meaning the identity as JSON, a drifted or missing
copy reported, and only a drifted copy written again.
"""

from json    import loads
from pathlib import Path

from thermur.docs.outputs     import Outputs
from thermur.identity.schemas import Identity


def test_a_missing_or_edited_copy_drifts_and_is_written_again(tmp_path: Path):
    """
    Asserts that a missing file and an edited one both count as drifted,
    that writing rewrites each, and that a second write finds nothing to
    write.
    """
    outputs = Outputs(root=tmp_path)
    assert outputs.drifted == [outputs.identity]

    assert outputs.write() == [outputs.identity]
    (tmp_path / outputs.identity).write_text("{}\n", encoding="utf-8")

    assert outputs.drifted == [outputs.identity]
    assert outputs.write() == [outputs.identity]
    assert outputs.write() == []


def test_the_render_writes_the_identity_as_json():
    """
    Asserts that the identity file holds the identity as `Identity.shipped`
    reads it, so the site reads the very records the chart theme does.
    """
    outputs = Outputs()

    assert loads(outputs.written[outputs.identity]) == Identity.shipped().model_dump(
        mode = "json"
    )
