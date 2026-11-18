"""
Declares the digest of each subpackage's source and `uv.lock` as a Hamilton
step named `<subpackage>_digest`. A step reads its own subpackage's digest,
which also covers every module of the package that the subpackage imports,
followed through their own imports, so a change to any code the step runs or
to a library it imports computes the step again.
"""

from hamilton.function_modifiers import cache, parameterize, value

from thermur.pipeline.schemas import Subpackage


@cache(behavior="recompute")
@parameterize(
    **{
        f"{subpackage.name}_digest": {"name": value(subpackage.name)}
        for subpackage in Subpackage.listed()
    }
)
def digest(lock_digest: str, name: str) -> str:
    """
    Hashes the source of the subpackage called `name` beside the digest of
    `uv.lock`, computed again on every run, since Hamilton's cache reads
    none of the files the hash covers.
    """
    return Subpackage.named(name).digest(lock_digest)
