"""
Declares the digest of each subpackage's source and `uv.lock` as a Hamilton
step named `<subpackage>_digest`. A step calling into a subpackage's code
reads that subpackage's digest, so a change to the code or to a library it
imports computes the reading step again.
"""

from hamilton.function_modifiers import cache, parameterize, value

from thermur.pipeline.schemas import Subpackage


@cache(behavior="recompute")
@parameterize(
    **{
        f"{subpackage.name}_digest": {"subpackage": value(subpackage)}
        for subpackage in Subpackage.listed()
    }
)
def digest(lock_digest: str, subpackage: Subpackage) -> str:
    """
    Hashes the source of one subpackage beside the digest of `uv.lock`,
    computed again on every run, since Hamilton's cache reads none of the
    files the hash covers.
    """
    return subpackage.digest(lock_digest)
