"""
Holds `StandIn`, a program a task test puts on the path in place of the real
one, which the `stand_in` fixture writes and returns.
"""

from pathlib  import Path
from pydantic import BaseModel
from shlex    import join


class StandIn(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One stand-in program, beside the log it writes each command it receives
    to before running the lines a case gives it.
    """

    log: Path
    """
    The log, one command a line with the program's name and each argument
    closed by a NUL.
    """

    @property
    def commands(self) -> list[str]:
        """
        Reads each command the program received, quoted as a shell reads it,
        or none where it received none.
        """
        if not self.log.exists():
            return []

        return [
            join(record.split("\0")[:-1])
            for record in self.log.read_text(encoding="utf-8").splitlines()
        ]
