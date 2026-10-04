"""
Holds the shell variables each test starts without, which the autouse
`environment` fixture clears and `tests/isolation.py` checks.
"""

CLEARED = (
    "CLICOLOR", "CLICOLOR_FORCE", "COLORTERM", "FORCE_COLOR",
    "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "NO_COLOR", "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"
)
