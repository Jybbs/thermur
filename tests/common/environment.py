"""
Holds the shell variables each test starts without, which the autouse
`environment` fixture clears and `tests/isolation.py` checks.
"""

CLEARED = (
    "CI", "CLICOLOR", "CLICOLOR_FORCE", "COLORTERM",
    "COMMIT", "FORCE_COLOR", "GITHUB_ACTIONS", "GITHUB_HEAD_REF",
    "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "MISE_YES", "NO_COLOR",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"
)
