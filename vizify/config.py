"""Local configuration helpers for Vizify."""

import os
from pathlib import Path


def get_api_key():
    """Return the configured Gemini key without bundling secrets with the package.

    Environment configuration takes precedence. An optional local Gemini key
    file can be selected with ``VIZIFY_API_KEY_FILE``. The repository's
    ``API.key`` is reserved for PyPI publishing and is never read here.
    """
    configured = os.getenv("GEMINI_API_KEY", "").strip()
    if configured:
        return "" if configured.startswith("pypi-") else configured

    explicit_path = os.getenv("VIZIFY_API_KEY_FILE")
    if not explicit_path:
        return ""
    try:
        value = Path(explicit_path).expanduser().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""
    if value and "=" in value and "\n" not in value:
        name, candidate = value.split("=", 1)
        if name.strip().upper() in {"GEMINI_API_KEY", "API_KEY"}:
            value = candidate.strip().strip("\"'")
    return "" if value.startswith("pypi-") else value
