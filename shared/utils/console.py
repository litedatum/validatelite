"""
Console-safe output helpers.

Provide encoding-safe wrappers around click.echo to avoid UnicodeEncodeError on
legacy Windows consoles (e.g., GBK) when printing symbols like ✓, ✗, │, etc.
"""

from __future__ import annotations

import sys
from typing import Optional

import click


def safe_encode_for_console(text: str, *, is_error: bool = False) -> str:
    """Encode text for the current console, replacing unsupported characters.

    This avoids UnicodeEncodeError on some Windows terminals.
    """
    try:
        stream = sys.stderr if is_error else sys.stdout
        encoding = getattr(stream, "encoding", None) or "utf-8"
        return text.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
    except Exception:
        return text


def safe_echo(text: str, *, fg: Optional[str] = None, err: bool = False) -> None:
    """Echo text with color and encoding safety using click.echo."""
    safe_text = safe_encode_for_console(text, is_error=err)
    if fg:
        safe_text = click.style(safe_text, fg=fg)
    click.echo(safe_text, err=err)
