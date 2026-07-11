"""Uniform club / league badges.

Real crests are trademarked, differently sized, and awkward to redistribute, so
the UI's baseline is a **generated** badge: a clean shield with a deterministic
two-tone gradient derived from the name plus the club's initials. Every badge
shares one 64x64 viewBox, so they line up perfectly at any render size.

If a real crest has been fetched into ``static/logos/`` (see
``scripts/fetch_logos.py``), the Flask layer prefers it; otherwise it serves
these SVGs. Result: always something tidy and same-sized, online or off.
"""
from __future__ import annotations

import hashlib
import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Drop generic filler words when computing initials so "Paris SG" -> "PSG",
# "Manchester United" -> "MU", "1. FC Koln" -> "FCK".
_STOP = {"the", "of", "fc", "cf", "sc", "ac", "afc", "cd", "ud", "sv", "us",
         "as", "rc", "b", "ii"}


def slugify(name: str) -> str:
    """Filesystem/URL-safe key for a name: 'Paris SG' -> 'paris-sg'."""
    return "-".join(_WORD_RE.findall(str(name).lower())) or "na"


def initials(name: str, maxlen: int = 3) -> str:
    """Compact monogram: significant word initials, else first letters."""
    words = [w for w in _WORD_RE.findall(str(name)) if w]
    sig = [w for w in words if w.lower() not in _STOP] or words
    if len(sig) >= 2:
        mono = "".join(w[0] for w in sig[:maxlen])
    elif sig:
        mono = sig[0][:maxlen]
    else:
        mono = "?"
    return mono.upper()


def _hash(name: str) -> int:
    return int(hashlib.md5(str(name).encode("utf-8")).hexdigest(), 16)


def palette_for(name: str) -> tuple[str, str, str]:
    """Deterministic (top, bottom, ink) colors for a name.

    Two hues a little apart give a lively gradient; the ink is white or near
    black depending on the base lightness so the initials always read.
    """
    h = _hash(name)
    hue = h % 360
    hue2 = (hue + 24 + (h >> 8) % 40) % 360
    sat = 62 + (h >> 3) % 20          # 62..82
    light = 40 + (h >> 5) % 12        # 40..52
    top = f"hsl({hue},{sat}%,{light + 8}%)"
    bottom = f"hsl({hue2},{sat}%,{max(light - 6, 18)}%)"
    ink = "#ffffff" if light < 60 else "#10151c"
    return top, bottom, ink


def _shield_path() -> str:
    # A simple, modern crest silhouette inside the 64x64 box.
    return ("M32 4 L56 12 L56 34 C56 48 46 56 32 61 "
            "C18 56 8 48 8 34 L8 12 Z")


def team_badge_svg(name: str, size: int = 64) -> str:
    """Self-contained SVG crest for a club. Deterministic, uniform, offline."""
    top, bottom, ink = palette_for(name)
    mono = initials(name)
    gid = "g" + hashlib.md5(name.encode()).hexdigest()[:8]
    fs = 26 if len(mono) <= 2 else 20
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
        f'width="{size}" height="{size}" role="img" aria-label="{name}">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{top}"/>'
        f'<stop offset="1" stop-color="{bottom}"/></linearGradient></defs>'
        f'<path d="{_shield_path()}" fill="url(#{gid})" '
        f'stroke="rgba(255,255,255,.35)" stroke-width="1.5"/>'
        f'<path d="M8 20 H56" stroke="rgba(255,255,255,.18)" stroke-width="3"/>'
        f'<text x="32" y="34" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{fs}" font-weight="800" fill="{ink}" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'letter-spacing="0.5">{mono}</text></svg>'
    )


def league_badge_svg(league: str, label: str | None = None, size: int = 64) -> str:
    """Self-contained SVG roundel for a league/competition."""
    top, bottom, ink = palette_for(league)
    # Prefer the country initial + tier digit if the label looks like "X · Y".
    text = league.split("-")[-1] if "-" in league else initials(label or league)
    text = text.upper()[:4]
    gid = "l" + hashlib.md5(league.encode()).hexdigest()[:8]
    fs = 22 if len(text) <= 2 else (18 if len(text) == 3 else 15)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
        f'width="{size}" height="{size}" role="img" aria-label="{label or league}">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{top}"/>'
        f'<stop offset="1" stop-color="{bottom}"/></linearGradient></defs>'
        f'<circle cx="32" cy="32" r="28" fill="url(#{gid})" '
        f'stroke="rgba(255,255,255,.35)" stroke-width="2"/>'
        f'<circle cx="32" cy="32" r="28" fill="none" '
        f'stroke="rgba(0,0,0,.15)" stroke-width="1"/>'
        f'<text x="32" y="33" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{fs}" font-weight="800" fill="{ink}" '
        f'text-anchor="middle" dominant-baseline="central">{text}</text></svg>'
    )
