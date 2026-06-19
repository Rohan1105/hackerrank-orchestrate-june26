"""Deterministic, code-only image checks that run before any model call.

These are intentionally cheap heuristics, not a replacement for the vision model's own
possible_manipulation / non_original_image judgment -- they're a second, independent
signal (defense in depth) and a way to catch images the model never gets to see because
they fail to load at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# EXIF tag IDs we care about (see Pillow's TiffTags / EXIF spec).
EXIF_SOFTWARE_TAG = 305
# Editing tools that, if present in EXIF Software, are a real (if soft) manipulation signal.
KNOWN_EDITOR_KEYWORDS = ("photoshop", "gimp", "snapseed", "lightroom", "pixlr", "canva")


@dataclass
class ImageInspection:
    loadable: bool
    width: int | None = None
    height: int | None = None
    has_exif: bool = False
    editing_software: str | None = None
    risk_flags: list[str] = field(default_factory=list)
    note: str = ""


def inspect_image(path: Path) -> ImageInspection:
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            exif = img.getexif()
            has_exif = len(exif) > 0
            software = exif.get(EXIF_SOFTWARE_TAG)
            software = str(software) if software else None
    except (UnidentifiedImageError, OSError):
        return ImageInspection(
            loadable=False,
            risk_flags=["damage_not_visible"],
            note="Image file could not be opened/decoded.",
        )

    flags = []
    notes = []

    if software and any(k in software.lower() for k in KNOWN_EDITOR_KEYWORDS):
        flags.append("possible_manipulation")
        notes.append(f"EXIF Software tag indicates editing tool: {software}.")

    # A real camera/phone photo almost always carries *some* EXIF. A total absence of
    # EXIF data is a soft signal the image was re-saved, screenshotted, or came from a
    # source other than the original capture -- not proof, just a hint to combine with
    # the model's own read of the image.
    if not has_exif:
        flags.append("non_original_image")
        notes.append("No EXIF metadata present (re-saved/screenshot/stripped-metadata image).")

    return ImageInspection(
        loadable=True,
        width=width,
        height=height,
        has_exif=has_exif,
        editing_software=software,
        risk_flags=flags,
        note=" ".join(notes),
    )


def inspection_to_cache_dict(inspection: ImageInspection) -> dict:
    return {
        "loadable": inspection.loadable,
        "width": inspection.width,
        "height": inspection.height,
        "has_exif": inspection.has_exif,
        "editing_software": inspection.editing_software,
        "risk_flags": inspection.risk_flags,
        "note": inspection.note,
    }


def inspection_from_cache_dict(d: dict) -> ImageInspection:
    return ImageInspection(**d)
