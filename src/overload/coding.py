"""Attribute coding of the original study's options.

The functions reproduce the earlier analysis pipeline's coding exactly, since every reported number
depends on it. File options are coded on extension, owner, modification time, permissions and size.
Laptop options are coded on memory, graphics, storage, processor, battery, brand and price. An
attribute that a display did not show receives its lowest level, so it is constant within the trial.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta

import numpy as np

FILE_ATTRIBUTES: tuple[str, ...] = ("extension", "owner", "modified", "permissions", "size")
LAPTOP_ATTRIBUTES: tuple[str, ...] = (
    "memory",
    "graphics",
    "storage",
    "processor",
    "battery",
    "brand",
    "price",
)

_EXECUTABLE_NAMES = {"patcher", "zephyr_run", "process_1", "process_2", "jappa", "kite_build"}
_HIGH_BRANDS = {"yellow", "monza", "maranello"}
_MID_BRANDS = {"aurora", "silverstone", "lusail"}


# ----------------------------------------------------------------------------- files
def _parse_size(text: object) -> float | None:
    if not isinstance(text, str) or text.strip() == "":
        return None
    match = re.match(r"([\d.]+)\s*([KMGT]?)i?B?", text.strip(), re.IGNORECASE)
    if not match:
        try:
            return float(text.strip())
        except ValueError:
            return None
    multiplier = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    return float(match.group(1)) * multiplier.get(match.group(2).upper(), 1.0)


def _parse_modified(text: object) -> datetime | None:
    if not isinstance(text, str) or text.strip() == "":
        return None
    # Listings show either a time (current year) or a year. Every date is placed in 2024, as before.
    try:
        return datetime.strptime(f"{text.strip()} 2024", "%b %d %H:%M %Y")
    except ValueError:
        pass
    try:
        return datetime.strptime(text.strip(), "%b %d %Y").replace(year=2024)
    except ValueError:
        return None


def _is_executable(permissions: object) -> bool:
    return isinstance(permissions, str) and "x" in permissions


def _extension(name: str, meta: Mapping[str, str]) -> str:
    if meta.get("Extension", ""):
        return meta["Extension"].lower()
    if "." in name:
        return name.rsplit(".", 1)[1].lower()
    return ""


def code_extension(name: str, meta: Mapping[str, str]) -> int:
    ext = _extension(name, meta)
    if ext in {"enc", "zip"}:
        return 3
    if _is_executable(meta.get("Perm", "")) or ext in {"sh", "py"}:
        return 2
    if ext == "" and name.lower() in _EXECUTABLE_NAMES:
        return 2
    return 1


def code_owner(meta: Mapping[str, str]) -> int:
    owner = str(meta.get("Owner", "unknown")).strip().lower()
    return 3 if owner == "james" else (2 if owner == "alice" else 1)


def code_modified(meta: Mapping[str, str], reference: datetime | None) -> int:
    when = _parse_modified(meta.get("Modified", ""))
    if when is None or reference is None:
        return 1
    age = reference - when
    return 3 if age <= timedelta(days=1) else (2 if age <= timedelta(days=7) else 1)


def code_permissions(meta: Mapping[str, str]) -> int:
    permissions = meta.get("Perm", "")
    if _is_executable(permissions):
        return 3
    return 2 if permissions and permissions.strip()[8:9] != "w" else 1


def code_size(meta: Mapping[str, str]) -> int:
    size = _parse_size(meta.get("Size", ""))
    if size is None:
        return 1
    return 4 if size > 1e9 else (3 if size > 1e8 else (2 if size > 1e6 else 1))


def code_file_trial(options: Mapping[str, Mapping[str, str]]) -> np.ndarray:
    """Coded levels of every file option, in the order of ``options``."""
    dates = [_parse_modified(meta.get("Modified", "")) for meta in options.values()]
    dates = [d for d in dates if d is not None]
    reference = max(dates) if dates else None
    rows = [
        [
            code_extension(name, meta),
            code_owner(meta),
            code_modified(meta, reference),
            code_permissions(meta),
            code_size(meta),
        ]
        for name, meta in options.items()
    ]
    return np.asarray(rows, dtype=float)


# ----------------------------------------------------------------------------- laptops
def code_memory(meta: Mapping[str, object]) -> int:
    ram = str(meta.get("RAM", ""))
    if "32" in ram:
        return 4
    if "16" in ram:
        return 3
    if "8" in ram.split("G")[0]:
        return 2
    return 1


def code_graphics(meta: Mapping[str, object]) -> int:
    gpu = str(meta.get("GPU", "")).lower()
    if "dedicated 8" in gpu or "dedicated8" in gpu:
        return 4
    if "dedicated 4" in gpu or "dedicated4" in gpu:
        return 3
    if "dedicated 2" in gpu or "dedicated2" in gpu:
        return 2
    if "integrated 8" in gpu:
        return 2
    return 1


def code_storage(meta: Mapping[str, object]) -> int:
    drive = str(meta.get("HD", "")).lower()
    if "ssd" in drive:
        if "2tb" in drive:
            return 5
        if "1tb" in drive:
            return 4
        if "512" in drive:
            return 3
        if "256" in drive or "128" in drive:
            return 2
    return 1


def code_processor(meta: Mapping[str, object]) -> int:
    cpu = str(meta.get("Processor", "")).lower()
    if "i9" in cpu or "ryzen 7" in cpu:
        return 4
    if "i7" in cpu or "ryzen 5" in cpu:
        return 3
    if "i5" in cpu:
        return 2
    return 1


def code_battery(meta: Mapping[str, object]) -> int:
    try:
        hours = int(str(meta.get("Battery", "")).strip())
    except ValueError:
        return 1
    if hours >= 12:
        return 4
    if hours >= 10:
        return 3
    if hours >= 8:
        return 2
    return 1


def code_brand(meta: Mapping[str, object]) -> int:
    brand = str(meta.get("Brand", "")).strip().lower()
    if brand in _HIGH_BRANDS:
        return 3
    if brand in _MID_BRANDS:
        return 2
    return 1


def code_price(meta: Mapping[str, object]) -> int:
    try:
        price = float(str(meta.get("Price", 0)).replace("$", "").replace(",", ""))
    except ValueError:
        return 1
    if price > 2000:
        return 4
    if price > 1500:
        return 3
    if price >= 1000:
        return 2
    return 1


def code_laptop_trial(options: Mapping[str, Mapping[str, object]]) -> np.ndarray:
    """Coded levels of every laptop option, in the order of ``options``."""
    rows = [
        [
            code_memory(meta),
            code_graphics(meta),
            code_storage(meta),
            code_processor(meta),
            code_battery(meta),
            code_brand(meta),
            code_price(meta),
        ]
        for meta in options.values()
    ]
    return np.asarray(rows, dtype=float)
