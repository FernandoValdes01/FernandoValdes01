#!/usr/bin/env python3
"""Generate the particle morph used by the profile terminal SVGs."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
except ImportError:  # Keep the generator usable without SciPy.
    linear_sum_assignment = None
    cdist = None


ROOT = Path(__file__).resolve().parents[1]
PARTICLE_COUNT = 520
LOOP_SECONDS = 10
KEY_TIMES = "0;.22;.5;.78;1"

BOBA_FETT = [
    "⢀⣀⣀⡀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⠀",
    "⠈⢹⠉⢁⣴⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀",
    "⠀⢸⠀⣾⣿⣿⣿⣿⣿⣿⣷⣶⣿⣿⡀⠀",
    "⠀⣾⢀⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⡀⠀",
    "⠀⣿⣤⣀⠈⠉⠉⢻⣿⡟⠉⠉⠉⣀⣤⠀",
    "⠀⠀⣿⣿⣷⣄⠀⢸⣿⡇⠀⢠⣾⣿⣿⡀",
    "⠀⢰⣿⣿⣿⣿⡆⢸⣿⡇⢠⣿⣿⣿⣿⡇",
    "⠀⠈⠙⠻⢿⣿⣿⢸⣿⡇⣸⣿⡿⠟⠋⠁",
    "⠀⠀⠀⠀⠀⠈⠉⠈⠉⠁⠉⠁⠀⠀⠀⠀",
]

BRAILLE_DOTS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (0, 3),
    (1, 3),
)

THEMES = {
    "profile-terminal-morph-v2.svg": {
        "line": "#1d3557",
        "muted": "#8293b0",
        "code": "#11e6ff",
        "boba": "#b995ff",
        "footer": "#00e676",
    },
    "profile-terminal-morph-v2-light.svg": {
        "line": "#cbd5e1",
        "muted": "#64748b",
        "code": "#0891b2",
        "boba": "#4a3d7a",
        "footer": "#059669",
    },
}


def braille_points() -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row, line in enumerate(BOBA_FETT):
        for column, character in enumerate(line):
            value = ord(character) - 0x2800
            if not 0 <= value <= 0xFF:
                continue
            for bit, (offset_x, offset_y) in enumerate(BRAILLE_DOTS):
                if value & (1 << bit):
                    points.append((column * 2 + offset_x, row * 4 + offset_y))

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    scale = min(220 / (max_x - min_x), 104 / (max_y - min_y))
    center_x, center_y = 185, 263

    normalized = [
        (
            center_x + (x - (min_x + max_x) / 2) * scale,
            center_y + (y - (min_y + max_y) / 2) * scale,
        )
        for x, y in points
    ]

    expanded: list[tuple[float, float]] = []
    for x, y in normalized:
        for offset_x, offset_y in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            expanded.append((x + offset_x, y + offset_y))
    return expanded


def segment_points(
    start: tuple[float, float], end: tuple[float, float], width: float = 6
) -> list[tuple[float, float]]:
    distance = math.dist(start, end)
    steps = max(2, math.ceil(distance / 1.25))
    points: list[tuple[float, float]] = []
    for step in range(steps + 1):
        progress = step / steps
        center_x = start[0] + (end[0] - start[0]) * progress
        center_y = start[1] + (end[1] - start[1]) * progress
        radius = math.ceil(width)
        for offset_x in range(-radius, radius + 1):
            for offset_y in range(-radius, radius + 1):
                if offset_x * offset_x + offset_y * offset_y <= width * width:
                    points.append((center_x + offset_x, center_y + offset_y))
    return points


def code_points() -> list[tuple[float, float]]:
    segments = (
        ((148, 215), (112, 263)),
        ((112, 263), (148, 311)),
        ((203, 210), (167, 316)),
        ((220, 215), (256, 263)),
        ((256, 263), (220, 311)),
    )
    points: list[tuple[float, float]] = []
    for start, end in segments:
        points.extend(segment_points(start, end))
    return points


def resample(points: list[tuple[float, float]], seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    if len(points) >= PARTICLE_COUNT:
        return rng.sample(points, PARTICLE_COUNT)

    repeated: list[tuple[float, float]] = []
    for index in range(PARTICLE_COUNT):
        x, y = points[index % len(points)]
        jitter = 0.35 if index >= len(points) else 0
        repeated.append(
            (
                x + rng.uniform(-jitter, jitter),
                y + rng.uniform(-jitter, jitter),
            )
        )
    return repeated


def transport(
    source: list[tuple[float, float]], target: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if linear_sum_assignment is not None and cdist is not None:
        rows, columns = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
        ordered: list[tuple[float, float] | None] = [None] * len(source)
        for row, column in zip(rows, columns):
            ordered[row] = target[column]
        return [point for point in ordered if point is not None]

    source_order = sorted(range(len(source)), key=lambda index: (source[index][1], source[index][0]))
    target_order = sorted(range(len(target)), key=lambda index: (target[index][1], target[index][0]))
    ordered = [None] * len(source)
    for source_index, target_index in zip(source_order, target_order):
        ordered[source_index] = target[target_index]
    return [point for point in ordered if point is not None]


def fmt(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def particle_group(theme: dict[str, str]) -> str:
    code = resample(code_points(), seed=41)
    boba = resample(braille_points(), seed=42)
    boba = transport(code, boba)
    frames = (code, code, boba, boba, code)
    parts = [
        f'  <path d="M52 124h10 M52 124v10 M318 124h-10 M318 124v10 M52 395h10 M52 395v-10 M318 395h-10 M318 395v-10" fill="none" stroke="{theme["code"]}" stroke-width="1" opacity=".65"/>',
        '  <g id="morph-particles" shape-rendering="crispEdges">',
    ]
    for index in range(PARTICLE_COUNT):
        positions = ";".join(
            f"{fmt(frame[index][0])} {fmt(frame[index][1])}" for frame in frames
        )
        initial_x, initial_y = code[index]
        parts.append(
            f'    <path d="M-.7-.7h1.4v1.4h-1.4z" transform="translate({fmt(initial_x)} {fmt(initial_y)})" fill="{theme["code"]}">'
            f'<animateTransform attributeName="transform" type="translate" dur="{LOOP_SECONDS}s" repeatCount="indefinite" calcMode="linear" keyTimes="{KEY_TIMES}" values="{positions}"/>'
            f'<animate attributeName="fill" dur="{LOOP_SECONDS}s" repeatCount="indefinite" keyTimes="{KEY_TIMES}" values="{theme["code"]};{theme["code"]};{theme["boba"]};{theme["boba"]};{theme["code"]}"/>'
            "</path>"
        )
    parts.extend(
        [
            "  </g>",
            f'  <text x="52" y="418" fill="{theme["footer"]}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="9">PTS {PARTICLE_COUNT:04d} · a.k.a. &quot;Corvo&quot;</text>',
        ]
    )
    return "\n".join(parts)


def generate(filename: str, theme: dict[str, str]) -> None:
    path = ROOT / filename
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'  <path d="M52 124h10.*?  <text x="52" y="418"[^>]*>.*?</text>',
        re.DOTALL,
    )
    updated, replacements = pattern.subn(particle_group(theme), source, count=1)
    if replacements != 1:
        raise RuntimeError(f"Could not find the visual panel in {filename}")
    path.write_text(updated, encoding="utf-8")


for output, colors in THEMES.items():
    generate(output, colors)
