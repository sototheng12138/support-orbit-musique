from __future__ import annotations

from support_orbit_musique.audit import (
    SEMANTIC_FEATURE_NAMES,
    SURFACE_FEATURE_NAMES,
    normalize_text,
    semantic_features,
    surface_features,
)


def _paragraphs() -> list[dict[str, object]]:
    return [
        {
            "idx": index,
            "title": f"Title {index}",
            "paragraph_text": f"Text {index} about rail and city with number {index}",
            "is_supporting": False,
        }
        for index in range(20)
    ]


def test_frozen_feature_dimensions_and_surface_question_exclusion() -> None:
    paragraphs = _paragraphs()
    surface = surface_features(paragraphs)
    first = semantic_features("rail city?", paragraphs)
    second = semantic_features("completely unrelated tokens?", paragraphs)
    assert len(surface) == len(SURFACE_FEATURE_NAMES) == 18
    assert len(first) == len(SEMANTIC_FEATURE_NAMES) == 23
    assert first[:18] == surface == second[:18]
    assert first[18:] != second[18:]


def test_unicode_normalization_is_frozen() -> None:
    assert normalize_text("  América—RAIL\nCity  ") == "américa rail city"
