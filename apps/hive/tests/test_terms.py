"""Tests for hive scoring terms — pure functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
from hive.terms import (
    varroa_score, queenless_score, queen_cell_score,
    stores_score, disease_score, brood_score,
    HIVE_TERMS, inspection_to_context,
)
from hive.models import HiveInspection
from hive.modes import detect_mode, get_mode_weights, MODES
from hololoom.core.memory.scored_observation import score_terms


class TestVarroaScore:
    def test_emergency(self):
        assert varroa_score({"varroa_per_100": 8}) == 1.0
        assert varroa_score({"varroa_per_100": 12}) == 1.0

    def test_treat(self):
        assert varroa_score({"varroa_per_100": 5}) == 0.7

    def test_concern(self):
        assert varroa_score({"varroa_per_100": 3}) == 0.4

    def test_healthy(self):
        assert varroa_score({"varroa_per_100": 1}) == 0.0
        assert varroa_score({"varroa_per_100": 0}) == 0.0


class TestQueenlessScore:
    def test_no_queen_no_eggs(self):
        assert queenless_score({"queen_seen": False, "eggs_seen": False}) == 1.0

    def test_no_queen_eggs_present(self):
        assert queenless_score({"queen_seen": False, "eggs_seen": True}) == 0.5

    def test_queen_present(self):
        assert queenless_score({"queen_seen": True, "eggs_seen": True}) == 0.0


class TestQueenCellScore:
    def test_many_cells(self):
        assert queen_cell_score({"queen_cells": 5}) == 1.0

    def test_few_cells(self):
        assert queen_cell_score({"queen_cells": 2}) == 0.7

    def test_one_cell(self):
        assert queen_cell_score({"queen_cells": 1}) == 0.4

    def test_no_cells(self):
        assert queen_cell_score({"queen_cells": 0}) == 0.0


class TestStoresScore:
    def test_light(self):
        assert stores_score({"stores_level": "light"}) == 0.8

    def test_adequate(self):
        assert stores_score({"stores_level": "adequate"}) == 0.2

    def test_heavy(self):
        assert stores_score({"stores_level": "heavy"}) == 0.0


class TestDiseaseScore:
    def test_no_disease(self):
        assert disease_score({"disease_signs": []}) == 0.0

    def test_one_disease(self):
        assert disease_score({"disease_signs": ["chalkbrood"]}) == 0.5

    def test_multiple_diseases(self):
        assert disease_score({"disease_signs": ["chalkbrood", "EFB", "nosema"]}) == 1.0


class TestBroodScore:
    def test_solid(self):
        assert brood_score({"brood_pattern": "solid"}) == 0.0

    def test_spotty(self):
        assert brood_score({"brood_pattern": "spotty"}) == 0.6

    def test_absent(self):
        assert brood_score({"brood_pattern": "absent"}) == 1.0


class TestInspectionToContext:
    def test_converts_inspection(self):
        insp = HiveInspection(
            hive_id="h1", date="2026-03-17",
            varroa_count=5.0, queen_seen=False, eggs_seen=True,
            brood_pattern="spotty", honey_stores="light",
            queen_cells=2, disease_signs=["chalkbrood"],
        )
        ctx = inspection_to_context(insp)
        assert ctx["varroa_per_100"] == 5.0
        assert ctx["queen_seen"] is False
        assert ctx["stores_level"] == "light"
        assert ctx["disease_signs"] == ["chalkbrood"]


class TestFullScoring:
    def test_healthy_hive_scores_low(self):
        ctx = {"varroa_per_100": 1, "queen_seen": True, "eggs_seen": True,
               "brood_pattern": "solid", "stores_level": "adequate",
               "queen_cells": 0, "disease_signs": []}
        composite, breakdown = score_terms(HIVE_TERMS, ctx)
        assert composite < 18  # P3

    def test_troubled_hive_scores_high(self):
        ctx = {"varroa_per_100": 8, "queen_seen": False, "eggs_seen": False,
               "brood_pattern": "absent", "stores_level": "light",
               "queen_cells": 5, "disease_signs": ["EFB", "chalkbrood"]}
        composite, breakdown = score_terms(HIVE_TERMS, ctx)
        assert composite >= 80  # P0

    def test_mode_weights_amplify(self):
        ctx = {"varroa_per_100": 5, "queen_seen": True, "eggs_seen": True,
               "brood_pattern": "solid", "stores_level": "adequate",
               "queen_cells": 0, "disease_signs": []}
        base, _ = score_terms(HIVE_TERMS, ctx)
        fall_weights = get_mode_weights("fall")
        amplified, _ = score_terms(HIVE_TERMS, ctx, mode_weights=fall_weights)
        # Fall amplifies w_varroa by 1.5x, so score should increase
        assert amplified > base


class TestModes:
    def test_detect_spring(self):
        assert detect_mode("2026-03-17") == "spring"

    def test_detect_nectar_flow(self):
        assert detect_mode("2026-06-15") == "nectar_flow"

    def test_detect_fall(self):
        assert detect_mode("2026-09-01") == "fall"

    def test_detect_winter(self):
        assert detect_mode("2026-01-15") == "winter"

    def test_all_modes_have_weights(self):
        for mode in MODES:
            weights = get_mode_weights(mode)
            assert isinstance(weights, dict)
