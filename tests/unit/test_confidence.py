from model_router.confidence import compute_confidence


class TestComputeConfidence:
    def test_empty_dict_returns_zero(self):
        assert compute_confidence({}) == 0.0

    def test_no_confidence_fields_returns_zero(self):
        assert compute_confidence({"title": "Hello", "pages": 5}) == 0.0

    def test_single_confidence_field(self):
        data = {"title": {"value": "Report", "confidence": 0.9}}
        assert compute_confidence(data) == 0.9

    def test_multiple_confidence_fields(self):
        data = {
            "title": {"value": "Report", "confidence": 0.8},
            "author": {"value": "Jane", "confidence": 0.6},
        }
        assert abs(compute_confidence(data) - 0.7) < 1e-9

    def test_nested_dict_confidence(self):
        data = {
            "section": {
                "heading": {"value": "Intro", "confidence": 0.95},
                "body": {"value": "...", "confidence": 0.75},
            }
        }
        assert abs(compute_confidence(data) - 0.85) < 1e-9

    def test_list_of_items(self):
        data = {
            "items": [
                {"value": "a", "confidence": 0.5},
                {"value": "b", "confidence": 1.0},
            ]
        }
        assert abs(compute_confidence(data) - 0.75) < 1e-9

    def test_float_confidence(self):
        data = {"field": {"value": 42, "confidence": 0.333}}
        assert abs(compute_confidence(data) - 0.333) < 1e-9

    def test_integer_confidence_treated_as_float(self):
        data = {"field": {"value": "x", "confidence": 1}}
        assert compute_confidence(data) == 1.0

    def test_mixed_nested_and_flat(self):
        data = {
            "title": {"value": "Doc", "confidence": 0.8},
            "meta": {
                "tags": [
                    {"value": "tag1", "confidence": 0.6},
                    {"value": "tag2", "confidence": 0.4},
                ]
            },
        }
        expected = (0.8 + 0.6 + 0.4) / 3
        assert abs(compute_confidence(data) - expected) < 1e-9
