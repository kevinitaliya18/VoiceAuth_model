import unittest

from pydantic import ValidationError

from api import EngineRequest, engine_only
from risk_engine import RiskEngine


class EngineRequestTests(unittest.TestCase):
    def test_incomplete_model_outputs_are_rejected(self):
        incomplete = (
            ({"spoof": {}}, "p_spoof"),
            ({"speaker": {}}, "similarity"),
            ({"intent": {}}, "scam_probability"),
        )
        for payload, field in incomplete:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError) as error:
                    EngineRequest.model_validate(payload)
                self.assertIn(field, str(error.exception))

    def test_valid_model_outputs_are_assessed(self):
        request = EngineRequest.model_validate({
            "spoof": {"p_spoof": 0.12},
            "speaker": {"similarity": 0.91, "threshold": 0.86},
            "intent": {"scam_probability": 0.08, "flags": {}},
            "metadata": {"request_type": "none"},
        })

        class PipelineStub:
            engine = RiskEngine()

        result = engine_only(request, PipelineStub())
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["missing_inputs"], [])


if __name__ == "__main__":
    unittest.main()
