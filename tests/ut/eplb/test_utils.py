import unittest
from types import SimpleNamespace

import torch

from vllm_ascend.eplb.utils import model_register


class FakeExperts:
    def __init__(self, load_value):
        self.moe_load = torch.tensor([load_value], dtype=torch.float32)
        self.clear_calls = 0

    def clear_moe_load(self):
        self.clear_calls += 1


class FakeLayer:
    def __init__(self, load_value):
        self.mlp = SimpleNamespace(experts=FakeExperts(load_value))


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                first_k_dense_replace=1,
                num_hidden_layers=3,
            )
        )
        self.model = SimpleNamespace(
            layers=[
                FakeLayer(0.0),
                FakeLayer(1.0),
                FakeLayer(2.0),
            ]
        )


class TestEplbUtils(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        model_register(self.model)

    def test_get_all_moe_loads_with_nested_text_config(self):
        moe_loads = self.model.get_all_moe_loads()

        expected = torch.tensor([[1.0], [2.0]])
        self.assertTrue(torch.equal(moe_loads, expected))

    def test_clear_all_moe_loads_with_nested_text_config(self):
        self.model.clear_all_moe_loads()

        self.assertEqual(self.model.model.layers[0].mlp.experts.clear_calls, 0)
        self.assertEqual(self.model.model.layers[1].mlp.experts.clear_calls, 1)
        self.assertEqual(self.model.model.layers[2].mlp.experts.clear_calls, 1)


if __name__ == "__main__":
    unittest.main()
