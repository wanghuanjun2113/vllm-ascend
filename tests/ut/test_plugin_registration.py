import sys

import transformers

import vllm_ascend
import vllm_ascend.models


def test_register_model_without_hunyuan_processor(monkeypatch):
    monkeypatch.delattr(transformers, "HunYuanVLProcessor", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "vllm_ascend.patch.hunyuan_vl_processor_compat",
        raising=False,
    )

    registered = False

    def register_models():
        nonlocal registered
        registered = True

    monkeypatch.setattr(vllm_ascend.models, "register_model", register_models)

    vllm_ascend.register_model()

    assert registered
    assert "vllm_ascend.patch.hunyuan_vl_processor_compat" not in sys.modules
