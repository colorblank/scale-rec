from __future__ import annotations

import pytest
import yaml

from train.core.config import ModelConfig


def _contract():
    return {
        "version": 1,
        "graph": {
            "towers": [{"name": "score", "kind": "score"}],
            "relations": [],
        },
        "objectives": [],
        "metrics": [],
        "outputs": [{"name": "score", "source": "score"}],
    }


def test_model_config_accepts_native_output_contract_without_task_config():
    config = ModelConfig.from_dict({"type": "rankmixer", "output_contract": _contract()})

    assert config.params["output_contract"]["version"] == 1


def test_model_config_rejects_mixed_native_and_legacy_contracts():
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelConfig.from_dict(
            {
                "type": "rankmixer",
                "output_contract": _contract(),
                "task_config": {"towers": []},
            }
        )


def test_model_config_rejects_unknown_output_contract_version():
    raw = _contract()
    raw["version"] = 2

    with pytest.raises(ValueError, match="version"):
        ModelConfig.from_dict(
            {"type": "lr", "output_contract": yaml.safe_load(yaml.safe_dump(raw))}
        )
