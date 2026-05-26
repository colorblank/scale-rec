from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_generator() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "examples" / "gen_discover_config.py"
    spec = importlib.util.spec_from_file_location("gen_discover_config", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_config_generator_contract() -> None:
    generator = _load_generator()

    full = generator.generate_config()
    item = generator.generate_item_config()
    user = generator.generate_user_config()

    assert len(full["sources"]) == 38
    assert len(full["operators"]) == 69
    assert sum(len(op.get("outputs", [])) for op in full["operators"] if "embed" in op) == 44
    assert len(item["sources"]) == 18
    assert len(user["sources"]) == 28
    assert all("embed" not in source for source in full["sources"])


def test_discover_config_operator_names_are_unique() -> None:
    generator = _load_generator()
    operators = generator.generate_config()["operators"]
    names = [op["name"] for op in operators]

    assert len(names) == len(set(names))
