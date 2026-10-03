import re

import yaml
from bench.config import pinned_image

from spec.collect import REPO_ROOT


def test_trainer_uses_serving_pin() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["trainer"]["build"]["args"]["VLLM_IMAGE"] == pinned_image()


def test_extract_hidden_is_container_safe() -> None:
    # The trainer container has no sim/bench (nor spec.collect, which needs them).
    source = (REPO_ROOT / "spec" / "extract_hidden.py").read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.MULTILINE)
    assert imports, "no imports found"
    assert not [m for m in imports if m.split(".")[0] in {"sim", "bench", "spec"}]
