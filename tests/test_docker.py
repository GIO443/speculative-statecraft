import re

import pytest
import yaml
from bench.config import pinned_image

from spec.collect import REPO_ROOT


def test_trainer_uses_serving_pin() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["trainer"]["build"]["args"]["VLLM_IMAGE"] == pinned_image()


@pytest.mark.parametrize("module", ["extract_hidden", "draft_head", "export"])
def test_container_modules_are_container_safe(module: str) -> None:
    # The trainer container has no sim/bench, nor spec.collect (which needs them).
    source = (REPO_ROOT / "spec" / f"{module}.py").read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.MULTILINE)
    assert imports, "no imports found"
    banned = [m for m in imports if m.split(".")[0] in {"sim", "bench"} or m == "spec.collect"]
    assert not banned


def test_experiments_mount_the_compose_data_volume() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    name = compose["volumes"]["spec-data"]["name"]
    for path in (REPO_ROOT / "configs" / "experiments").glob("*.yaml"):
        exp = yaml.safe_load(path.read_text(encoding="utf-8"))
        for v in exp["server"].get("volumes", []):
            assert v.split(":")[0] == name, path.name
