from pathlib import Path

import bench.config
import sim.engine


def test_statecraft_serving_is_editable_checkout() -> None:
    # Editable install: REPO_ROOT must be the sibling checkout, which holds configs/.
    assert (bench.config.REPO_ROOT / "configs").is_dir()
    assert Path(sim.engine.__file__).resolve().is_relative_to(bench.config.REPO_ROOT.resolve())
