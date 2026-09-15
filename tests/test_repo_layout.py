"""T0.1 验收自检：仓库目录结构符合任务书第 1 节。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS = [
    "LICENSE",
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "docs/architecture.md",
    "docs/licenses.md",
    "docs/roadmap.md",
    "docs/relation-to-paper.md",
    "annotator/server",
    "annotator/web",
    "converter/rq_convert",
    "converter/ops/ncnn_ops.json",
    "deployer",
    "runtime/pc_sim",
    "runtime/board",
    "demo/neu_det",
    "demo/sorting_arm",
    "scripts",
    "tests",
    ".github/workflows/ci.yml",
]


def test_repo_layout_complete():
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    assert not missing, f"缺少任务书要求的路径: {missing}"


def test_license_is_apache2():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0" in text
