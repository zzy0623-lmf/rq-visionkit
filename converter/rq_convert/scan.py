# -*- coding: utf-8 -*-
"""RQ-VisionKit M2 算子兼容扫描（T1.5）。

    rq-scan model.onnx

解析 ONNX 计算图，逐节点对照 ncnn_ops.json，输出三档报告：
    ✅ 直接支持 / ⚠️ 需替换（含建议） / ❌ 不支持
"""

import argparse
import json
from collections import Counter
from pathlib import Path

OPS_JSON = Path(__file__).resolve().parent.parent / "ops" / "ncnn_ops.json"


def load_ops(ops_path: str | None = None) -> dict:
    p = Path(ops_path) if ops_path else OPS_JSON
    return json.loads(p.read_text(encoding="utf-8"))


def classify_ops(op_counts: dict, ops: dict) -> dict:
    """纯逻辑：把 {op_type: count} 对照 ncnn_ops.json 分成三档（不依赖 onnx）。"""
    layers = set(ops["layers"])
    supported = ops["onnx_supported"]
    replace = ops["onnx_replace"]
    result = {"supported": [], "replace": [], "unsupported": []}
    for op, count in sorted(op_counts.items()):
        if op in supported:
            result["supported"].append({"op": op, "ncnn": supported[op], "count": count})
        elif op in replace:
            r = replace[op]
            result["replace"].append({"op": op, "count": count, "to": r["to"], "hint": r["hint"]})
        elif op.lower() in layers:
            result["supported"].append({"op": op, "ncnn": op, "count": count})
        else:
            result["unsupported"].append({"op": op, "count": count})
    return result


def scan(model_path: str, ops_path: str | None = None) -> dict:
    """解析 ONNX 模型并扫描算子。"""
    import onnx

    ops = load_ops(ops_path)
    model = onnx.load(model_path)
    op_counts = Counter(node.op_type for node in model.graph.node)
    report = {
        "model": str(model_path),
        "ncnn_version": ops["ncnn_version"],
        "source": ops["source"],
        "total_ops": len(model.graph.node),
        "distinct_ops": len(op_counts),
    }
    report.update(classify_ops(op_counts, ops))
    return report


def render(report: dict) -> str:
    lines = [
        f"模型: {report['model']}",
        f"对照 NCNN {report['ncnn_version']}（来源: {report['source']}）",
        f"共 {report['total_ops']} 个节点 / {report['distinct_ops']} 种算子",
        "",
    ]
    lines.append(f"✅ 直接支持（{len(report['supported'])} 种）:")
    for s in report["supported"]:
        lines.append(f"    {s['op']} -> {s['ncnn']}（×{s['count']}）")
    lines.append(f"⚠️ 需替换（{len(report['replace'])} 种）:")
    for r in report["replace"]:
        lines.append(f"    {r['op']} -> {r['to']}（×{r['count']}）建议: {r['hint']}")
    lines.append(f"❌ 不支持（{len(report['unsupported'])} 种）:")
    for u in report["unsupported"]:
        lines.append(f"    {u['op']}（×{u['count']}）")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rq-scan", description="ONNX 算子兼容扫描（对照 NCNN 支持清单）")
    p.add_argument("model", help="输入 ONNX 模型路径")
    p.add_argument("--ops", help="ncnn_ops.json 路径（默认 converter/ops/ncnn_ops.json）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.model).is_file():
        print(f"模型文件不存在: {args.model}")
        return 2
    report = scan(args.model, args.ops)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
