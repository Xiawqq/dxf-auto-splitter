"""DXF 拆图项目的新入口。

直接运行时打开 DXF 文件选择窗口；选择文件后识别图框并生成保守分割结果。
跨边界和无法判断的实体进入共享内容文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audit import audit_dxf, default_report_path
from frame_detection import debug_report_markdown, detect_simple_frames, report_markdown
from file_picker import choose_dxf_file
from splitter import split_dxf, split_report_markdown


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "原始文件"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "输出文件"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DXF 拆图项目（当前阶段：坐标与实体事实审计）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit", help="审计 DXF 的布局、实体边界和块变换，不修改源文件"
    )
    audit_parser.add_argument("input", type=Path, help="输入 DXF 文件")
    audit_parser.add_argument(
        "-o", "--output", type=Path, help="审计 JSON 输出路径，默认与输入文件同目录"
    )
    audit_parser.add_argument(
        "--include-entities",
        action="store_true",
        help="将每个模型空间实体的边界写入报告；大文件报告会明显增大",
    )
    return parser


def run_audit(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    if not source.is_file():
        print(f"错误：文件不存在：{source}", file=sys.stderr)
        return 2
    if source.suffix.lower() != ".dxf":
        print("错误：当前阶段只审计 DXF，暂不处理 DWG。", file=sys.stderr)
        return 2

    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_report_path(source)
    )
    try:
        report = audit_dxf(source, include_entities=args.include_entities)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as error:
        print(f"审计失败：{error}", file=sys.stderr)
        return 1

    model = report["model_space"]
    print(f"审计完成：{source.name}")
    print(f"  模型空间实体：{model['entity_count']}")
    print(f"  可计算边界：{model['bounded_entity_count']}")
    print(f"  不可计算边界：{model['unbounded_entity_count']}")
    print(f"  INSERT 数量：{len(model['inserts'])}")
    print(f"  布局数量：{len(report['layouts'])}")
    print(f"  报告：{destination}")
    return 0


def run_file_picker() -> int:
    try:
        source = choose_dxf_file(DEFAULT_INPUT_DIR)
    except Exception as error:
        print(f"打开文件选择窗口失败：{error}", file=sys.stderr)
        return 1

    if source is None:
        print("未选择文件，程序结束。")
        return 0

    print(f"已选择 DXF：{source}")
    output_dir = DEFAULT_OUTPUT_DIR / source.stem
    try:
        report = detect_simple_frames(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "识别数据.json"
        markdown_path = output_dir / "识别报告.md"
        debug_markdown_path = output_dir / "识别调试报告.md"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        markdown_path.write_text(report_markdown(report), encoding="utf-8")
        debug_markdown_path.write_text(
            debug_report_markdown(report), encoding="utf-8"
        )
    except Exception as error:
        print(f"图框候选识别失败：{error}", file=sys.stderr)
        return 1

    print(f"当前发现 {report['candidate_count']} 个正式图框候选。")
    print(f"另记录 {report['rectangle_candidate_count']} 个内部矩形，不计入图框。")
    print(f"识别报告：{markdown_path}")
    print(f"调试报告：{debug_markdown_path}")
    print(f"识别数据：{json_path}")

    if report["candidate_count"] == 0:
        print("未发现可靠图框，暂不执行分割。")
        return 0

    try:
        split_report = split_dxf(source, output_dir)
        split_json_path = output_dir / "拆分数据.json"
        split_markdown_path = output_dir / "拆分报告.md"
        split_json_path.write_text(
            json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        split_markdown_path.write_text(
            split_report_markdown(split_report), encoding="utf-8"
        )
    except Exception as error:
        print(f"DXF 分割失败：{error}", file=sys.stderr)
        return 1

    print(f"已生成 {len(split_report['outputs'])} 个输出文件。")
    print(f"拆分报告：{split_markdown_path}")
    print(f"拆分数据：{split_json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        return run_file_picker()

    args = build_parser().parse_args(arguments)
    if args.command == "audit":
        return run_audit(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
