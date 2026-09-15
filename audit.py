"""DXF 源文件事实审计。

这一模块只读取源文件并输出事实，不做图框猜测、实体归属和文件导出。
它的目标是先暴露坐标空间、块变换和边界可靠性，给后续算法提供可验证输入。
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO, TextIOWrapper
import math
from pathlib import Path

import ezdxf
from ezdxf import bbox


def default_report_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_audit.json")


def _dxf_get(dxf, name, default=None):
    """读取可选 DXF 属性；特殊实体可能没有标准实体共有字段。"""
    try:
        return dxf.get(name, default)
    except Exception:
        return default


def _vector(value):
    if value is None:
        return None
    try:
        return [float(value.x), float(value.y), float(value.z)]
    except AttributeError:
        try:
            values = list(value)
            return [float(values[0]), float(values[1]), float(values[2])]  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            return None


def _finite_box(entity, cache):
    try:
        extents = bbox.extents([entity], fast=False, cache=cache)
        if not extents.has_data:
            return None, "no_data"
        values = [
            float(extents.extmin.x),
            float(extents.extmin.y),
            float(extents.extmax.x),
            float(extents.extmax.y),
        ]
    except Exception as error:
        return None, type(error).__name__
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite"
    return values, None


def _insert_record(entity, box):
    dxf = entity.dxf
    return {
        "handle": dxf.handle,
        "name": _dxf_get(dxf, "name") or "",
        "layer": _dxf_get(dxf, "layer") or "",
        "bbox": box,
        "insert": _vector(_dxf_get(dxf, "insert")),
        "rotation": float(_dxf_get(dxf, "rotation", 0.0) or 0.0),
        "scale": {
            "x": float(_dxf_get(dxf, "xscale", 1.0) or 1.0),
            "y": float(_dxf_get(dxf, "yscale", 1.0) or 1.0),
            "z": float(_dxf_get(dxf, "zscale", 1.0) or 1.0),
        },
        "array": {
            "rows": int(_dxf_get(dxf, "row_count", 1) or 1),
            "columns": int(_dxf_get(dxf, "column_count", 1) or 1),
        },
    }


def _layout_record(layout):
    type_counts = Counter(entity.dxftype() for entity in layout)
    return {
        "name": layout.name,
        "entity_count": len(layout),
        "entity_types": dict(sorted(type_counts.items())),
    }


def _record_value(record, code):
    for current_code, value in record:
        if current_code == code and value.strip():
            return value.strip().decode("utf-8", errors="replace")
    return None


def _repair_missing_table_names(source: Path):
    """为 TABLES 中缺失 group code 2 的记录补充唯一名称。"""
    lines = source.read_bytes().splitlines(keepends=True)
    pairs = []
    for index in range(0, len(lines) - 1, 2):
        code = lines[index].strip().decode("ascii", errors="ignore")
        pairs.append((code, lines[index + 1]))

    output = bytearray()
    record = []
    record_type = None
    section = None
    table = None
    repair_count = 0

    def flush_record():
        nonlocal record, record_type, table, section, repair_count
        if not record:
            return
        if (
            section == "TABLES"
            and table
            and record_type not in {"TABLE", "ENDTAB", "ENDSEC"}
            and _record_value(record, "2") is None
        ):
            repair_count += 1
            fallback = f"_RECOVERED_{table}_{repair_count}".encode("ascii")
            record.insert(1, ("2", fallback + b"\n"))

        for code, raw_value in record:
            code_line = f"{int(code):3d}\n".encode("ascii")
            output.extend(code_line)
            output.extend(raw_value)

        if record_type == "SECTION":
            section = _record_value(record, "2")
        elif record_type == "ENDSEC":
            section = None
            table = None
        elif section == "TABLES" and record_type == "TABLE":
            table = _record_value(record, "2")
        elif section == "TABLES" and record_type == "ENDTAB":
            table = None
        record = []

    for index, (code, raw_value) in enumerate(pairs):
        if code == "0":
            flush_record()
            record_type = raw_value.decode("utf-8", errors="replace").strip()
            record = [(code, raw_value)]
        else:
            record.append((code, raw_value))
    flush_record()
    return bytes(output), repair_count


def _load_document(source: Path):
    """优先正常读取，失败时只做通用表记录名称修复。"""
    try:
        return ezdxf.readfile(source), {"mode": "normal", "repairs": 0}
    except Exception as original_error:
        repaired_bytes, repair_count = _repair_missing_table_names(source)
        if repair_count == 0:
            raise original_error
        try:
            stream = TextIOWrapper(
                BytesIO(repaired_bytes), encoding="utf-8", errors="surrogateescape"
            )
            document = ezdxf.read(stream)
        except Exception as repaired_error:
            raise RuntimeError(
                f"DXF 读取失败，通用表记录修复后仍无法读取：{repaired_error}"
            ) from repaired_error
        return document, {"mode": "repaired", "repairs": repair_count}


def audit_dxf(source: Path, include_entities: bool = False) -> dict:
    """读取 DXF 并返回 JSON 可序列化的事实报告。"""
    doc, load_info = _load_document(source)
    model = doc.modelspace()
    cache = bbox.Cache()
    type_counts = Counter()
    layer_counts = Counter()
    inserts = []
    unbounded = []
    entities = []
    model_boxes = []

    for entity in model:
        entity_type = entity.dxftype()
        layer = _dxf_get(entity.dxf, "layer") or ""
        type_counts[entity_type] += 1
        layer_counts[layer] += 1
        box, reason = _finite_box(entity, cache)
        if box is None:
            unbounded.append(
                {
                    "handle": entity.dxf.handle,
                    "type": entity_type,
                    "layer": layer,
                    "reason": reason,
                }
            )
        else:
            model_boxes.append(box)

        if entity_type == "INSERT":
            inserts.append(_insert_record(entity, box))

        if include_entities:
            record = {
                "handle": entity.dxf.handle,
                "type": entity_type,
                "layer": layer,
                "bbox": box,
            }
            if reason is not None:
                record["bbox_error"] = reason
            entities.append(record)

    model_extents = None
    if model_boxes:
        model_extents = [
            min(box[0] for box in model_boxes),
            min(box[1] for box in model_boxes),
            max(box[2] for box in model_boxes),
            max(box[3] for box in model_boxes),
        ]

    report = {
        "source": str(source),
        "load": load_info,
        "dxf_version": doc.dxfversion,
        "active_layout": doc.layouts.active_layout().name,
        "header": {
            "insunits": doc.header.get("$INSUNITS"),
            "extmin": _vector(doc.header.get("$EXTMIN")),
            "extmax": _vector(doc.header.get("$EXTMAX")),
        },
        "layouts": [_layout_record(layout) for layout in doc.layouts],
        "model_space": {
            "entity_count": len(model),
            "bounded_entity_count": len(model) - len(unbounded),
            "unbounded_entity_count": len(unbounded),
            "extents": model_extents,
            "entity_types": dict(sorted(type_counts.items())),
            "layers": dict(sorted(layer_counts.items())),
            "inserts": inserts,
            "unbounded_entities": unbounded,
        },
    }
    if include_entities:
        report["model_space"]["entities"] = entities
    return report
