"""第一阶段的简单图框识别。

当前在模型空间中分析重复 INSERT 块族，并把闭合四边正交多段线作为内部矩形记录。
本模块只生成候选和报告，不判断实体归属，也不导出拆分 DXF。
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

from ezdxf import bbox as ezdxf_bbox

from audit import _dxf_get, _load_document


def _polyline_points(entity):
    if entity.dxftype() == "LWPOLYLINE":
        points = [
            (float(point[0]), float(point[1]), float(point[2]))
            for point in entity.get_points("xyb")
        ]
        closed = bool(entity.closed)
    else:
        points = []
        for vertex in entity.vertices:
            location = vertex.dxf.location
            points.append(
                (
                    float(location.x),
                    float(location.y),
                    float(_dxf_get(vertex.dxf, "bulge", 0.0) or 0.0),
                )
            )
        closed = bool(entity.is_closed)

    if len(points) > 1 and points[0][:2] == points[-1][:2]:
        points.pop()
    return points, closed


def _polyline_width(entity):
    """Read geometric polyline width without assuming one DXF schema."""
    widths = []
    if entity.dxftype() == "LWPOLYLINE":
        constant = _dxf_get(entity.dxf, "const_width")
        if constant is not None:
            try:
                widths.append(abs(float(constant)))
            except (TypeError, ValueError):
                pass
        try:
            points = entity.get_points("xyseb")
        except Exception:
            points = []
        for point in points:
            for value in point[2:4]:
                try:
                    widths.append(abs(float(value)))
                except (TypeError, ValueError):
                    continue
    else:
        for vertex in entity.vertices:
            for name in ("start_width", "end_width"):
                value = _dxf_get(vertex.dxf, name)
                if value is None:
                    continue
                try:
                    widths.append(abs(float(value)))
                except (TypeError, ValueError):
                    continue

    return max(widths, default=0.0), len(widths)


def _rectangle_candidate(entity):
    try:
        points, closed = _polyline_points(entity)
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if not closed or len(points) != 4:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0:
        return None

    tolerance = max(width, height, 1.0) * 1e-9
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        dx = next_point[0] - point[0]
        dy = next_point[1] - point[1]
        if abs(dx) > tolerance and abs(dy) > tolerance:
            return None
        if abs(point[2]) > tolerance:
            return None

    expected_corners = {
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
    }
    actual_corners = {(point[0], point[1]) for point in points}
    if actual_corners != expected_corners:
        return None

    raw_lineweight = _dxf_get(entity.dxf, "lineweight")
    try:
        lineweight = float(raw_lineweight) if raw_lineweight is not None else None
    except (TypeError, ValueError):
        lineweight = None
    line_width, width_sample_count = _polyline_width(entity)

    return {
        "entity_handle": entity.dxf.handle,
        "entity_type": entity.dxftype(),
        "layer": _dxf_get(entity.dxf, "layer") or "",
        "bbox": [xmin, ymin, xmax, ymax],
        "width": width,
        "height": height,
        "aspect_ratio": max(width, height) / min(width, height),
        "a_series_like": abs(
            max(width, height) / min(width, height) - math.sqrt(2)
        ) <= 0.08,
        "line_width": line_width,
        "width_sample_count": width_sample_count,
        "width_source": (
            "polyline_geometry"
            if line_width > 0
            else "not_recorded"
        ),
        "lineweight": lineweight,
        # Keep this source value shared by model-space and block-definition
        # rectangles; downstream border-pair logic relies on it.
        "source": "closed_polyline",
        "status": "几何矩形候选",
        "note": "已确认四边闭合且正交，是否为正式图框需结合样本复核",
    }


def _line_rectangle_candidates(block):
    """Find rectangles drawn as four independent orthogonal LINE entities."""
    horizontal_groups = defaultdict(list)
    vertical_groups = defaultdict(list)
    line_records = []
    x_coordinates = []
    y_coordinates = []

    for entity in block.query("LINE"):
        try:
            start = entity.dxf.start
            end = entity.dxf.end
            x1, y1 = float(start.x), float(start.y)
            x2, y2 = float(end.x), float(end.y)
        except (AttributeError, TypeError, ValueError):
            continue

        x_coordinates.extend((x1, x2))
        y_coordinates.extend((y1, y2))
        line_records.append((entity, x1, y1, x2, y2))

    if not line_records:
        return []

    coordinate_tolerance = max(
        max(max(x_coordinates) - min(x_coordinates),
            max(y_coordinates) - min(y_coordinates), 1.0)
        * 1e-9,
        1e-12,
    )

    def key(value):
        return round(value / coordinate_tolerance)

    for entity, x1, y1, x2, y2 in line_records:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) <= coordinate_tolerance and abs(dy) > coordinate_tolerance:
            ymin, ymax = sorted((y1, y2))
            vertical_groups[(key(ymin), key(ymax))].append(
                {
                    "entity": entity,
                    "x": (x1 + x2) / 2.0,
                    "xmin": min(x1, x2),
                    "xmax": max(x1, x2),
                }
            )
        elif abs(dy) <= coordinate_tolerance and abs(dx) > coordinate_tolerance:
            xmin, xmax = sorted((x1, x2))
            horizontal_groups[(key(xmin), key(xmax))].append(
                {
                    "entity": entity,
                    "y": (y1 + y2) / 2.0,
                    "ymin": min(y1, y2),
                    "ymax": max(y1, y2),
                }
            )

    candidates = []
    seen = set()
    for (xmin_key, xmax_key), horizontals in horizontal_groups.items():
        if len(horizontals) < 2:
            continue

        for first, second in combinations(horizontals, 2):
            ymin, ymax = sorted((first["y"], second["y"]))
            verticals = vertical_groups.get((key(ymin), key(ymax)), [])
            if not verticals:
                continue

            boundary_verticals = [
                vertical
                for vertical in verticals
                if key(vertical["x"]) in (xmin_key, xmax_key)
            ]
            left = [vertical for vertical in boundary_verticals
                    if key(vertical["x"]) == xmin_key]
            right = [vertical for vertical in boundary_verticals
                     if key(vertical["x"]) == xmax_key]
            if not left or not right:
                continue

            geometry_key = (xmin_key, xmax_key, key(ymin), key(ymax))
            if geometry_key in seen:
                continue
            seen.add(geometry_key)

            boundary_entities = [
                first["entity"],
                second["entity"],
                left[0]["entity"],
                right[0]["entity"],
            ]
            # The horizontal group's coordinates are the rectangle's x range;
            # its two line positions are the rectangle's y range.
            x_min = min(
                first["entity"].dxf.start.x,
                first["entity"].dxf.end.x,
                second["entity"].dxf.start.x,
                second["entity"].dxf.end.x,
            )
            x_max = max(
                first["entity"].dxf.start.x,
                first["entity"].dxf.end.x,
                second["entity"].dxf.start.x,
                second["entity"].dxf.end.x,
            )
            y_min, y_max = min(ymin, ymax), max(ymin, ymax)
            width, height = x_max - x_min, y_max - y_min
            lineweights = [
                float(_dxf_get(entity.dxf, "lineweight"))
                for entity in boundary_entities
                if _dxf_get(entity.dxf, "lineweight") is not None
            ]
            layers = {
                str(_dxf_get(entity.dxf, "layer") or "")
                for entity in boundary_entities
            }
            candidates.append(
                {
                    "entity_handle": "+".join(
                        sorted(entity.dxf.handle for entity in boundary_entities)
                    ),
                    "entity_handles": [entity.dxf.handle for entity in boundary_entities],
                    "entity_type": "LINE_RECTANGLE",
                    "layer": next(iter(layers)) if len(layers) == 1 else "多个图层",
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "width": float(width),
                    "height": float(height),
                    "aspect_ratio": max(width, height) / min(width, height),
                    "a_series_like": abs(
                        max(width, height) / min(width, height) - math.sqrt(2)
                    ) <= 0.08,
                    "line_width": 0.0,
                    "width_sample_count": 0,
                    "width_source": "lineweight_only",
                    "lineweight": max(lineweights) if lineweights else None,
                    "source": "four_lines",
                    "status": "四条独立 LINE 矩形候选",
                    "note": "四条正交 LINE 共用矩形边界，需结合内侧矩形确认正式图框",
                }
            )

    return candidates


def _annotate_border_evidence(candidates):
    """Mark explicit border-width evidence without declaring formal frames."""
    geometric_widths = [
        candidate["line_width"]
        for candidate in candidates
        if candidate.get("line_width", 0.0) > 0
    ]
    lineweights = [
        candidate["lineweight"]
        for candidate in candidates
        if candidate.get("lineweight") is not None
        and candidate["lineweight"] > 0
    ]

    def relative_rank(value, values):
        if value is None or not values:
            return None
        return sum(item <= value for item in values) / len(values)

    for candidate in candidates:
        geometric_width = candidate.get("line_width", 0.0)
        lineweight = candidate.get("lineweight")
        evidence = []
        if geometric_width > 0:
            evidence.append("geometric_width")
        if lineweight is not None and lineweight > 0:
            evidence.append("lineweight")

        candidate["border_width_evidence"] = evidence
        candidate["geometric_width_rank"] = relative_rank(
            geometric_width, geometric_widths
        )
        candidate["lineweight_rank"] = relative_rank(lineweight, lineweights)
        candidate["border_priority"] = (
            "explicit_width_evidence" if evidence else "geometry_only"
        )

    return candidates


def detect_model_frames(doc):
    """从模型空间提取多种表达形式的矩形候选，并按阅读顺序编号。"""
    candidates = []
    for entity in doc.modelspace().query("LWPOLYLINE POLYLINE"):
        candidate = _rectangle_candidate(entity)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(_line_rectangle_candidates(doc.modelspace()))

    _annotate_border_evidence(candidates)
    candidates.sort(key=lambda item: (-item["bbox"][3], item["bbox"][0]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["index"] = index
    return candidates


def _insert_record(entity):
    """读取一个 INSERT 的变换后范围，失败时返回 None。"""
    try:
        extents = ezdxf_bbox.extents([entity], fast=False)
        if extents.extmin is None or extents.extmax is None:
            return None
    except Exception:
        # 单个异常块不应阻断整份 DXF 的识别。
        return None

    xmin, ymin = float(extents.extmin.x), float(extents.extmin.y)
    xmax, ymax = float(extents.extmax.x), float(extents.extmax.y)
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0:
        return None

    return {
        "entity_handle": entity.dxf.handle,
        "entity_type": "INSERT",
        "block_name": str(entity.dxf.get("name") or ""),
        "layer": str(entity.dxf.get("layer") or ""),
        "bbox": [xmin, ymin, xmax, ymax],
        "width": width,
        "height": height,
        "aspect_ratio": max(width, height) / min(width, height),
    }


def _line_is_thicker(inner, outer):
    if inner["line_width"] > outer["line_width"]:
        return True
    if inner["lineweight"] is not None and outer["lineweight"] is not None:
        return inner["lineweight"] > outer["lineweight"]
    return False


def _close_parallel_track(pair):
    """判断两条嵌套矩形是否足够接近，可能共同表现为一条粗边。"""
    reference_size = min(pair["inner_width"], pair["inner_height"])
    if reference_size <= 0:
        return False
    # 使用相对间距，不依赖图纸的绝对单位或纸张尺寸。
    return max(pair["margins"]) / reference_size <= 0.02


def _get_block_definition(doc, block_name):
    """Resolve block names exactly, including anonymous names containing '$'."""
    for block in doc.blocks:
        if getattr(block, "name", None) == block_name:
            return block
    try:
        return doc.blocks.get(block_name)
    except Exception:
        return None


def _block_border_profile(doc, block_name, _visited=None):
    """查找块定义中的连续嵌套矩形，以及内框线宽证据。"""
    visited = set(_visited or ())
    if block_name in visited:
        return {
            "nested_border": False,
            "thick_inner": False,
            "parallel_track": False,
            "nested_depth": 0,
            "border_evidence": "none",
            "pair": None,
        }
    visited.add(block_name)
    block = _get_block_definition(doc, block_name)
    if block is None:
        return {
            "nested_border": False,
            "thick_inner": False,
            "parallel_track": False,
            "nested_depth": 0,
            "border_evidence": "none",
            "pair": None,
        }

    rectangles = [
        candidate
        for entity in block.query("LWPOLYLINE POLYLINE")
        if (candidate := _rectangle_candidate(entity)) is not None
    ]
    rectangles.extend(_line_rectangle_candidates(block))
    pairs = []
    for first, second in combinations(rectangles, 2):
        first_box = first["bbox"]
        second_box = second["bbox"]
        if (
            first_box[0] < second_box[0]
            and first_box[1] < second_box[1]
            and first_box[2] > second_box[2]
            and first_box[3] > second_box[3]
        ):
            outer, inner = first, second
        elif (
            second_box[0] < first_box[0]
            and second_box[1] < first_box[1]
            and second_box[2] > first_box[2]
            and second_box[3] > first_box[3]
        ):
            outer, inner = second, first
        else:
            continue

        margins = [
            inner["bbox"][0] - outer["bbox"][0],
            inner["bbox"][1] - outer["bbox"][1],
            outer["bbox"][2] - inner["bbox"][2],
            outer["bbox"][3] - inner["bbox"][3],
        ]
        if min(margins) <= 0:
            continue

        pairs.append(
            {
                "outer_handle": outer["entity_handle"],
                "inner_handle": inner["entity_handle"],
                "outer_bbox": outer["bbox"],
                "inner_bbox": inner["bbox"],
                "outer_line_width": outer["line_width"],
                "inner_line_width": inner["line_width"],
                "outer_lineweight": outer["lineweight"],
                "inner_lineweight": inner["lineweight"],
                "outer_source": outer.get("source", "unknown"),
                "inner_source": inner.get("source", "unknown"),
                "margins": margins,
                "outer_area": outer["width"] * outer["height"],
                "line_width_evidence": _line_is_thicker(inner, outer),
            }
        )

    for pair in pairs:
        pair["inner_width"] = pair["inner_bbox"][2] - pair["inner_bbox"][0]
        pair["inner_height"] = pair["inner_bbox"][3] - pair["inner_bbox"][1]
        pair["parallel_track_evidence"] = _close_parallel_track(pair)

    if not pairs:
        nested_profiles = []
        for entity in block.query("INSERT"):
            nested_name = str(entity.dxf.get("name") or "")
            if not nested_name:
                continue
            nested_profile = _block_border_profile(
                doc, nested_name, _visited=visited
            )
            if nested_profile.get("nested_border"):
                nested_profiles.append(nested_profile)
        if nested_profiles:
            nested_profile = max(
                nested_profiles,
                key=lambda profile: (
                    bool(profile.get("thick_inner")),
                    bool(profile.get("parallel_track")),
                    profile.get("nested_depth", 0),
                ),
            )
            nested_profile = dict(nested_profile)
            nested_profile["nested_depth"] = (
                nested_profile.get("nested_depth", 0) + 1
            )
            nested_profile["border_evidence"] = (
                "nested_insert_" + nested_profile.get("border_evidence", "none")
            )
            nested_profile["nested_insert"] = True
            return nested_profile
        return {
            "nested_border": False,
            "thick_inner": False,
            "parallel_track": False,
            "nested_depth": 0,
            "border_evidence": "none",
            "pair": None,
        }

    # 计算最长的连续包含链。局部构造块常见的只是一个内外小框，
    # 正式图框通常还会有第三层边界；该结构比单纯面积更有区分度。
    nested_depth = 1
    for rectangle in rectangles:
        depth = 1
        current = rectangle
        while True:
            containers = [
                candidate
                for candidate in rectangles
                if candidate is not current
                and candidate["bbox"][0] < current["bbox"][0]
                and candidate["bbox"][1] < current["bbox"][1]
                and candidate["bbox"][2] > current["bbox"][2]
                and candidate["bbox"][3] > current["bbox"][3]
            ]
            if not containers:
                break
            current = min(
                containers,
                key=lambda candidate: candidate["width"] * candidate["height"],
            )
            depth += 1
        nested_depth = max(nested_depth, depth)

    best_pair = max(
        pairs,
        key=lambda pair: (
            pair["line_width_evidence"],
            pair["parallel_track_evidence"],
            pair["outer_area"],
        ),
    )
    thick_inner = any(pair["line_width_evidence"] for pair in pairs)
    parallel_track = any(pair["parallel_track_evidence"] for pair in pairs)
    border_evidence = (
        "outer_thin_inner_thick"
        if thick_inner
        else "outer_thin_inner_thick_geometric"
        if parallel_track
        else "complete_pair_with_nested_support"
    )
    return {
        "nested_border": nested_depth >= 3 or thick_inner or parallel_track,
        "thick_inner": thick_inner,
        "parallel_track": parallel_track,
        "nested_depth": nested_depth,
        "border_evidence": border_evidence,
        "pair": best_pair,
    }


def _same_candidate_box(first, second) -> bool:
    scale = max(
        first["width"], first["height"], second["width"], second["height"], 1.0
    )
    tolerance = scale * 1e-9
    return all(
        abs(left - right) <= tolerance
        for left, right in zip(first["bbox"], second["bbox"])
    )


def _candidate_quality(candidate):
    pair = candidate.get("border_pair") or {}
    return (
        pair.get("outer_source") == "closed_polyline",
        bool(candidate.get("thick_inner")),
        bool(candidate.get("parallel_track")),
        candidate.get("nested_depth", 0),
    )


def _deduplicate_candidates(candidates):
    """Keep one INSERT when several blocks describe the same visual frame."""
    unique = []
    for candidate in candidates:
        candidate_handles = list(candidate.get("entity_handles") or [])
        if not candidate_handles:
            handle = candidate.get("entity_handle")
            if handle and "+" not in handle:
                candidate_handles = [handle]
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(unique)
                if _same_candidate_box(existing, candidate)
            ),
            None,
        )
        if duplicate_index is None:
            if candidate_handles:
                candidate["entity_handles"] = candidate_handles
            unique.append(candidate)
            continue

        existing = unique[duplicate_index]
        existing_handles = list(existing.get("entity_handles") or [])
        combined_handles = list(dict.fromkeys(existing_handles + candidate_handles))
        if _candidate_quality(candidate) > _candidate_quality(existing):
            if combined_handles:
                candidate["entity_handles"] = combined_handles
            unique[duplicate_index] = candidate
        elif combined_handles:
            existing["entity_handles"] = combined_handles
    return unique


def _single_frame_shape(doc, block_name):
    """提取单次 INSERT 的最外层矩形和一个内部矩形，不读取文字语义。"""
    block = _get_block_definition(doc, block_name)
    if block is None:
        return None

    rectangles = [
        candidate
        for entity in block.query("LWPOLYLINE POLYLINE")
        if (candidate := _rectangle_candidate(entity)) is not None
    ]
    if len(rectangles) < 2:
        rectangles.extend(_line_rectangle_candidates(block))
    if len(rectangles) < 2:
        return None

    outer = max(rectangles, key=lambda item: item["width"] * item["height"])
    inner_exists = any(
        other is not outer
        and outer["bbox"][0] < other["bbox"][0]
        and outer["bbox"][1] < other["bbox"][1]
        and outer["bbox"][2] > other["bbox"][2]
        and outer["bbox"][3] > other["bbox"][3]
        for other in rectangles
    )
    if not inner_exists:
        return None

    return {
        "outer_aspect_ratio": outer["aspect_ratio"],
        "outer_bbox": outer["bbox"],
        "outer_area": outer["width"] * outer["height"],
        "rectangle_count": len(rectangles),
    }


def _aspect_matches_reference(aspect_ratio, reference_aspects):
    if not reference_aspects:
        return False
    return min(
        abs(aspect_ratio - reference) / max(reference, 1.0)
        for reference in reference_aspects
    ) <= 0.05


_SINGLE_FRAME_MIN_DOMINANT_AREA_RATIO = 0.25


def _strong_single_frame_scale_supported(shape, profile, border_families):
    """Allow a large, strongly bordered singleton with a different aspect."""
    if not profile.get("nested_border") or not profile.get("thick_inner"):
        return False
    reference_areas = [
        family.get("median_area", 0.0)
        for family in border_families
        if family.get("median_area", 0.0) > 0.0
    ]
    if not reference_areas:
        return False
    dominant_area = max(reference_areas)
    return shape.get("outer_area", 0.0) >= (
        dominant_area * _SINGLE_FRAME_MIN_DOMINANT_AREA_RATIO
    )


def _insert_family_summary(block_name, records, block_entity_count, border_profile):
    widths = [record["width"] for record in records]
    heights = [record["height"] for record in records]
    median_width = median(widths)
    median_height = median(heights)
    width_consistency = max(widths) / min(widths)
    height_consistency = max(heights) / min(heights)
    size_variants = len(
        {
            (round(record["width"], 6), round(record["height"], 6))
            for record in records
        }
    )
    return {
        "block_name": block_name,
        "instance_count": len(records),
        "block_entity_count": block_entity_count,
        "median_width": median_width,
        "median_height": median_height,
        "median_area": median_width * median_height,
        "width_consistency": width_consistency,
        "height_consistency": height_consistency,
        "size_variant_count": size_variants,
        "consistent_size": width_consistency <= 1.02 and height_consistency <= 1.02,
        "nested_border": border_profile["nested_border"],
        "thick_inner": border_profile["thick_inner"],
        "parallel_track": border_profile.get("parallel_track", False),
        "nested_depth": border_profile["nested_depth"],
        "border_evidence": border_profile.get("border_evidence", "none"),
        "border_pair": border_profile["pair"],
    }


def _detect_insert_frames_with_families(doc):
    """通过重复块族和几何尺度，寻找不依赖名称的图框候选。"""
    records_by_block = defaultdict(list)
    for entity in doc.modelspace().query("INSERT"):
        record = _insert_record(entity)
        if record is not None:
            records_by_block[record["block_name"]].append(record)

    families = []
    for block_name, records in records_by_block.items():
        if len(records) < 2:
            continue
        try:
            block_entity_count = len(list(doc.blocks.get(block_name)))
        except Exception:
            block_entity_count = None
        border_profile = _block_border_profile(doc, block_name)
        families.append(
            _insert_family_summary(
                block_name, records, block_entity_count, border_profile
            )
        )

    # 图幅大小不作为正式边框族的硬门槛。同一个块定义可能被以不同
    # 比例插入。先保留原有的嵌套边框证据，再把明确粗细证据作为更高
    # 优先级；只有在当前图纸确实存在强边框族时，才过滤掉嵌套-only小块。
    nested_border_families = [
        family
        for family in families
        if family["nested_border"]
    ]
    strong_border_families = [
        family
        for family in nested_border_families
        if family["thick_inner"] or family.get("parallel_track", False)
    ]
    border_families = strong_border_families or nested_border_families
    # 尺寸一致只能说明它们是重复块，不能说明它们是图框。
    # 没有边框证据的族只保留在调试报告中，避免把轴号、标注等小块
    # 作为正式图框输出。
    selected_families = border_families

    candidates = []
    for selected_family in selected_families:
        selected_records = records_by_block[selected_family["block_name"]]
        for record in selected_records:
            candidate = dict(record)
            has_nested_border = selected_family["nested_border"]
            candidate.update(
                {
                    "nested_border": has_nested_border,
                    "thick_inner": selected_family["thick_inner"],
                    "parallel_track": selected_family.get("parallel_track", False),
                    "nested_depth": selected_family["nested_depth"],
                    "border_evidence": selected_family.get("border_evidence", "none"),
                    "size_variant_count": selected_family["size_variant_count"],
                    "border_pair": selected_family["border_pair"],
                    "status": (
                        "外细内粗边框候选"
                        if selected_family["thick_inner"]
                        or selected_family.get("parallel_track", False)
                        else "完整边框对 + 多层辅助候选"
                        if has_nested_border
                        else "重复块族候选"
                    ),
                    "note": (
                        "同一块定义具备完整外框与内框证据；优先依据外细内粗，图幅尺寸允许存在多个比例"
                        + (
                            "；并存在内框线宽更大的明确证据"
                            if selected_family["thick_inner"]
                            else "；并存在内外边界近邻平行线的几何证据"
                            if selected_family.get("parallel_track", False)
                            else "；未读取到明确线宽差异，多层边界仅作为辅助证据"
                        )
                        if has_nested_border
                        else "该重复块族没有足够边框证据，仅保留在诊断报告中",
                    ),
                }
            )
            candidates.append(candidate)

    # 同一视觉图幅可能因为被单独编辑而生成新的块名，只出现一次。
    # 仅在已经找到正式边框参考族时，按外框比例和内外矩形结构补充单次候选。
    single_frame_candidates = []
    reference_aspects = [
        max(family["median_width"], family["median_height"])
        / min(family["median_width"], family["median_height"])
        for family in border_families
    ]
    selected_handles = {candidate["entity_handle"] for candidate in candidates}
    if reference_aspects:
        for block_name, records in records_by_block.items():
            if len(records) != 1:
                continue
            record = records[0]
            if record["entity_handle"] in selected_handles:
                continue
            shape = _single_frame_shape(doc, block_name)
            profile = _block_border_profile(doc, block_name)
            aspect_matches = shape is not None and _aspect_matches_reference(
                shape["outer_aspect_ratio"], reference_aspects
            )
            strong_single_frame = shape is not None and _strong_single_frame_scale_supported(
                shape, profile, border_families
            )
            if shape is None or not (aspect_matches or strong_single_frame):
                continue

            candidate = dict(record)
            candidate.update(
                {
                    "nested_border": profile.get("nested_border", False),
                    "thick_inner": profile.get("thick_inner", False),
                    "parallel_track": profile.get("parallel_track", False),
                    "nested_depth": profile.get("nested_depth", 0),
                    "border_evidence": (
                        "single_frame_strong_border"
                        if strong_single_frame and not aspect_matches
                        else "single_frame_shape_reference"
                    ),
                    "size_variant_count": 1,
                    "border_pair": profile.get("pair"),
                    "status": "单次同形图幅候选",
                    "note": (
                        "该 INSERT 只出现一次，但外框比例、内外矩形结构与已确认图幅相近；"
                        "未使用块名、文字或图层语义判断"
                    ),
                }
            )
            candidates.append(candidate)
            single_frame_candidates.append(candidate)

    candidates = _deduplicate_candidates(candidates)
    selected_candidate_handles = {
        candidate["entity_handle"] for candidate in candidates
    }
    single_frame_candidates = [
        candidate
        for candidate in single_frame_candidates
        if candidate["entity_handle"] in selected_candidate_handles
    ]
    candidates.sort(key=lambda item: (-item["bbox"][3], item["bbox"][0]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["index"] = index
    families.sort(
        key=lambda family: (
            not family["nested_border"],
            -family["median_area"],
        )
    )
    return candidates, families, single_frame_candidates


def _select_model_space_frame_candidates(rectangle_candidates):
    """Select only explicit-width model-space rectangles for the first formal route."""
    candidates = [
        dict(candidate)
        for candidate in rectangle_candidates
        if candidate.get("source") == "closed_polyline"
        and candidate.get("line_width", 0.0) > 0
    ]
    candidates = _deduplicate_candidates(candidates)
    for candidate in candidates:
        candidate.update(
            {
                "border_evidence": "model_space_explicit_width",
                "status": "模型空间粗边框候选",
                "note": (
                    "闭合正交矩形具有明确几何宽度；本阶段不依赖图层、文件名或固定尺寸"
                ),
            }
        )
    return candidates


_MODEL_BORDER_MAX_MARGIN_RATIO = 0.05
_MODEL_BORDER_MAX_ASPECT_DELTA = 0.10
_MODEL_CONTENT_MIN_COVERAGE = 0.05
_MODEL_CONTENT_RELATIVE_FLOOR = 0.25


def _strictly_contains_rectangle(outer, inner) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _model_border_pair(outer, inner):
    """Find a nearby thin outer border for an explicit-width inner border."""
    if not _strictly_contains_rectangle(outer["bbox"], inner["bbox"]):
        return None

    inner_width = inner["width"]
    inner_height = inner["height"]
    if inner_width <= 0 or inner_height <= 0:
        return None

    margins = [
        inner["bbox"][0] - outer["bbox"][0],
        inner["bbox"][1] - outer["bbox"][1],
        outer["bbox"][2] - inner["bbox"][2],
        outer["bbox"][3] - inner["bbox"][3],
    ]
    if min(margins) <= 0:
        return None

    if max(margins) / min(inner_width, inner_height) > _MODEL_BORDER_MAX_MARGIN_RATIO:
        return None

    aspect_delta = abs(outer["aspect_ratio"] - inner["aspect_ratio"])
    if aspect_delta > _MODEL_BORDER_MAX_ASPECT_DELTA:
        return None

    if not _line_is_thicker(inner, outer):
        return None

    return {
        "outer_handle": outer["entity_handle"],
        "inner_handle": inner["entity_handle"],
        "outer_bbox": outer["bbox"],
        "inner_bbox": inner["bbox"],
        "outer_line_width": outer["line_width"],
        "inner_line_width": inner["line_width"],
        "outer_lineweight": outer["lineweight"],
        "inner_lineweight": inner["lineweight"],
        "outer_source": outer.get("source", "unknown"),
        "inner_source": inner.get("source", "unknown"),
        "margins": margins,
        "outer_area": outer["width"] * outer["height"],
        "line_width_evidence": True,
        "max_margin_ratio": max(margins) / min(inner_width, inner_height),
        "aspect_delta": aspect_delta,
    }


def _drop_nested_formal_candidates(candidates):
    """Keep top-level formal groups; do not count internal rectangles as frames."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -(candidate["width"] * candidate["height"]),
            candidate["bbox"][0],
        ),
    )
    selected = []
    for candidate in ordered:
        if any(
            _strictly_contains_rectangle(existing["bbox"], candidate["bbox"])
            for existing in selected
        ):
            candidate["formal_group_role"] = "internal_rectangle"
            candidate["formal_rejection"] = (
                "formal frame group is contained by another formal frame group"
            )
            continue
        selected.append(candidate)
    return selected


def _finalize_formal_candidates(candidates):
    """Apply the common non-nesting rule to every formal-candidate route."""
    candidates = _deduplicate_candidates(candidates)
    candidates = _drop_nested_formal_candidates(candidates)
    candidates.sort(key=lambda item: (-item["bbox"][3], item["bbox"][0]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["index"] = index
    return candidates


def _paper_viewport_model_box(viewport):
    """Return the model-space rectangle visible through a paper viewport.

    A paper-space sheet and the model-space drawing use different coordinate
    systems.  The viewport's ``view_center_point`` and ``view_height`` are
    model coordinates; its paper width/height gives the aspect ratio of the
    visible model rectangle.  This deliberately handles the common, untwisted
    viewport first.  A twisted viewport is still reported, but is not used as
    a frame candidate until its axis-aligned range can be trusted.
    """
    try:
        center = _dxf_get(viewport.dxf, "view_center_point")
        view_height = float(_dxf_get(viewport.dxf, "view_height") or 0.0)
        paper_width = float(_dxf_get(viewport.dxf, "width") or 0.0)
        paper_height = float(_dxf_get(viewport.dxf, "height") or 0.0)
        twist = float(_dxf_get(viewport.dxf, "view_twist_angle") or 0.0)
        cx, cy = float(center.x), float(center.y)
    except (AttributeError, TypeError, ValueError):
        return None

    if (
        not all(math.isfinite(value) for value in (cx, cy, view_height))
        or view_height <= 0.0
        or paper_width <= 0.0
        or paper_height <= 0.0
    ):
        return None

    # The current splitter uses axis-aligned bboxes.  A non-zero twist needs
    # a rotated polygon rather than this box, so keep it as diagnostic data
    # and let the caller reject it conservatively for now.
    if abs(twist) > 1e-7:
        return None

    model_width = view_height * paper_width / paper_height
    if not math.isfinite(model_width) or model_width <= 0.0:
        return None
    return [
        cx - model_width / 2.0,
        cy - view_height / 2.0,
        cx + model_width / 2.0,
        cy + view_height / 2.0,
    ]


def _model_support_records_for_view(doc):
    """Build cheap model-space records for validating paper viewports."""
    records = []
    for entity in doc.modelspace():
        entity_box = _support_entity_bbox(entity)
        if entity_box is None:
            continue
        if not all(math.isfinite(float(value)) for value in entity_box):
            continue
        records.append(
            {
                "handle": _dxf_get(entity.dxf, "handle") or "",
                "type": entity.dxftype(),
                "bbox": entity_box,
            }
        )
    return records


def _paper_layout_frame_candidates(doc):
    """Find formal sheets represented by paper-space templates.

    This is intentionally separate from model-space frame detection.  A
    layout template alone is not enough: it must also contain a viewport that
    points at real model-space content.  That prevents unused/old layouts
    containing the same title block from becoming duplicate formal frames.
    """
    model_records = _model_support_records_for_view(doc)
    if not model_records:
        return []

    candidates = []
    for layout in doc.layouts:
        if layout.is_modelspace:
            continue

        template_records = []
        for entity in layout.query("INSERT"):
            record = _insert_record(entity)
            if record is None:
                continue
            block_name = record.get("block_name") or ""
            if not block_name:
                continue
            profile = _block_border_profile(doc, block_name)
            if not (
                profile.get("thick_inner")
                or profile.get("parallel_track")
            ):
                continue
            record["profile"] = profile
            template_records.append(record)

        if not template_records:
            continue

        viewports = []
        for entity in layout.query("VIEWPORT"):
            model_box = _paper_viewport_model_box(entity)
            if model_box is None:
                continue
            content_count = sum(
                1 for record in model_records
                if _boxes_intersect(record["bbox"], model_box)
            )
            if content_count <= 0:
                continue
            viewports.append(
                {
                    "handle": _dxf_get(entity.dxf, "handle") or "",
                    "model_bbox": model_box,
                    "content_entity_count": content_count,
                    "paper_center": [
                        float(entity.dxf.center.x),
                        float(entity.dxf.center.y),
                    ],
                }
            )

        if not viewports:
            # A template with no usable model viewport is retained in the
            # shared-layout report, but is not promoted to a formal frame.
            continue

        for template in template_records:
            paper_box = template["bbox"]
            related_viewports = [
                viewport
                for viewport in viewports
                if _box_contains(
                    paper_box,
                    [
                        viewport["paper_center"][0],
                        viewport["paper_center"][1],
                        viewport["paper_center"][0],
                        viewport["paper_center"][1],
                    ],
                )
            ]
            if not related_viewports:
                # If the viewport center is unavailable or outside the
                # template, do not guess which sheet owns the model view.
                continue

            model_boxes = [
                viewport["model_bbox"] for viewport in related_viewports
            ]
            model_bbox = [
                min(box[0] for box in model_boxes),
                min(box[1] for box in model_boxes),
                max(box[2] for box in model_boxes),
                max(box[3] for box in model_boxes),
            ]
            profile = template["profile"]
            candidate = {
                "entity_handle": template["entity_handle"],
                "entity_type": "PAPERSPACE_FRAME",
                "entity_handles": [template["entity_handle"]],
                "block_name": template["block_name"],
                "layer": template["layer"],
                "bbox": paper_box,
                "width": template["width"],
                "height": template["height"],
                "aspect_ratio": template["aspect_ratio"],
                "a_series_like": abs(
                    template["aspect_ratio"] - math.sqrt(2)
                ) <= 0.08,
                "line_width": 0.0,
                "lineweight": None,
                "border_width_evidence": ["block_inner_geometric_width"],
                "border_priority": "explicit_width_evidence",
                "border_evidence": "paperspace_outer_thin_inner_thick",
                "formal_group_role": "paperspace_layout_frame",
                "status": "纸空间正式图框（模板粗细边框 + 有效模型视口）",
                "note": (
                    "图框来自纸空间布局；保留该布局中的标题栏、图例、文字、"
                    "视口和其他说明性实体，并按视口可见范围分配模型空间实体"
                ),
                "thick_inner": bool(profile.get("thick_inner")),
                "parallel_track": bool(profile.get("parallel_track")),
                "nested_border": bool(profile.get("nested_border")),
                "nested_depth": profile.get("nested_depth", 0),
                "border_pair": profile.get("pair"),
                "space": "Paper",
                "layout_name": layout.name,
                "paper_layout_name": layout.name,
                "viewport_handles": [
                    viewport["handle"] for viewport in related_viewports
                ],
                "model_view_boxes": model_boxes,
                "model_bbox": model_bbox,
                "viewport_model_entity_count": sum(
                    viewport["content_entity_count"]
                    for viewport in related_viewports
                ),
                "paper_entity_count": len(layout),
            }
            candidates.append(candidate)

    return _deduplicate_candidates(candidates)


def _boxes_intersect(first, second):
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _box_contains(outer, inner):
    tolerance = max(
        outer[2] - outer[0], outer[3] - outer[1], 1.0
    ) * 1e-9
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _support_entity_bbox(entity):
    """Return a cheap model-space bbox used only as frame-content evidence."""
    entity_type = entity.dxftype()
    points = []
    try:
        if entity_type == "LINE":
            points = [entity.dxf.start, entity.dxf.end]
        elif entity_type == "LWPOLYLINE":
            points = [point[:2] for point in entity.get_points("xyb")]
        elif entity_type == "POLYLINE":
            points = [vertex.dxf.location for vertex in entity.vertices]
        elif entity_type in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "INSERT"}:
            point = _dxf_get(entity.dxf, "insert")
            if point is not None:
                points = [point]
        elif entity_type in {"CIRCLE", "ARC"}:
            center = entity.dxf.center
            radius = abs(float(entity.dxf.radius))
            points = [
                (center.x - radius, center.y - radius),
                (center.x + radius, center.y + radius),
            ]
        elif entity_type == "POINT":
            points = [entity.dxf.location]
    except (AttributeError, TypeError, ValueError):
        return None

    coordinates = []
    for point in points:
        try:
            coordinates.append((float(point[0]), float(point[1])))
        except (IndexError, TypeError, ValueError):
            try:
                coordinates.append((float(point.x), float(point.y)))
            except (AttributeError, TypeError, ValueError):
                continue
    if not coordinates:
        return None
    return [
        min(point[0] for point in coordinates),
        min(point[1] for point in coordinates),
        max(point[0] for point in coordinates),
        max(point[1] for point in coordinates),
    ]


def _annotate_model_content_support(doc, candidates):
    """Add content-density evidence without using layer names or fixed sizes."""
    records = []
    for entity in doc.modelspace():
        entity_box = _support_entity_bbox(entity)
        if entity_box is None:
            continue
        records.append(
            {
                "handle": _dxf_get(entity.dxf, "handle") or "",
                "type": entity.dxftype(),
                "bbox": entity_box,
            }
        )

    for candidate in candidates:
        candidate_box = candidate["bbox"]
        boundary_handles = set(candidate.get("entity_handles") or [])
        contained = [
            record
            for record in records
            if record["handle"] not in boundary_handles
            and record["bbox"][0] >= candidate_box[0]
            and record["bbox"][1] >= candidate_box[1]
            and record["bbox"][2] <= candidate_box[2]
            and record["bbox"][3] <= candidate_box[3]
        ]
        candidate["content_entity_count"] = len(contained)
        candidate["content_entity_types"] = dict(
            defaultdict(int, {
                entity_type: sum(
                    1 for record in contained if record["type"] == entity_type
                )
                for entity_type in {record["type"] for record in contained}
            })
        )
        if not contained:
            candidate["content_coverage_ratio"] = 0.0
            continue

        content_box = [
            min(record["bbox"][0] for record in contained),
            min(record["bbox"][1] for record in contained),
            max(record["bbox"][2] for record in contained),
            max(record["bbox"][3] for record in contained),
        ]
        candidate_area = max(
            0.0,
            candidate_box[2] - candidate_box[0],
        ) * max(0.0, candidate_box[3] - candidate_box[1])
        content_area = max(0.0, content_box[2] - content_box[0]) * max(
            0.0, content_box[3] - content_box[1]
        )
        candidate["content_coverage_ratio"] = (
            content_area / candidate_area if candidate_area else 0.0
        )
    return candidates


def _content_supported_geometry_fallback(rectangle_candidates):
    """Find complete geometry-only containers when explicit frames are empty."""
    geometry_candidates = [
        candidate
        for candidate in rectangle_candidates
        if candidate.get("source") == "closed_polyline"
        and not candidate.get("border_width_evidence")
        and candidate.get("content_entity_count", 0) > 0
    ]
    if not geometry_candidates:
        return []

    maximum_content = max(
        candidate.get("content_entity_count", 0)
        for candidate in geometry_candidates
    )
    if maximum_content <= 0:
        return []

    fallback = [
        candidate
        for candidate in geometry_candidates
        if candidate.get("content_entity_count", 0)
        >= maximum_content * _MODEL_CONTENT_RELATIVE_FLOOR
        and candidate.get("content_coverage_ratio", 0.0)
        >= _MODEL_CONTENT_MIN_COVERAGE
    ]
    result = []
    for candidate in fallback:
        selected = dict(candidate)
        selected.update(
            {
                "border_evidence": "model_space_content_supported_geometry",
                "formal_group_role": "geometry_only_content_fallback",
                "status": "模型空间内容支撑的几何图幅候选",
                "note": (
                    "明确粗细边框候选内部内容不足，改用能覆盖大量实体的完整几何矩形；"
                    "不依赖图层、文件名或固定尺寸"
                ),
            }
        )
        result.append(selected)
    return result


def _select_model_space_formal_candidates(rectangle_candidates, doc=None):
    """Select model-space frames by border groups and non-nesting semantics."""
    explicit = [
        candidate
        for candidate in rectangle_candidates
        if candidate.get("source") == "closed_polyline"
        and candidate.get("line_width", 0.0) > 0
    ]
    formal_candidates = []
    paired_inner_handles = set()
    paired_outer_handles = set()

    for inner in explicit:
        pairs = [
            pair
            for outer in rectangle_candidates
            if (pair := _model_border_pair(outer, inner)) is not None
        ]
        if not pairs:
            continue

        pair = min(
            pairs,
            key=lambda item: (item["max_margin_ratio"], item["aspect_delta"]),
        )
        outer_width = pair["outer_bbox"][2] - pair["outer_bbox"][0]
        outer_height = pair["outer_bbox"][3] - pair["outer_bbox"][1]
        candidate = dict(inner)
        candidate.update(
            {
                "bbox": pair["outer_bbox"],
                "width": outer_width,
                "height": outer_height,
                "aspect_ratio": max(outer_width, outer_height)
                / min(outer_width, outer_height),
                "border_pair": pair,
                "border_evidence": "model_space_outer_thin_inner_thick",
                "formal_group_role": "inner_border_of_formal_group",
                "status": "模型空间外细内粗正式图幅候选",
                "note": (
                    "外细框与内粗框先合并为一个正式图幅组，"
                    "不将内外边框分别计数"
                ),
                "entity_handles": list(
                    dict.fromkeys(
                        list(inner.get("entity_handles") or [inner["entity_handle"]])
                        + [pair["outer_handle"]]
                    )
                ),
            }
        )
        formal_candidates.append(candidate)
        paired_inner_handles.add(inner["entity_handle"])
        paired_outer_handles.add(pair["outer_handle"])

    # A standalone thick frame remains a fallback.  A rectangle that is
    # contained by another complete rectangle is treated as internal content.
    for standalone in explicit:
        handle = standalone["entity_handle"]
        if handle in paired_inner_handles or handle in paired_outer_handles:
            continue
        containers = [
            candidate
            for candidate in rectangle_candidates
            if candidate["entity_handle"] != handle
            and _strictly_contains_rectangle(
                candidate["bbox"], standalone["bbox"]
            )
        ]
        if containers:
            standalone["formal_group_role"] = "internal_rectangle"
            standalone["formal_rejection"] = (
                "被更大闭合矩形包含，未形成独立正式图幅组"
            )
            continue

        candidate = dict(standalone)
        candidate.update(
            {
                "border_evidence": "model_space_standalone_explicit_width",
                "formal_group_role": "standalone_thick_frame_fallback",
                "status": "模型空间单独粗边框候选",
                "note": (
                    "未发现近距离外框配对，作为单独粗边框回退候选；"
                    "不依赖图层、文件名或固定尺寸"
                ),
            }
        )
        formal_candidates.append(candidate)

    if doc is not None:
        geometry_candidates = [
            candidate
            for candidate in rectangle_candidates
            if candidate.get("source") == "closed_polyline"
            and not candidate.get("border_width_evidence")
        ]
        _annotate_model_content_support(
            doc, formal_candidates + geometry_candidates
        )
        has_substantial_explicit_content = any(
            candidate.get("content_entity_count", 0)
            > len(candidate.get("entity_handles") or [])
            and candidate.get("content_coverage_ratio", 0.0)
            >= _MODEL_CONTENT_MIN_COVERAGE
            for candidate in formal_candidates
        )
        if not has_substantial_explicit_content:
            fallback_candidates = _content_supported_geometry_fallback(
                rectangle_candidates
            )
            if fallback_candidates:
                for candidate in formal_candidates:
                    candidate["formal_group_role"] = "border_only_local_rectangle"
                    candidate["formal_rejection"] = (
                        "明确粗细边框候选只包含边框本身，"
                        "被内容支撑更充分的几何图幅候选替代"
                    )
                formal_candidates = fallback_candidates

    return _deduplicate_candidates(_drop_nested_formal_candidates(formal_candidates))


def detect_frames(doc):
    """Detect formal candidates from paper space before model space."""
    paper_candidates = _paper_layout_frame_candidates(doc)
    if paper_candidates:
        return _finalize_formal_candidates(paper_candidates)

    insert_candidates, _, _ = _detect_insert_frames_with_families(doc)
    rectangle_candidates = detect_model_frames(doc)
    # A drawing normally uses one dominant frame representation.  Keep the
    # established INSERT route primary; use direct model-space frames only
    # when no INSERT frame was found, so local thick rectangles in a block-
    # based drawing do not create extra formal frames.
    model_candidates = (
        []
        if insert_candidates
        else _select_model_space_formal_candidates(rectangle_candidates, doc)
    )
    return _finalize_formal_candidates(insert_candidates + model_candidates)


def detect_insert_frames(doc):
    """识别重复且尺寸一致的 INSERT 主尺度候选。"""
    candidates, _, _ = _detect_insert_frames_with_families(doc)
    return candidates


def detect_simple_frames(source: Path) -> dict:
    """读取 DXF 并生成简单图框候选报告数据。"""
    doc, load_info = _load_document(source)
    paper_space_candidates = _paper_layout_frame_candidates(doc)
    insert_candidates, insert_families, single_frame_candidates = (
        ([], [], [])
        if paper_space_candidates
        else _detect_insert_frames_with_families(doc)
    )
    rectangle_candidates = detect_model_frames(doc)
    model_frame_candidates = (
        []
        if insert_candidates or paper_space_candidates
        else _select_model_space_formal_candidates(rectangle_candidates, doc)
    )
    frame_candidates = _finalize_formal_candidates(
        paper_space_candidates + insert_candidates + model_frame_candidates
    )
    selected_frame_handles = {
        candidate["entity_handle"] for candidate in frame_candidates
    }
    single_frame_candidates = [
        candidate
        for candidate in single_frame_candidates
        if candidate["entity_handle"] in selected_frame_handles
    ]
    inserts = list(doc.modelspace().query("INSERT"))
    return {
        "source": str(source),
        "space": "Paper" if paper_space_candidates else "Model",
        "detector": "外细内粗优先 + 模型空间明确粗边框 + 多层嵌套辅助",
        "debug_detector": "闭合四边正交多段线或四条独立正交 LINE（候选诊断）",
        "load": load_info,
        "candidate_count": len(frame_candidates),
        "candidates": frame_candidates,
        "frames": frame_candidates,
        "rectangle_candidate_count": len(rectangle_candidates),
        "rectangle_candidates": rectangle_candidates,
        "model_frame_candidates": model_frame_candidates,
        "paper_space_candidates": paper_space_candidates,
        "model_insert_count": len(inserts),
        "insert_families": insert_families,
        "single_frame_candidates": single_frame_candidates,
    }


def report_markdown(report: dict) -> str:
    """生成面向使用者的正式图框报告，不展示内部矩形明细。"""
    frames = report.get("frames") or report.get("candidates", [])
    lines = [
        "# 图框识别报告",
        "",
        f"- 源文件：`{report['source']}`",
        f"- 检查空间：`{report['space']}`",
        f"- 当前识别器：{report['detector']}",
        f"- 正式图框候选数量：{report['candidate_count']}",
        f"- 单次同形图幅候选数量：{len(report.get('single_frame_candidates', []))}",
        f"- 模型空间 INSERT 数量：{report.get('model_insert_count', '未记录')}",
        f"- 读取方式：{report['load']['mode']}，结构修复次数：{report['load']['repairs']}",
        "",
        "> 本报告只展示正式图框候选。当前阶段不执行实体归属和 DXF 分割，任何实体都不会被删除。",
        "",
        "## 正式图框候选",
        "",
            "| 编号 | 实体句柄 | 块名 | 图层 | 左下角 | 右上角 | 宽×高 | 长宽比 | 主要证据 | 线宽/近邻线 | 状态 |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for candidate in frames:
        xmin, ymin, xmax, ymax = candidate["bbox"]
        lines.append(
            "| {index} | `{handle}` | `{block_name}` | `{layer}` | "
            "({xmin:.3f}, {ymin:.3f}) | ({xmax:.3f}, {ymax:.3f}) | "
            "{width:.3f} × {height:.3f} | {ratio:.4f} | {evidence} | "
            "{thick_inner} | {status} |".format(
                index=candidate["index"],
                handle=candidate["entity_handle"],
                block_name=candidate.get("block_name", ""),
                layer=candidate["layer"],
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                width=candidate["width"],
                height=candidate["height"],
                ratio=candidate["aspect_ratio"],
                evidence=(
                    "外细内粗"
                    if candidate.get("thick_inner")
                    or candidate.get("parallel_track")
                    else "模型空间明确粗边框"
                    if candidate.get("border_evidence")
                    == "model_space_explicit_width"
                    else "单次同形参考"
                    if candidate.get("border_evidence") == "single_frame_shape_reference"
                    else "多层辅助"
                    if candidate.get("nested_border")
                    else "无"
                ),
                thick_inner=(
                    "是"
                    if candidate.get("thick_inner")
                    or candidate.get("parallel_track")
                    else "明确几何宽度"
                    if candidate.get("border_evidence")
                    == "model_space_explicit_width"
                    else "否"
                ),
                status=candidate["status"],
            )
        )

    if not frames:
        lines.extend(["", "当前没有找到重复且尺寸一致的 INSERT 主尺度候选。"])

    lines.extend(
        [
            "",
            "## 当前判定依据",
            "",
            "- 正式候选优先来自同一块定义中的外细内粗完整边框对；没有明确线宽时，多层嵌套只作为辅助证据；",
            "- 正式边框族不要求所有 INSERT 尺寸一致，因为同一块定义可能按不同比例插入；尺寸一致性仅用于调试和回退判断；",
            "- 单次同形图幅只参考外框比例和内外矩形结构，不使用块名、文字或图层语义；",
            "- 块名和图层名只作为报告信息，不参与正式候选判定；",
            "- INSERT 的范围按块内容和插入变换计算；",
            "- 边框证据优先来自外细内粗的完整矩形对；DXF 未记录明确线宽时，才使用多层矩形作为辅助证据；",
            "- 内部矩形和 INSERT 块族明细见同目录下的 `识别调试报告.md`；",
            "- 未被识别的实体不会被删除。",
        ]
    )
    return "\n".join(lines) + "\n"


def debug_report_markdown(report: dict) -> str:
    """生成供开发排查误识别的详细报告。"""
    rectangle_candidates = report.get("rectangle_candidates", [])
    families = report.get("insert_families", [])
    lines = [
        "# 图框识别调试报告",
        "",
        f"- 源文件：`{report['source']}`",
        f"- 检查空间：`{report['space']}`",
        f"- 正式识别器：{report['detector']}",
        f"- 内部矩形诊断器：{report.get('debug_detector', '未记录')}",
        f"- 正式图框候选数量：{report['candidate_count']}",
        f"- 单次同形图幅候选数量：{len(report.get('single_frame_candidates', []))}",
        f"- 模型空间 INSERT 数量：{report.get('model_insert_count', '未记录')}",
        f"- 内部矩形记录数量：{len(rectangle_candidates)}",
        f"- 读取方式：{report['load']['mode']}，结构修复次数：{report['load']['repairs']}",
        "",
        "> 本报告只用于定位识别问题。内部矩形不是正式图框，也不会因为出现在本报告中而被删除。",
        "",
        "## INSERT 块族分析",
        "",
    ]
    families = report.get("insert_families", [])
    if families:
        lines.extend(
            [
                "| 块名 | 实例数 | 尺寸变体 | 块内实体数 | 中位宽 | 中位高 | 中位面积 | 尺寸一致 | 嵌套辅助 | 线宽证据 | 近邻线证据 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for family in families:
            lines.append(
                "| `{block_name}` | {instance_count} | {size_variants} | {entity_count} | "
                "{width:.3f} | {height:.3f} | {area:.3f} | {consistent} | "
                "{nested_border} | {thick_inner} | {parallel_track} |".format(
                    block_name=family["block_name"],
                    instance_count=family["instance_count"],
                    size_variants=family.get("size_variant_count", "未知"),
                    entity_count=family["block_entity_count"]
                    if family["block_entity_count"] is not None
                    else "未知",
                    width=family["median_width"],
                    height=family["median_height"],
                    area=family["median_area"],
                    consistent="是" if family["consistent_size"] else "否",
                    nested_border="是" if family.get("nested_border") else "否",
                    thick_inner="是" if family.get("thick_inner") else "否",
                    parallel_track="是" if family.get("parallel_track") else "否",
                )
            )
    else:
        lines.append("没有发现重复出现的 INSERT 块族。")

    lines.extend(
        [
            "",
        "## 模型空间矩形候选（暂不计入正式图框）",
            "",
            f"共发现 {len(rectangle_candidates)} 个模型空间矩形候选，来源包括闭合多段线和四条独立 LINE。它们可能是表格、局部详图或正式图框，本阶段只记录，不删除。",
            "",
            "| 编号 | 实体类型 | 来源 | 实体句柄 | 图层 | 左下角 | 右上角 | 宽×高 | 几何宽度 | 边框证据 |",
            "| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for candidate in rectangle_candidates:
        xmin, ymin, xmax, ymax = candidate["bbox"]
        lines.append(
            "| {index} | `{entity_type}` | `{source}` | `{handle}` | `{layer}` | "
            "({xmin:.3f}, {ymin:.3f}) | ({xmax:.3f}, {ymax:.3f}) | "
            "{width:.3f} × {height:.3f} | {line_width:.3f} | {evidence} |".format(
                index=candidate["index"],
                entity_type=candidate.get("entity_type", ""),
                source=candidate.get("source", ""),
                handle=candidate["entity_handle"],
                layer=candidate["layer"],
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                width=candidate["width"],
                height=candidate["height"],
                line_width=candidate.get("line_width", 0.0),
                evidence=(
                    "、".join(candidate.get("border_width_evidence", []))
                    or "无明确宽度"
                ),
            )
        )

    lines.extend(
        [
            "",
            "## 调试规则边界",
            "",
            "- 只检查模型空间；",
            "- 正式候选优先来自同一块定义中的外细内粗完整边框对；没有明确线宽时，多层嵌套只作为辅助证据；",
            "- 正式边框族不要求所有 INSERT 尺寸一致；尺寸变体数量只用于解释样本，不作为正式边框的否决条件；",
            "- 块名和图层名只作为报告信息，不参与正式候选判定；",
            "- INSERT 的范围按块内容和插入变换计算；",
            "- 边框证据优先来自外细内粗的完整矩形对；DXF 未记录明确线宽时，才使用多层矩形作为辅助证据；",
            "- 模型空间中的闭合四边正交多段线和四条独立 LINE 只作为候选记录，不计入正式图框；",
            "- 未被识别的实体不会被删除。",
        ]
    )
    return "\n".join(lines) + "\n"
