"""DXF 图框的保守分割。

本模块按模型空间实体的变换后包围盒进行保守归属：完整落在图框内的实体进入
对应文件；对于锚点在框内且主体大部分在框内的文字，整体保留到对应文件；
其余实体进入共享文件，避免第一版分割造成信息丢失。
"""

from __future__ import annotations

import copy
from collections import Counter
import math
from pathlib import Path
import shutil

import ezdxf
from ezdxf import bbox as ezdxf_bbox
from ezdxf.entities.dxfentity import DXFTagStorage
from ezdxf.xclip import XClip
from ezdxf.xref import ConflictPolicy, Loader

from audit import _dxf_get, _load_document
from frame_detection import _support_entity_bbox, detect_frames


def _entity_bbox(entity):
    try:
        extents = ezdxf_bbox.extents([entity], fast=False)
    except Exception:
        return None
    if extents.extmin is None or extents.extmax is None:
        return None
    values = [
        float(extents.extmin.x),
        float(extents.extmin.y),
        float(extents.extmax.x),
        float(extents.extmax.y),
    ]
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _percentile(values, fraction):
    """Return a linearly interpolated percentile for a non-empty sequence."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_display_bbox(entities, reference_bbox=None):
    """Build a view range without letting extreme coordinates hide the drawing."""
    if reference_bbox is not None:
        values = [float(value) for value in reference_bbox]
        if (
            len(values) == 4
            and all(math.isfinite(value) for value in values)
            and values[2] > values[0]
            and values[3] > values[1]
        ):
            return values

    boxes = []
    for entity in entities:
        entity_box = _support_entity_bbox(entity)
        if entity_box is None:
            continue
        if not all(math.isfinite(float(value)) for value in entity_box):
            continue
        boxes.append([float(value) for value in entity_box])
    if not boxes:
        return None

    x_values = [value for box in boxes for value in (box[0], box[2])]
    y_values = [value for box in boxes for value in (box[1], box[3])]
    # The range is only for the initial view.  Trim the outer 1% so malformed
    # or remote helper entities do not make a valid drawing appear blank.
    trim_fraction = 0.01 if len(boxes) >= 20 else 0.0
    xmin = _percentile(x_values, trim_fraction)
    xmax = _percentile(x_values, 1.0 - trim_fraction)
    ymin = _percentile(y_values, trim_fraction)
    ymax = _percentile(y_values, 1.0 - trim_fraction)
    if xmax <= xmin or ymax <= ymin:
        xmin, xmax = min(x_values), max(x_values)
        ymin, ymax = min(y_values), max(y_values)
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def _set_display_view(target_doc, display_bbox):
    """Initialize DXF view metadata so large-coordinate drawings open visibly."""
    if display_bbox is None:
        return None
    xmin, ymin, xmax, ymax = display_bbox
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        return None

    padding = max(width, height) * 0.03
    xmin -= padding
    ymin -= padding
    xmax += padding
    ymax += padding
    center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
    view_width = xmax - xmin
    view_height = ymax - ymin

    target_doc.header["$EXTMIN"] = (xmin, ymin, 0.0)
    target_doc.header["$EXTMAX"] = (xmax, ymax, 0.0)
    target_doc.header["$LIMMIN"] = (xmin, ymin)
    target_doc.header["$LIMMAX"] = (xmax, ymax)

    active_viewports = target_doc.viewports.get("*ACTIVE")
    if active_viewports:
        viewport = active_viewports[0]
        viewport.dxf.center = center
        viewport.dxf.height = view_height
        try:
            viewport.dxf.width = view_width
        except (AttributeError, TypeError, ValueError):
            pass

    return [xmin, ymin, xmax, ymax]


def _contains(outer, inner) -> bool:
    tolerance = max(outer[2] - outer[0], outer[3] - outer[1], 1.0) * 1e-9
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _intersects(first, second) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


_TEXT_ENTITY_TYPES = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}
_TEXT_INSIDE_RATIO = 0.80

# These are relative rules.  They do not depend on a drawing's absolute
# coordinates or on a particular paper size.
_MINOR_OVERFLOW_INSIDE_RATIO = 0.80
_MINOR_OVERFLOW_MAX_OUTSIDE_RATIO = 0.10
_GLOBAL_ENTITY_SIZE_RATIO = 3.0
_GEOMETRY_RATIO_EPSILON = 1e-12

_BOUNDARY_CATEGORY_LABELS = {
    "minor_overflow": "轻微越界候选",
    "cross_frame": "明显跨框实体",
    "global_entity": "大范围/全局实体候选",
    "boundary_adjacent": "图框边界邻接实体",
    "outside": "图框外实体",
}


def _entity_anchor(entity):
    """Return the insertion point used as the text ownership anchor."""
    if entity.dxftype() not in _TEXT_ENTITY_TYPES:
        return None
    try:
        point = entity.dxf.insert
    except (AttributeError, TypeError):
        return None
    return [float(point.x), float(point.y)]


def _insert_anchor_frame(entity, frame_boxes):
    """Find the sole frame containing an INSERT insertion point."""
    if entity.dxftype() != "INSERT":
        return None
    try:
        point = entity.dxf.insert
        anchor_box = [float(point.x), float(point.y), float(point.x), float(point.y)]
    except (AttributeError, TypeError, ValueError):
        return None

    matches = [
        index
        for index, frame_box in frame_boxes.items()
        if _contains(frame_box, anchor_box)
    ]
    return matches[0] if len(matches) == 1 else None


def _box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first, second) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _boundary_relation(entity_box, frame_index, frame_box) -> dict:
    """Describe the relation between one entity box and one frame box."""
    entity_width = max(0.0, entity_box[2] - entity_box[0])
    entity_height = max(0.0, entity_box[3] - entity_box[1])
    frame_width = max(0.0, frame_box[2] - frame_box[0])
    frame_height = max(0.0, frame_box[3] - frame_box[1])
    overlap_width = max(
        0.0, min(entity_box[2], frame_box[2]) - max(entity_box[0], frame_box[0])
    )
    overlap_height = max(
        0.0, min(entity_box[3], frame_box[3]) - max(entity_box[1], frame_box[1])
    )
    overlap_area = overlap_width * overlap_height
    entity_area = entity_width * entity_height
    frame_area = frame_width * frame_height

    if entity_area > 0:
        bbox_inside_ratio = overlap_area / entity_area
    elif entity_width >= entity_height and entity_width > 0:
        bbox_inside_ratio = overlap_width / entity_width
    elif entity_height > 0:
        bbox_inside_ratio = overlap_height / entity_height
    else:
        bbox_inside_ratio = 1.0 if _contains(frame_box, entity_box) else 0.0

    outside_left = max(0.0, frame_box[0] - entity_box[0])
    outside_bottom = max(0.0, frame_box[1] - entity_box[1])
    outside_right = max(0.0, entity_box[2] - frame_box[2])
    outside_top = max(0.0, entity_box[3] - frame_box[3])
    outside_ratios = {
        "left": outside_left / frame_width if frame_width else 0.0,
        "bottom": outside_bottom / frame_height if frame_height else 0.0,
        "right": outside_right / frame_width if frame_width else 0.0,
        "top": outside_top / frame_height if frame_height else 0.0,
    }

    return {
        "frame_index": frame_index,
        "bbox_overlap_ratio": bbox_inside_ratio,
        "frame_coverage_ratio": overlap_area / frame_area if frame_area else 0.0,
        "outside_ratio_by_side": outside_ratios,
        "max_outside_ratio": max(outside_ratios.values(), default=0.0),
        "entity_width_to_frame": entity_width / frame_width if frame_width else None,
        "entity_height_to_frame": entity_height / frame_height if frame_height else None,
    }


def _analyze_boundary(entity_box, frame_boxes) -> dict:
    """Classify a non-contained entity without changing its assignment."""
    relations = [
        _boundary_relation(entity_box, index, frame_box)
        for index, frame_box in frame_boxes.items()
        if _intersects(frame_box, entity_box)
    ]

    if not relations:
        category = "outside"
        reason = "未与已识别图框的包围盒相交"
        dominant = None
    else:
        dominant = max(relations, key=lambda item: item["bbox_overlap_ratio"])
        max_size_ratio = max(
            max(item["entity_width_to_frame"] or 0.0 for item in relations),
            max(item["entity_height_to_frame"] or 0.0 for item in relations),
        )
        only_boundary_touch = all(
            item["bbox_overlap_ratio"] <= _GEOMETRY_RATIO_EPSILON
            for item in relations
        )
        is_minor_overflow = (
            len(relations) == 1
            and dominant["bbox_overlap_ratio"] >= _MINOR_OVERFLOW_INSIDE_RATIO
            and dominant["max_outside_ratio"] <= _MINOR_OVERFLOW_MAX_OUTSIDE_RATIO
        )

        if only_boundary_touch:
            category = "boundary_adjacent"
            reason = "仅与图框边界相切或相邻，未形成有效面积重叠"
        elif max_size_ratio >= _GLOBAL_ENTITY_SIZE_RATIO:
            category = "global_entity"
            reason = "实体尺度相对于相交图框明显偏大，疑似跨图框或全局辅助实体"
        elif is_minor_overflow:
            category = "minor_overflow"
            reason = "只涉及一个图框，包围盒主体落入框内且越界比例较小"
        else:
            category = "cross_frame"
            reason = "实体相对图框存在明显越界，暂不判断为单一图框内容"

    result = {
        "category": category,
        "label": _BOUNDARY_CATEGORY_LABELS[category],
        "reason": reason,
        "intersected_frame_count": len(relations),
        "intersected_frames": [item["frame_index"] for item in relations],
        "relations": relations,
    }
    if dominant is not None:
        result["dominant_frame"] = dominant["frame_index"]
        result["dominant_bbox_overlap_ratio"] = dominant["bbox_overlap_ratio"]
        result["dominant_max_outside_ratio"] = dominant["max_outside_ratio"]
    return result


def _text_overflow_matches(entity, entity_box, frame_boxes):
    """Find frames that conservatively own slightly overflowing text."""
    if entity.dxftype() not in _TEXT_ENTITY_TYPES:
        return []

    anchor = _entity_anchor(entity)
    if anchor is None:
        return []

    anchor_box = [anchor[0], anchor[1], anchor[0], anchor[1]]
    entity_area = _box_area(entity_box)
    if entity_area <= 0:
        return []

    matches = []
    for index, frame_box in frame_boxes.items():
        if not _contains(frame_box, anchor_box):
            continue
        inside_ratio = _intersection_area(entity_box, frame_box) / entity_area
        if inside_ratio >= _TEXT_INSIDE_RATIO:
            matches.append({
                "index": index,
                "inside_ratio": inside_ratio,
                "anchor": anchor,
            })
    return matches


def classify_model_entities(doc, frames):
    """把模型空间实体分成各图框内容和共享内容。"""
    frame_regions = {
        frame["index"]: list(frame.get("model_view_boxes") or [frame["bbox"]])
        for frame in frames
    }
    frame_boxes = {
        index: [
            min(region[0] for region in regions),
            min(region[1] for region in regions),
            max(region[2] for region in regions),
            max(region[3] for region in regions),
        ]
        for index, regions in frame_regions.items()
    }
    paper_frame_indexes = {
        frame["index"]
        for frame in frames
        if str(frame.get("space", "")).lower() == "paper"
    }
    assignments = {index: [] for index in frame_boxes}
    shared_entities = []
    status_counts = Counter()
    boundary_category_counts = Counter()
    review_records = []

    for entity in doc.modelspace():
        boundary_analysis = None
        entity_box = _entity_bbox(entity)
        if entity_box is None:
            status = "无法计算边界"
            shared_entities.append(entity)
            status_counts[status] += 1
            review_records.append(
                {
                    "handle": entity.dxf.handle,
                    "type": entity.dxftype(),
                    "layer": _dxf_get(entity.dxf, "layer") or "",
                    "status": status,
                }
            )
            continue

        contained = [
            index
            for index, regions in frame_regions.items()
            if any(_contains(region, entity_box) for region in regions)
        ]
        if contained:
            status = "框内"
            if len(contained) == 1:
                assignments[contained[0]].append(entity)
            if len(contained) > 1:
                status = "同时落入多个图框"
                shared_entities.append(entity)
        else:
            visible_paper_frames = [
                index
                for index in paper_frame_indexes
                if any(
                    _intersects(region, entity_box)
                    for region in frame_regions[index]
                )
            ]
            if visible_paper_frames:
                if len(visible_paper_frames) == 1:
                    target_index = visible_paper_frames[0]
                    assignments[target_index].append(entity)
                    status = "PAPER_VIEWPORT_VISIBLE_KEEP_FULL_ENTITY"
                else:
                    shared_entities.append(entity)
                    status = "MULTIPLE_PAPER_VIEWPORTS_SHARED_ONCE"
                status_counts[status] += 1
                review_records.append(
                    {
                        "handle": entity.dxf.handle,
                        "type": entity.dxftype(),
                        "layer": _dxf_get(entity.dxf, "layer") or "",
                        "bbox": entity_box,
                        "status": status,
                        "frame_indexes": visible_paper_frames,
                        "reason": (
                            "Entity intersects a paper-space viewport model range; "
                            "the complete entity is kept without geometric clipping."
                        ),
                    }
                )
                continue

            insert_anchor_frame = _insert_anchor_frame(entity, frame_boxes)
            if insert_anchor_frame is not None:
                assignments[insert_anchor_frame].append(entity)
                status = "INSERT 定位点在单一图框内，整体保留"
                status_counts[status] += 1
                review_records.append(
                    {
                        "handle": entity.dxf.handle,
                        "type": entity.dxftype(),
                        "layer": _dxf_get(entity.dxf, "layer") or "",
                        "bbox": entity_box,
                        "status": status,
                        "frame_index": insert_anchor_frame,
                        "reason": "块包围盒跨界，但插入点只位于一个图框内",
                    }
                )
                continue

            boundary_analysis = _analyze_boundary(entity_box, frame_boxes)
            boundary_category_counts[boundary_analysis["category"]] += 1
            relaxed_matches = _text_overflow_matches(
                entity, entity_box, frame_boxes
            )
            minor_overflow_frame = None
            if boundary_analysis["category"] == "minor_overflow":
                minor_overflow_frame = boundary_analysis.get("dominant_frame")

            if len(relaxed_matches) == 1 or minor_overflow_frame is not None:
                if len(relaxed_matches) == 1:
                    match = relaxed_matches[0]
                    target_index = match["index"]
                    status = "文字主体在框内，轻微越界（整体保留）"
                else:
                    target_index = minor_overflow_frame
                    match = None
                    status = "实体主体在框内，轻微越界（整体保留）"

                assignments[target_index].append(entity)
                status_counts[status] += 1
                record = {
                    "handle": entity.dxf.handle,
                    "type": entity.dxftype(),
                    "layer": _dxf_get(entity.dxf, "layer") or "",
                    "bbox": entity_box,
                    "status": status,
                    "boundary_analysis": boundary_analysis,
                }
                if match is not None:
                    record["anchor"] = match["anchor"]
                    record["inside_ratio"] = match["inside_ratio"]
                review_records.append(record)
                continue

            status = boundary_analysis["label"]
            shared_entities.append(entity)

        status_counts[status] += 1
        if status != "框内":
            record = {
                "handle": entity.dxf.handle,
                "type": entity.dxftype(),
                "layer": _dxf_get(entity.dxf, "layer") or "",
                "bbox": entity_box,
                "status": status,
            }
            if boundary_analysis is not None:
                record["boundary_analysis"] = boundary_analysis
            review_records.append(record)

    return {
        "assignments": assignments,
        "shared_entities": shared_entities,
        "status_counts": dict(status_counts),
        "boundary_category_counts": dict(boundary_category_counts),
        "review_records": review_records,
        "model_entity_count": len(doc.modelspace()),
    }


def _retain_frame_boundary_entities(doc, frames, classification):
    """Always keep the model-space entities used as detected frame boundaries."""
    entities_by_handle = {
        entity.dxf.handle: entity
        for entity in doc.modelspace()
        if _dxf_get(entity.dxf, "handle")
    }
    boundary_handles = set()
    retained_counts = {}

    for frame in frames:
        handles = list(frame.get("entity_handles") or [])
        if not handles:
            handle = frame.get("entity_handle")
            if handle and "+" not in handle:
                handles.append(handle)

        retained = 0
        for handle in handles:
            entity = entities_by_handle.get(handle)
            if entity is None:
                # INSERT frame geometry lives in its block definition; its
                # model-space INSERT itself is still available through handle.
                continue
            boundary_handles.add(handle)
            if entity not in classification["assignments"][frame["index"]]:
                classification["assignments"][frame["index"]].append(entity)
                retained += 1
        retained_counts[frame["index"]] = retained

    if boundary_handles:
        classification["shared_entities"] = [
            entity
            for entity in classification["shared_entities"]
            if _dxf_get(entity.dxf, "handle") not in boundary_handles
        ]

    classification["frame_boundary_entity_counts"] = retained_counts
    return classification


def _copy_header_settings(source_doc, target_doc):
    for name in ("$INSUNITS", "$MEASUREMENT", "$LUNITS", "$AUNITS", "$LWDISPLAY"):
        value = source_doc.header.get(name)
        if value is not None:
            target_doc.header[name] = value


def _copy_class_definitions(source_doc, target_doc):
    """Keep source DXF CLASS records required by custom graphic entities."""
    copied_classes = []
    for source_class in source_doc.classes:
        try:
            copied_classes.append(source_class.copy())
        except Exception:
            continue
    target_doc.classes.register(copied_classes)


def _restore_unmapped_graphic_entities(
    source_entities, target_doc, target_layout=None
):
    """Keep unknown graphic entities that ezdxf's resource loader skips."""
    target_space = target_layout or target_doc.modelspace()
    existing_handles = {
        entity.dxf.handle
        for entity in target_space
        if entity.dxf.handle
    }
    existing_signatures = Counter(
        _raw_entity_signature(entity)
        for entity in target_space
        if isinstance(entity, DXFTagStorage) and entity.is_graphic_entity
    )
    restored = []

    for source_entity in source_entities:
        if not isinstance(source_entity, DXFTagStorage):
            continue
        if not source_entity.is_graphic_entity:
            continue
        source_handle = _dxf_get(source_entity.dxf, "handle")
        if source_handle and source_handle in existing_handles:
            continue
        signature = _raw_entity_signature(source_entity)
        if existing_signatures[signature] > 0:
            existing_signatures[signature] -= 1
            continue

        try:
            restored_entity = DXFTagStorage.load(
                copy.deepcopy(source_entity.xtags), doc=target_doc
            )
            # The source handle belongs to another document.  Let the target
            # document bind a new handle when the entity is added.
            restored_entity.dxf.handle = None
            target_space.add_entity(restored_entity)
        except Exception:
            continue
        restored.append(restored_entity)
        if restored_entity.dxf.handle:
            existing_handles.add(restored_entity.dxf.handle)
        existing_signatures[signature] += 1

    return restored


def _raw_entity_signature(entity):
    """Build a comparison key that ignores document-local handles."""
    values = []
    for tag in entity.xtags:
        # Group code 5 is the entity handle and 330 is normally the owner
        # handle.  Both are expected to change when an entity is moved to the
        # output document.
        if tag.code in (5, 330):
            continue
        value = tag.value
        if isinstance(value, (list, tuple)):
            value = tuple(value)
        values.append((tag.code, repr(value)))
    return tuple(values)


def _block_name(entity):
    if entity.dxftype() != "INSERT":
        return ""
    return _dxf_get(entity.dxf, "name") or ""


# A few CAD applications store several visually unrelated note panels in one
# reusable block definition.  Treating every INSERT as indivisible is still
# the safe default; this narrow profile is only enabled for a block that is
# demonstrably a large, text-heavy collection of spatially separated groups.
_COMPOUND_BLOCK_MIN_ENTITIES = 100
_COMPOUND_BLOCK_MIN_TEXT_ENTITIES = 80
_COMPOUND_BLOCK_MIN_TEXT_RATIO = 0.25
_COMPOUND_BLOCK_MIN_CLUSTERS = 4
_COMPOUND_BLOCK_MAX_CLUSTERS = 30
_COMPOUND_BLOCK_MAX_LARGEST_CLUSTER_RATIO = 0.35
_COMPOUND_BLOCK_GAP_FRACTION = 0.005
_COMPOUND_BLOCK_GAP_LIMIT = 2000.0


def _source_block_by_name(source_doc, block_name):
    """Return an exact block-name match before using ezdxf's name lookup."""
    for block in source_doc.blocks:
        if block.name == block_name:
            return block
    try:
        return source_doc.blocks.get(block_name)
    except Exception:
        return None


def _finite_entity_bbox(entity):
    """Return a finite entity bbox, or None for unsupported/empty geometry."""
    entity_box = _entity_bbox(entity)
    if entity_box is None:
        return None
    if not all(math.isfinite(float(value)) for value in entity_box):
        return None
    return entity_box


def _compound_block_groups(source_doc, block_name):
    """Find safe spatial groups inside a demonstrably compound block.

    This is intentionally conservative.  A block is eligible only when it
    contains many entities, a substantial text population, and several
    well-separated spatial groups.  Ordinary structural/detail blocks usually
    form one connected group and therefore stay on the normal INSERT path.
    """
    block = _source_block_by_name(source_doc, block_name)
    if block is None:
        return None

    children = list(block)
    if len(children) < _COMPOUND_BLOCK_MIN_ENTITIES:
        return None

    text_count = sum(
        1 for child in children if child.dxftype() in _TEXT_ENTITY_TYPES
    )
    if (
        text_count < _COMPOUND_BLOCK_MIN_TEXT_ENTITIES
        or text_count / max(1, len(children)) < _COMPOUND_BLOCK_MIN_TEXT_RATIO
    ):
        return None

    valid_items = []
    invalid_children = []
    for child in children:
        child_box = _finite_entity_bbox(child)
        if child_box is None:
            invalid_children.append(child)
        else:
            valid_items.append((child, child_box))

    if len(valid_items) < _COMPOUND_BLOCK_MIN_ENTITIES:
        return None

    full_box = [
        min(item[1][0] for item in valid_items),
        min(item[1][1] for item in valid_items),
        max(item[1][2] for item in valid_items),
        max(item[1][3] for item in valid_items),
    ]
    full_width = max(0.0, full_box[2] - full_box[0])
    full_height = max(0.0, full_box[3] - full_box[1])
    gap_limit = min(
        max(full_width, full_height) * _COMPOUND_BLOCK_GAP_FRACTION,
        _COMPOUND_BLOCK_GAP_LIMIT,
    )

    parent = list(range(len(valid_items)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first = find(first)
        second = find(second)
        if first != second:
            parent[second] = first

    def bbox_gap(first, second):
        return max(
            0.0,
            second[0] - first[2],
            first[0] - second[2],
            second[1] - first[3],
            first[1] - second[3],
        )

    for first in range(len(valid_items)):
        for second in range(first + 1, len(valid_items)):
            if bbox_gap(valid_items[first][1], valid_items[second][1]) <= gap_limit:
                union(first, second)

    grouped_positions = {}
    for index in range(len(valid_items)):
        grouped_positions.setdefault(find(index), []).append(index)

    groups = []
    for member_indexes in grouped_positions.values():
        group_children = [valid_items[index][0] for index in member_indexes]
        group_box = [
            min(valid_items[index][1][0] for index in member_indexes),
            min(valid_items[index][1][1] for index in member_indexes),
            max(valid_items[index][1][2] for index in member_indexes),
            max(valid_items[index][1][3] for index in member_indexes),
        ]
        groups.append({"children": group_children, "bbox": group_box})

    groups.sort(
        key=lambda group: (
            -group["bbox"][1],
            group["bbox"][0],
        )
    )

    full_area = max(0.0, full_width * full_height)
    largest_group_area = max(
        (
            max(0.0, group["bbox"][2] - group["bbox"][0])
            * max(0.0, group["bbox"][3] - group["bbox"][1])
            for group in groups
        ),
        default=0.0,
    )
    if not (
        _COMPOUND_BLOCK_MIN_CLUSTERS
        <= len(groups)
        <= _COMPOUND_BLOCK_MAX_CLUSTERS
        and full_area > 0.0
        and largest_group_area / full_area
        <= _COMPOUND_BLOCK_MAX_LARGEST_CLUSTER_RATIO
    ):
        return None

    # Empty/unsupported child geometry must not disappear.  Keep it in a
    # residual part which will conservatively go to shared content.
    if invalid_children:
        groups.append({"children": invalid_children, "bbox": None})

    return groups


def _clone_block_children(target_block, children):
    """Copy child entities into a synthetic block without source handles."""
    for child in children:
        # DXFEntity.copy() avoids copying the whole source document graph;
        # deepcopy() is prohibitively slow for entities owned by a document.
        clone = child.copy()
        for attribute in ("handle", "owner"):
            try:
                setattr(clone.dxf, attribute, None)
            except (AttributeError, TypeError, ValueError):
                pass
        target_block.add_entity(clone)


def _copy_insert_attributes(source_insert, target_insert):
    """Copy visual INSERT attributes while keeping target ownership fields."""
    for name, value in source_insert.dxfattribs().items():
        if name in {"handle", "owner", "name", "insert"}:
            continue
        try:
            target_insert.dxf.set(name, copy.deepcopy(value))
        except (AttributeError, TypeError, ValueError):
            continue


_UNSUPPORTED_COMPOUND_CLIP = object()


def _compound_insert_clip_box(source_insert):
    """Return an INSERT's clipping box in block coordinates.

    The boundary stored by a SPATIAL_FILTER is not necessarily already in
    block coordinates.  XClip applies the inverse INSERT matrix for us.  A
    non-rectangular or inverted clipping path is deliberately left to the
    original INSERT path instead of guessing and changing its visible result.
    """
    try:
        xclip = XClip(source_insert)
        spatial_filter = xclip.get_spatial_filter()
        if spatial_filter is None or not xclip.is_clipping_enabled:
            return None
        clipping_path = xclip.get_block_clipping_path()
        if clipping_path.is_inverted_clip:
            return _UNSUPPORTED_COMPOUND_CLIP
        vertices = list(clipping_path.vertices)
        if len(vertices) < 3:
            return _UNSUPPORTED_COMPOUND_CLIP
        values = [
            (float(vertex.x), float(vertex.y)) for vertex in vertices
        ]
        if not all(math.isfinite(value) for pair in values for value in pair):
            return _UNSUPPORTED_COMPOUND_CLIP
        return [
            min(pair[0] for pair in values),
            min(pair[1] for pair in values),
            max(pair[0] for pair in values),
            max(pair[1] for pair in values),
        ]
    except Exception:
        # A malformed filter must not make the whole source file fail.  Keep
        # that INSERT indivisible and let the established path handle it.
        return _UNSUPPORTED_COMPOUND_CLIP


def _bbox_intersects(first, second):
    return not (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )


def _compound_parts_for_clip(source_doc, groups, clip_box, counter):
    """Create filter-aware synthetic blocks without copying the filter."""
    generated_parts = []
    for group in groups:
        if clip_box is None or group["bbox"] is None:
            selected_children = list(group["children"])
        elif not _bbox_intersects(group["bbox"], clip_box):
            continue
        else:
            selected_children = []
            for child in group["children"]:
                child_box = _finite_entity_bbox(child)
                if child_box is None or _bbox_intersects(child_box, clip_box):
                    selected_children.append(child)
            if not selected_children:
                continue

        generated_block_name = f"__DXF_SPLIT_COMPOUND_{counter:04d}"
        counter += 1
        generated_block = source_doc.blocks.new(
            name=generated_block_name,
            base_point=(0.0, 0.0, 0.0),
        )
        _clone_block_children(generated_block, selected_children)
        generated_parts.append(
            {
                "block_name": generated_block_name,
                "bbox": group["bbox"],
            }
        )
    return generated_parts, counter


def _expand_compound_block_inserts(source_doc):
    """Replace only confirmed compound INSERTs with spatial child INSERTs.

    The source file is never saved after this in-memory transformation.  Each
    generated INSERT references a synthetic block containing one spatial group
    from the original definition, so the existing entity classification and
    export path can remain unchanged for every other entity.
    """
    modelspace = source_doc.modelspace()
    original_count = len(modelspace)
    compound_cache = {}
    generated_parts_cache = {}
    replacements = []
    generated_block_counter = 0

    source_inserts = [
        entity for entity in list(modelspace) if entity.dxftype() == "INSERT"
    ]
    for source_insert in source_inserts:
        block_name = _block_name(source_insert)
        if not block_name:
            continue
        if block_name not in compound_cache:
            groups = _compound_block_groups(source_doc, block_name)
            compound_cache[block_name] = groups
        groups = compound_cache[block_name]
        if not groups:
            continue

        clip_box = _compound_insert_clip_box(source_insert)
        if clip_box is _UNSUPPORTED_COMPOUND_CLIP:
            continue

        clip_key = None if clip_box is None else tuple(
            round(value, 6) for value in clip_box
        )
        parts_key = (block_name, clip_key)
        if parts_key not in generated_parts_cache:
            generated_parts, generated_block_counter = _compound_parts_for_clip(
                source_doc,
                groups,
                clip_box,
                generated_block_counter,
            )
            generated_parts_cache[parts_key] = generated_parts
        generated_parts = generated_parts_cache[parts_key]
        if not generated_parts:
            continue

        created_handles = []
        for part in generated_parts:
            # Create a clean INSERT deliberately.  Copying the source INSERT
            # would also copy SPATIAL_FILTER and make every child part render
            # its own white clipping rectangle.
            generated_insert = modelspace.add_blockref(
                part["block_name"],
                insert=source_insert.dxf.insert,
            )
            _copy_insert_attributes(source_insert, generated_insert)
            created_handles.append(generated_insert.dxf.handle)

        source_handle = _dxf_get(source_insert.dxf, "handle") or ""
        modelspace.delete_entity(source_insert)
        replacements.append(
            {
                "source_handle": source_handle,
                "source_block_name": block_name,
                "part_count": len(generated_parts),
                "generated_handles": created_handles,
            }
        )

    if not replacements:
        return {
            "original_model_entity_count": original_count,
            "expanded_model_entity_count": original_count,
            "replaced_insert_count": 0,
            "generated_part_count": 0,
            "replacements": [],
        }

    return {
        "original_model_entity_count": original_count,
        "expanded_model_entity_count": len(modelspace),
        "replaced_insert_count": len(replacements),
        "generated_part_count": sum(
            replacement["part_count"] for replacement in replacements
        ),
        "replacements": replacements,
    }


def _insert_export_issue(source_doc, entity):
    """Return a reason when an INSERT cannot be safely copied to a new DXF."""
    if entity.dxftype() != "INSERT":
        return None

    name = _dxf_get(entity.dxf, "name")
    if not isinstance(name, str) or not name.strip():
        return "INSERT 缺少有效块名"

    try:
        block = source_doc.blocks.get(name)
    except Exception as error:
        return f"INSERT 块定义读取失败：{error}"
    if block is None:
        return f"INSERT 引用的块定义不存在：{name}"
    return None


def _quarantined_entity_record(entity, reason):
    """Build a small JSON-safe record for an entity excluded from Loader."""
    record = {
        "handle": _dxf_get(entity.dxf, "handle") or "",
        "type": entity.dxftype(),
        "layer": _dxf_get(entity.dxf, "layer") or "",
        "reason": reason,
    }
    name = _dxf_get(entity.dxf, "name")
    record["block_name"] = name if isinstance(name, str) else repr(name)
    try:
        point = entity.dxf.insert
        record["insert"] = [float(point.x), float(point.y), float(point.z)]
    except (AttributeError, TypeError, ValueError):
        pass
    return record


def _partition_export_entities(source_doc, entities):
    """Keep normal entities on the Loader path and isolate malformed INSERTs."""
    exportable = []
    quarantined = []
    for entity in entities:
        issue = _insert_export_issue(source_doc, entity)
        if issue is None:
            exportable.append(entity)
        else:
            quarantined.append(_quarantined_entity_record(entity, issue))
    return exportable, quarantined


def _build_used_block_pairs(
    source_doc,
    target_doc,
    source_entities,
    target_entities=None,
):
    """Map source block names to their names in the target document.

    Named blocks normally keep their names.  Anonymous blocks are renamed by
    ezdxf, so those mappings are learned from the corresponding model-space
    INSERT order and then propagated through nested INSERTs.
    """
    source_top_inserts = [
        entity for entity in source_entities if entity.dxftype() == "INSERT"
    ]
    target_entities = (
        list(target_doc.modelspace())
        if target_entities is None
        else list(target_entities)
    )
    target_top_inserts = [
        entity for entity in target_entities if entity.dxftype() == "INSERT"
    ]
    pending = []
    mapping = {}

    for source_insert, target_insert in zip(source_top_inserts, target_top_inserts):
        source_name = _block_name(source_insert)
        target_name = _block_name(target_insert)
        if source_name and target_name:
            pending.append((source_name, target_name))

    # Named blocks are a reliable direct mapping and also cover cases where
    # model-space INSERT counts differ because of a malformed source entity.
    for source_name in {
        source_name
        for source_name, _ in pending
        if not source_name.startswith("*")
    }:
        try:
            if target_doc.blocks.get(source_name) is not None:
                mapping[source_name] = source_name
        except Exception:
            continue

    while pending:
        source_name, target_name = pending.pop(0)
        known_target = mapping.get(source_name)
        if known_target is not None and known_target != target_name:
            continue
        mapping[source_name] = target_name

        try:
            source_block = source_doc.blocks.get(source_name)
            target_block = target_doc.blocks.get(target_name)
        except Exception:
            continue
        if source_block is None or target_block is None:
            continue

        source_nested = [
            entity
            for entity in source_block
            if entity.dxftype() == "INSERT"
            and _insert_export_issue(source_doc, entity) is None
        ]
        target_nested = [
            entity
            for entity in target_block
            if entity.dxftype() == "INSERT" and _block_name(entity)
        ]
        for source_insert, target_insert in zip(source_nested, target_nested):
            child_source_name = _block_name(source_insert)
            child_target_name = _block_name(target_insert)
            if child_source_name and child_target_name:
                pending.append((child_source_name, child_target_name))

    return mapping


def _restore_unmapped_block_entities(
    source_doc,
    target_doc,
    source_entities,
    target_entities=None,
):
    """Restore unsupported graphic entities inside used block definitions.

    ``Loader`` correctly copies ordinary block members, but its copy machine
    skips unknown/custom DXF entities.  Such entities are not present in the
    model-space entity list, so the model-space fallback cannot restore them.
    This function fills only the missing raw graphic entities and leaves the
    normal Loader path untouched.
    """
    block_pairs = _build_used_block_pairs(
        source_doc,
        target_doc,
        source_entities,
        target_entities,
    )
    restored = []
    restored_by_type = Counter()

    for source_name, target_name in block_pairs.items():
        try:
            source_block = source_doc.blocks.get(source_name)
            target_block = target_doc.blocks.get(target_name)
        except Exception:
            continue
        if source_block is None or target_block is None:
            continue

        existing_signatures = Counter(
            _raw_entity_signature(entity)
            for entity in target_block
            if isinstance(entity, DXFTagStorage) and entity.is_graphic_entity
        )
        for source_entity in source_block:
            if not isinstance(source_entity, DXFTagStorage):
                continue
            if not source_entity.is_graphic_entity:
                continue
            if _insert_export_issue(source_doc, source_entity) is not None:
                continue
            signature = _raw_entity_signature(source_entity)
            if existing_signatures[signature] > 0:
                existing_signatures[signature] -= 1
                continue

            try:
                restored_entity = DXFTagStorage.load(
                    copy.deepcopy(source_entity.xtags), doc=target_doc
                )
                restored_entity.dxf.handle = None
                target_block.add_entity(restored_entity)
            except Exception:
                continue
            restored.append(restored_entity)
            restored_by_type[restored_entity.dxftype()] += 1

    return {
        "count": len(restored),
        "types": dict(restored_by_type),
        "block_count": len(block_pairs),
    }


def _copy_layout_settings(source_layout, target_layout):
    """Copy paper-space plotting/view settings without source handles."""
    source_record = getattr(source_layout, "dxf_layout", None)
    target_record = getattr(target_layout, "dxf_layout", None)
    if source_record is None or target_record is None:
        return

    handle_fields = {"handle", "owner", "block_record_handle", "viewport_handle"}
    for name, value in source_record.dxfattribs().items():
        if name in handle_fields:
            continue
        try:
            target_record.dxf.set(name, copy.deepcopy(value))
        except (AttributeError, TypeError, ValueError):
            continue


def _ensure_target_paper_layout(target_doc, source_layout):
    """Create or reuse a target paper layout while preserving its name."""
    layout_name = source_layout.name
    try:
        target_layout = target_doc.layouts.get(layout_name)
    except Exception:
        target_layout = None
    if target_layout is None:
        target_layout = target_doc.layouts.new(layout_name)
    return target_layout


def _delete_unused_default_layout(target_doc, copied_layout_names):
    """Remove ezdxf's empty starter layout only when it is not source data."""
    # A model-space-only output still needs one valid paperspace layout for
    # ezdxf to maintain a valid DXF document and report an active layout.
    if not copied_layout_names or "Layout1" in copied_layout_names:
        return
    try:
        layout = target_doc.layouts.get("Layout1")
        if layout is not None and len(layout) == 0:
            target_doc.layouts.delete("Layout1")
    except Exception:
        return


def _export_entities(
    source_doc,
    entities,
    destination: Path,
    display_bbox=None,
    paper_layouts=None,
    active_layout_name=None,
):
    """使用 ezdxf 的资源映射器复制实体及其块、样式和对象依赖。"""
    exportable_entities, quarantined_entities = _partition_export_entities(
        source_doc, entities
    )
    target_doc = ezdxf.new(dxfversion=source_doc.dxfversion)
    _copy_header_settings(source_doc, target_doc)
    _copy_class_definitions(source_doc, target_doc)
    handles = {entity.dxf.handle for entity in exportable_entities}
    loader = Loader(source_doc, target_doc, conflict_policy=ConflictPolicy.KEEP)
    loader.load_modelspace(filter_fn=lambda entity: entity.dxf.handle in handles)
    copied_paper_layouts = []
    for source_layout in list(paper_layouts or []):
        target_layout = _ensure_target_paper_layout(target_doc, source_layout)
        paper_handles = {
            entity.dxf.handle
            for entity in source_layout
            if entity.dxf.handle
        }
        loader.load_paperspace_layout_into(
            source_layout,
            target_layout,
            filter_fn=lambda entity, paper_handles=paper_handles: entity.dxf.handle
            in paper_handles,
        )
        copied_paper_layouts.append((source_layout, target_layout))
    loader.execute()
    for source_layout, target_layout in copied_paper_layouts:
        _copy_layout_settings(source_layout, target_layout)
    restored_entities = _restore_unmapped_graphic_entities(
        exportable_entities, target_doc
    )
    restored_block_entities = _restore_unmapped_block_entities(
        source_doc,
        target_doc,
        exportable_entities,
        list(target_doc.modelspace()),
    )
    restored_paper_entities = []
    restored_paper_block_entities = {
        "count": 0,
        "types": Counter(),
        "block_count": 0,
    }
    for source_layout, target_layout in copied_paper_layouts:
        restored_paper_entities.extend(
            _restore_unmapped_graphic_entities(
                list(source_layout), target_doc, target_layout=target_layout
            )
        )
        paper_block_result = _restore_unmapped_block_entities(
            source_doc,
            target_doc,
            list(source_layout),
            list(target_layout),
        )
        restored_paper_block_entities["count"] += paper_block_result["count"]
        restored_paper_block_entities["types"].update(
            paper_block_result["types"]
        )
        restored_paper_block_entities["block_count"] += paper_block_result[
            "block_count"
        ]
    view_bbox = _safe_display_bbox(entities, reference_bbox=display_bbox)
    view_bbox = _set_display_view(target_doc, view_bbox)
    copied_layout_names = {layout.name for _, layout in copied_paper_layouts}
    _delete_unused_default_layout(target_doc, copied_layout_names)
    if active_layout_name in copied_layout_names:
        target_doc.layouts.set_active_layout(active_layout_name)
    # Model space is not a paperspace layout and cannot be selected through
    # Layouts.set_active_layout().  Leave the default paperspace layout alone
    # for model-space-only outputs.
    destination.parent.mkdir(parents=True, exist_ok=True)
    target_doc.saveas(destination)

    check_doc = ezdxf.readfile(destination)
    check_entities = list(check_doc.modelspace())
    check_paper_layouts = [
        layout for layout in check_doc.layouts if not layout.is_modelspace
    ]
    paper_output_entity_count = sum(len(layout) for layout in check_paper_layouts)
    paper_source_entity_count = sum(
        len(layout) for layout in (paper_layouts or [])
    )
    return {
        "path": str(destination),
        "source_entity_count": len(entities),
        "output_entity_count": len(check_entities),
        "output_entity_types": dict(Counter(entity.dxftype() for entity in check_entities)),
        "source_paper_layout_count": len(list(paper_layouts or [])),
        "source_paper_entity_count": paper_source_entity_count,
        "output_paper_layout_count": len(check_paper_layouts),
        "output_paper_entity_count": paper_output_entity_count,
        "output_layouts": [layout.name for layout in check_doc.layouts],
        "active_layout": check_doc.layouts.active_layout().name,
        "display_bbox": view_bbox,
        "quarantined_invalid_entity_count": len(quarantined_entities),
        "quarantined_invalid_entities": quarantined_entities,
        "restored_unmapped_entity_count": len(restored_entities),
        "restored_unmapped_entity_types": dict(
            Counter(entity.dxftype() for entity in restored_entities)
        ),
        "restored_block_entity_count": restored_block_entities["count"],
        "restored_block_entity_types": restored_block_entities["types"],
        "restored_block_count": restored_block_entities["block_count"],
        "restored_paper_entity_count": len(restored_paper_entities),
        "restored_paper_entity_types": dict(
            Counter(entity.dxftype() for entity in restored_paper_entities)
        ),
        "restored_paper_block_entity_count": restored_paper_block_entities[
            "count"
        ],
        "restored_paper_block_entity_types": dict(
            restored_paper_block_entities["types"]
        ),
        "restored_paper_block_count": restored_paper_block_entities[
            "block_count"
        ],
    }


def _allocation_entity_key(entity):
    handle = _dxf_get(entity.dxf, "handle")
    return f"handle:{handle}" if handle else f"object:{id(entity)}"


def _build_allocation_stats(doc, classification):
    """Audit model-space ownership so duplication and loss are visible."""
    source_keys = {
        _allocation_entity_key(entity) for entity in doc.modelspace()
    }
    destination_counts = Counter()
    frame_counts = {}
    for index, entities in classification["assignments"].items():
        frame_counts[str(index)] = len(entities)
        destination_counts.update(_allocation_entity_key(entity) for entity in entities)
    shared_entities = classification["shared_entities"]
    destination_counts.update(
        _allocation_entity_key(entity) for entity in shared_entities
    )
    duplicated = [
        key for key, count in destination_counts.items() if count > 1
    ]
    assigned_keys = set(destination_counts)
    unassigned = source_keys - assigned_keys
    return {
        "source_model_entity_count": len(doc.modelspace()),
        "source_model_unique_key_count": len(source_keys),
        "frame_model_entity_counts": frame_counts,
        "shared_model_entity_count": len(shared_entities),
        "unique_destination_entity_count": len(assigned_keys),
        "duplicated_entity_count": len(duplicated),
        "unassigned_entity_count": len(unassigned),
        "duplicate_entity_keys_sample": duplicated[:20],
        "unassigned_entity_keys_sample": sorted(unassigned)[:20],
    }


def _split_single_frame_as_source(source, output_dir, doc, frames, load_info):
    """Keep a single formal frame as an intact source DXF.

    A paper-space layout already defines the relationship between the sheet,
    viewports, and model space.  Reassigning every model entity is unnecessary
    when there is only one formal frame and can damage malformed blocks.
    """
    frame = frames[0]
    destination = output_dir / f"图框_{frame['index']:03d}.dxf"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    # A previous entity-level run may have left a shared file behind.  It is
    # not part of the single-frame result and would otherwise look current.
    stale_shared = output_dir / "共享内容.dxf"
    if stale_shared.exists():
        stale_shared.unlink()

    paper_layouts = [
        layout for layout in doc.layouts if not layout.is_modelspace
    ]
    model_count = len(doc.modelspace())
    paper_entity_count = sum(len(layout) for layout in paper_layouts)
    layout_names = [layout.name for layout in doc.layouts]
    frame_layout_name = frame.get("layout_name")
    allocation_stats = {
        "source_model_entity_count": model_count,
        "source_model_unique_key_count": model_count,
        "frame_model_entity_counts": {str(frame["index"]): model_count},
        "shared_model_entity_count": 0,
        "unique_destination_entity_count": model_count,
        "duplicated_entity_count": 0,
        "unassigned_entity_count": 0,
        "duplicate_entity_keys_sample": [],
        "unassigned_entity_keys_sample": [],
        "mode": "whole_source_copy",
    }
    output = {
        "kind": "frame",
        "frame_index": frame["index"],
        "frame_handle": frame.get("entity_handle"),
        "frame_boundary_entity_count": 0,
        "entity_count": model_count,
        "layout_name": frame_layout_name,
        "path": str(destination),
        "source_entity_count": model_count,
        "output_entity_count": model_count,
        "source_paper_layout_count": len(paper_layouts),
        "source_paper_entity_count": paper_entity_count,
        "output_paper_layout_count": len(paper_layouts),
        "output_paper_entity_count": paper_entity_count,
        "output_layouts": layout_names,
        "active_layout": doc.layouts.active_layout().name,
        "display_bbox": None,
        "quarantined_invalid_entity_count": 0,
        "quarantined_invalid_entities": [],
        "restored_unmapped_entity_count": 0,
        "restored_unmapped_entity_types": {},
        "restored_block_entity_count": 0,
        "restored_block_entity_types": {},
        "restored_block_count": 0,
        "restored_paper_entity_count": 0,
        "restored_paper_entity_types": {},
        "restored_paper_block_entity_count": 0,
        "restored_paper_block_entity_types": {},
        "restored_paper_block_count": 0,
        "whole_source_copy": True,
    }
    classification = {
        "model_entity_count": model_count,
        "status_counts": {"整体保留": model_count},
        "boundary_category_counts": {},
        "review_records": [],
        "allocation_stats": allocation_stats,
        "assigned_paper_layout_names": (
            [frame_layout_name] if frame_layout_name else []
        ),
        "shared_paper_layout_names": [],
        "mode": "whole_source_copy",
    }
    return {
        "source": str(source),
        "output_dir": str(output_dir),
        "load": load_info,
        "mode": "whole_source_copy",
        "frame_count": 1,
        "frames": frames,
        "classification": classification,
        "outputs": [output],
    }


def split_dxf(source: Path, output_dir: Path) -> dict:
    """识别图框并为所有图框生成 DXF，同时生成共享内容文件。"""
    doc, load_info = _load_document(source)
    frames = detect_frames(doc)
    if len(frames) == 1:
        return _split_single_frame_as_source(
            source, output_dir, doc, frames, load_info
        )
    original_model_entity_count = len(doc.modelspace())
    compound_expansion = _expand_compound_block_inserts(doc)
    classification = classify_model_entities(doc, frames)
    classification = _retain_frame_boundary_entities(doc, frames, classification)
    paper_layouts_by_name = {
        layout.name: layout
        for layout in doc.layouts
        if not layout.is_modelspace
    }
    assigned_paper_layout_names = {
        frame.get("layout_name")
        for frame in frames
        if frame.get("layout_name") in paper_layouts_by_name
    }
    shared_paper_layouts = [
        layout
        for name, layout in paper_layouts_by_name.items()
        if name not in assigned_paper_layout_names
    ]
    classification["allocation_stats"] = _build_allocation_stats(
        doc, classification
    )
    classification["allocation_stats"][
        "original_source_model_entity_count"
    ] = original_model_entity_count
    classification["allocation_stats"][
        "expanded_source_model_entity_count"
    ] = len(doc.modelspace())
    outputs = []

    for frame in frames:
        frame_layout = paper_layouts_by_name.get(frame.get("layout_name"))
        frame_paper_layouts = [frame_layout] if frame_layout is not None else []
        frame_display_bbox = (
            frame.get("model_bbox")
            if str(frame.get("space", "")).lower() == "paper"
            else frame.get("bbox")
        )
        destination = output_dir / f"图框_{frame['index']:03d}.dxf"
        exported = _export_entities(
            doc,
            classification["assignments"][frame["index"]],
            destination,
            display_bbox=frame_display_bbox,
            paper_layouts=frame_paper_layouts,
            active_layout_name=frame.get("layout_name"),
        )
        outputs.append(
            {
                "kind": "frame",
                "frame_index": frame["index"],
                "frame_handle": frame["entity_handle"],
                "frame_boundary_entity_count": classification[
                    "frame_boundary_entity_counts"
                ].get(frame["index"], 0),
                "entity_count": len(classification["assignments"][frame["index"]]),
                "layout_name": frame.get("layout_name"),
                **exported,
            }
        )

    if classification["shared_entities"] or shared_paper_layouts:
        destination = output_dir / "共享内容.dxf"
        exported = _export_entities(
            doc,
            classification["shared_entities"],
            destination,
            paper_layouts=shared_paper_layouts,
            active_layout_name=(
                shared_paper_layouts[0].name if shared_paper_layouts else "Model"
            ),
        )
        outputs.append(
            {
                "kind": "shared",
                "entity_count": len(classification["shared_entities"]),
                "layout_names": [layout.name for layout in shared_paper_layouts],
                **exported,
            }
        )

    return {
        "source": str(source),
        "output_dir": str(output_dir),
        "load": load_info,
        "frame_count": len(frames),
        "original_model_entity_count": original_model_entity_count,
        "expanded_model_entity_count": len(doc.modelspace()),
        "compound_expansion": compound_expansion,
        "frames": frames,
        "classification": {
            "model_entity_count": classification["model_entity_count"],
            "original_model_entity_count": original_model_entity_count,
            "expanded_model_entity_count": len(doc.modelspace()),
            "compound_expansion": compound_expansion,
            "status_counts": classification["status_counts"],
            "boundary_category_counts": classification["boundary_category_counts"],
            "review_records": classification["review_records"],
            "allocation_stats": classification["allocation_stats"],
            "assigned_paper_layout_names": sorted(assigned_paper_layout_names),
            "shared_paper_layout_names": [
                layout.name for layout in shared_paper_layouts
            ],
        },
        "outputs": outputs,
    }


def split_report_markdown(report: dict) -> str:
    classification = report["classification"]
    original_model_entity_count = report.get(
        "original_model_entity_count",
        classification["model_entity_count"],
    )
    expanded_model_entity_count = report.get(
        "expanded_model_entity_count",
        classification["model_entity_count"],
    )
    lines = [
        "# DXF 分割报告",
        "",
        f"- 源文件：`{report['source']}`",
        f"- 输出目录：`{report['output_dir']}`",
        f"- 图框数量：{report['frame_count']}",
        f"- 原始模型空间实体总数：{original_model_entity_count}",
        f"- 分割归属实体单元数：{expanded_model_entity_count}",
        f"- 读取方式：{report['load']['mode']}，结构修复次数：{report['load']['repairs']}",
        "",
        "> 本次采用保守归属：完整落入图框的实体进入对应文件；跨边界、无法定位和图框外实体进入共享内容。无法安全复制的异常块实体会单独记录，不让它拖垮整个导出。",
        "",
        "## 输出文件",
        "",
            "| 类型 | 图框编号 | 来源实体数 | 输出实体数 | 未映射实体恢复 | 文件 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for output in report["outputs"]:
        frame_index = output.get("frame_index", "-")
        lines.append(
            f"| {output['kind']} | {frame_index} | {output['source_entity_count']} | "
            f"{output['output_entity_count']} | {output.get('restored_unmapped_entity_count', 0)} | "
            f"`{Path(output['path']).name}`（块内恢复 {output.get('restored_block_entity_count', 0)} 个，"
            f"涉及 {output.get('restored_block_count', 0)} 个块） |"
        )

    lines.extend(
        [
            "",
            "## 导出异常实体",
            "",
            "> 下列实体未交给普通块复制流程，原因是块名为空或块定义不存在。它们不会影响其他正常实体的导出，具体原始信息同时写入 `拆分数据.json`。",
            "",
        ]
    )
    allocation = classification.get("allocation_stats", {})
    lines.extend(
        [
            "",
            "## 结构保留与实体归属核查",
            "",
            "- 纸空间布局不会被默认丢弃；已关联图框的布局进入对应图框文件，未关联布局进入共享内容。",
            f"- 原始模型空间实体：{allocation.get('original_source_model_entity_count', allocation.get('source_model_entity_count', 0))}；展开后归属单元：{allocation.get('expanded_source_model_entity_count', allocation.get('source_model_entity_count', 0))}；目标文件唯一归属实体：{allocation.get('unique_destination_entity_count', 0)}。",
            f"- 重复归属实体：{allocation.get('duplicated_entity_count', 0)}；未归属实体：{allocation.get('unassigned_entity_count', 0)}。",
        ]
    )
    compound_expansion = report.get("compound_expansion") or {}
    if compound_expansion.get("replaced_insert_count", 0):
        lines.extend(
            [
                "",
                "## 复合块展开",
                "",
                f"- 替换复合 INSERT：{compound_expansion['replaced_insert_count']} 个；",
                f"- 展开后的内部归属单元：{compound_expansion['generated_part_count']} 个；",
                "- 仅对满足多区域、文本密集和空间分离条件的复合块启用，普通 INSERT 不展开。",
            ]
        )
    for output in report["outputs"]:
        lines.append(
            f"- `{Path(output['path']).name}`：模型空间 {output.get('output_entity_count', 0)} 个，"
            f"纸空间布局 {output.get('output_paper_layout_count', 0)} 个，"
            f"纸空间实体 {output.get('output_paper_entity_count', 0)} 个，"
            f"当前布局 `{output.get('active_layout', '')}`。"
        )

    quarantined_total = 0
    for output in report["outputs"]:
        quarantined = output.get("quarantined_invalid_entities", [])
        quarantined_total += len(quarantined)
        for item in quarantined:
            location = item.get("insert")
            location_text = f"，插入点={location}" if location else ""
            lines.append(
                f"- {output['kind']}：句柄 `{item.get('handle', '')}`，类型 `{item.get('type', '')}`，"
                f"图层 `{item.get('layer', '')}`，块名 `{item.get('block_name', '')}`"
                f"{location_text}；原因：{item.get('reason', '未说明')}"
            )
    if quarantined_total == 0:
        lines.append("- 未发现需要隔离的异常块实体。")

    lines.extend(["", "## 归属统计", ""])
    for status, count in classification["status_counts"].items():
        lines.append(f"- {status}：{count}")

    lines.extend(
        [
            "",
            "## 跨框实体分级",
            "",
            "> 本节只用于识别和复核，不改变当前保守归属，也不执行几何裁剪。",
            "",
            "| 分级 | 数量 | 当前处理 |",
            "| --- | ---: | --- |",
        ]
    )
    boundary_actions = {
        "minor_overflow": "暂不自动归属，保留在共享内容，后续可针对简单实体处理",
        "cross_frame": "保留在共享内容",
        "global_entity": "保留在共享内容",
        "boundary_adjacent": "保留在共享内容，通常是图框邻接线",
        "outside": "保留在共享内容",
    }
    for category, count in classification.get("boundary_category_counts", {}).items():
        label = _BOUNDARY_CATEGORY_LABELS.get(category, category)
        lines.append(
            f"| {label} | {count} | {boundary_actions.get(category, '保留在共享内容')} |"
        )

    lines.extend(
        [
            "",
            "## 当前边界",
            "",
            "- 当前按实体变换后包围盒判断，不做图形裁剪；",
            f"- 对 TEXT、MTEXT、ATTRIB 和 ATTDEF：锚点在图框内且主体占比达到 {_TEXT_INSIDE_RATIO:.0%} 时，轻微越界文字整体保留，不做裁剪；",
            "- 轻微越界、明显跨框、大范围/全局和图框外实体均先保留在 `共享内容.dxf`，并在 JSON 中记录相交图框、重叠比例和越界比例；",
            "- 通过资源映射复制块、样式和对象依赖；",
            "- 对资源映射器无法处理但仍带有图形数据的未知实体，按原始 DXF 标签重新绑定并保留；",
            "- 原始 DXF 不会被覆盖；",
            "- 输出 DXF 已重新读取并记录实体数量。",
        ]
    )
    return "\n".join(lines) + "\n"
