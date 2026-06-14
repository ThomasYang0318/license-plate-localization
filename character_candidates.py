from pathlib import Path

import cv2


def find_character_boxes(
    binary_image,
    min_width: int,
    max_width_ratio: float,
    min_height: int,
    max_height_ratio: float,
    min_area: int,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    y_tolerance_ratio: float,
    border_margin_ratio: float,
):
    image_height, image_width = binary_image.shape[:2]
    image_area = image_width * image_height
    margin_x = int(image_width * border_margin_ratio)
    margin_y = int(image_height * border_margin_ratio)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_image > 0).astype("uint8"),
        connectivity=8,
    )

    boxes = []
    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        aspect_ratio = width / height

        if x <= margin_x or y <= margin_y:
            continue
        if x + width >= image_width - margin_x or y + height >= image_height - margin_y:
            continue
        if width < min_width or width > image_width * max_width_ratio:
            continue
        if height < min_height or height > image_height * max_height_ratio:
            continue
        if area < min_area or area > image_area * max_area_ratio:
            continue
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        boxes.append((x, y, width, height))

    return filter_text_row(boxes, image_width, image_height, y_tolerance_ratio)


def find_best_character_boxes(
    primary_binary_image,
    fallback_binary_image,
    min_candidates: int,
    min_width: int,
    max_width_ratio: float,
    min_height: int,
    max_height_ratio: float,
    min_area: int,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    y_tolerance_ratio: float,
    border_margin_ratio: float,
):
    primary_boxes = find_character_boxes(
        primary_binary_image,
        min_width=min_width,
        max_width_ratio=max_width_ratio,
        min_height=min_height,
        max_height_ratio=max_height_ratio,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        y_tolerance_ratio=y_tolerance_ratio,
        border_margin_ratio=border_margin_ratio,
    )
    if len(primary_boxes) >= min_candidates:
        return primary_boxes, "primary"

    fallback_boxes = find_character_boxes(
        fallback_binary_image,
        min_width=min_width,
        max_width_ratio=max_width_ratio,
        min_height=min_height,
        max_height_ratio=max_height_ratio,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        y_tolerance_ratio=y_tolerance_ratio,
        border_margin_ratio=border_margin_ratio,
    )
    if len(fallback_boxes) > len(primary_boxes):
        return fallback_boxes, "adaptive"

    return primary_boxes, "primary"


def filter_text_row(boxes, image_width: int, image_height: int, y_tolerance_ratio: float):
    if not boxes:
        return []

    y_tolerance = image_height * y_tolerance_ratio
    centers_y = [y + height / 2 for _, y, _, height in boxes]
    best_center_y = centers_y[0]
    best_score = -1

    for center_y in centers_y:
        nearby_boxes = [
            box
            for box in boxes
            if abs((box[1] + box[3] / 2) - center_y) <= y_tolerance
        ]
        nearby_heights = sorted(height for _, _, _, height in nearby_boxes)
        median_height = nearby_heights[len(nearby_heights) // 2]
        score = len(nearby_boxes) * median_height
        if score > best_score:
            best_score = score
            best_center_y = center_y

    row_boxes = [
        box
        for box in boxes
        if abs((box[1] + box[3] / 2) - best_center_y) <= y_tolerance
    ]

    if not row_boxes:
        return []

    median_height = sorted(height for _, _, _, height in row_boxes)[len(row_boxes) // 2]
    height_tolerance = median_height * 0.65

    filtered_boxes = [
        box
        for box in row_boxes
        if abs(box[3] - median_height) <= height_tolerance
    ]

    return select_horizontal_character_group(filtered_boxes, image_width)


def select_horizontal_character_group(boxes, image_width: int):
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    median_height = sorted(height for _, _, _, height in sorted_boxes)[len(sorted_boxes) // 2]
    max_gap = max(int(median_height * 1.2), int(image_width * 0.015))

    groups = []
    current_group = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        previous_box = current_group[-1]
        previous_right = previous_box[0] + previous_box[2]
        gap = box[0] - previous_right
        if gap > max_gap:
            groups.append(current_group)
            current_group = [box]
        else:
            current_group.append(box)

    groups.append(current_group)

    best_group = max(
        groups,
        key=lambda group: len(group) * sorted(height for _, _, _, height in group)[len(group) // 2],
    )
    return sorted(best_group, key=lambda box: box[0])


def draw_boxes(color_image, boxes):
    boxed_image = color_image.copy()
    for x, y, width, height in boxes:
        cv2.rectangle(
            boxed_image,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 255, 0),
            2,
        )
    return boxed_image


def write_result_file(result_path: Path, image_name: str, boxes) -> None:
    lines = [image_name, str(len(boxes))]
    for x, y, width, height in boxes:
        lines.append(f"{x + 1} {y + 1} {width} {height}")
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
