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

    return select_best_character_group(boxes, image_width, image_height, y_tolerance_ratio)


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


def find_plate_guided_character_boxes(
    gray_image,
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
    candidates = find_plate_roi_candidates(gray_image)
    best_boxes = []
    best_score = -1.0

    for roi, roi_score in candidates:
        primary_roi = mask_to_roi(primary_binary_image, roi)
        fallback_roi = mask_to_roi(fallback_binary_image, roi)
        boxes, _ = find_best_character_boxes(
            primary_binary_image=primary_roi,
            fallback_binary_image=fallback_roi,
            min_candidates=min_candidates,
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
        if len(boxes) < min_candidates:
            continue

        heights = sorted(height for _, _, _, height in boxes)
        median_height = heights[len(heights) // 2]
        score = roi_score * len(boxes) * median_height
        if score > best_score:
            best_score = score
            best_boxes = boxes

    if best_boxes:
        return best_boxes, "plate_roi"

    return [], ""


def find_plate_roi_candidates(gray_image):
    image_height, image_width = gray_image.shape[:2]
    _, bright_image = cv2.threshold(gray_image, 140, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    bright_image = cv2.morphologyEx(bright_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(bright_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        aspect_ratio = width / height

        if x <= image_width * 0.02 or x + width >= image_width * 0.98:
            continue
        if y <= image_height * 0.05 or y + height >= image_height * 0.95:
            continue
        if width < 60 or height < 15:
            continue
        if not 2.0 <= aspect_ratio <= 6.5:
            continue
        if not image_width * image_height * 0.001 <= area <= image_width * image_height * 0.08:
            continue

        roi = gray_image[y:y + height, x:x + width]
        dark_ratio = (roi < 100).mean()
        bright_ratio = (roi > 150).mean()
        if bright_ratio < 0.08 or dark_ratio < 0.08:
            continue

        center_x = x + width / 2
        center_weight = max(0.35, 1.0 - abs(center_x - image_width / 2) / (image_width / 2))
        score = area * center_weight * (0.5 + dark_ratio) * (0.5 + bright_ratio)
        candidates.append(((x, y, width, height), score))

    return sorted(candidates, key=lambda item: item[1], reverse=True)


def mask_to_roi(binary_image, roi, padding: int = 6):
    image_height, image_width = binary_image.shape[:2]
    x, y, width, height = roi
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_width, x + width + padding)
    y2 = min(image_height, y + height + padding)

    masked_image = binary_image * 0
    masked_image[y1:y2, x1:x2] = binary_image[y1:y2, x1:x2]
    return masked_image


def select_best_character_group(boxes, image_width: int, image_height: int, y_tolerance_ratio: float):
    if not boxes:
        return []

    y_tolerance = image_height * y_tolerance_ratio
    centers_y = [y + height / 2 for _, y, _, height in boxes]
    best_group = []
    best_score = -1.0

    for center_y in centers_y:
        row_boxes = [
            box
            for box in boxes
            if abs((box[1] + box[3] / 2) - center_y) <= y_tolerance
        ]
        if len(row_boxes) < 2:
            continue

        median_height = sorted(height for _, _, _, height in row_boxes)[len(row_boxes) // 2]
        height_tolerance = median_height * 0.65

        filtered_boxes = [
            box
            for box in row_boxes
            if abs(box[3] - median_height) <= height_tolerance
        ]

        for group in split_horizontal_groups(filtered_boxes, image_width):
            if len(group) < 2:
                continue
            score = score_character_group(group, image_width, image_height)
            if score > best_score:
                best_score = score
                best_group = group

    if best_group:
        return sorted(best_group, key=lambda box: box[0])

    return sorted(boxes, key=lambda box: box[0])


def split_horizontal_groups(boxes, image_width: int):
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

    return groups


def score_character_group(group, image_width: int, image_height: int):
    sorted_group = sorted(group, key=lambda box: box[0])
    heights = sorted(height for _, _, _, height in sorted_group)
    median_height = heights[len(heights) // 2]
    group_left = sorted_group[0][0]
    group_right = sorted_group[-1][0] + sorted_group[-1][2]
    group_width = group_right - group_left
    center_x = (group_left + group_right) / 2
    center_y = sorted(y + height / 2 for _, y, _, height in sorted_group)[len(sorted_group) // 2]

    center_distance_ratio = abs(center_x - image_width / 2) / (image_width / 2)
    center_weight = max(0.35, 1.0 - center_distance_ratio)
    lower_weight = 0.65 + center_y / image_height
    compactness = len(sorted_group) / max(group_width / max(median_height, 1), 1)

    return len(sorted_group) * median_height * center_weight * lower_weight * (1 + compactness)


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


def refine_character_boxes(boxes, min_count: int = 6, max_count: int = 7):
    if not boxes:
        return []

    refined_boxes = split_wide_boxes(boxes, min_count=min_count, max_count=max_count)
    if len(refined_boxes) > max_count:
        refined_boxes = keep_best_character_boxes(
            refined_boxes,
            min_count=min_count,
            max_count=max_count,
        )
    elif len(refined_boxes) < min_count:
        refined_boxes = split_wide_boxes(
            refined_boxes,
            min_count=min_count,
            max_count=max_count,
            aggressive=True,
        )

    if len(refined_boxes) > max_count:
        refined_boxes = keep_best_character_boxes(
            refined_boxes,
            min_count=min_count,
            max_count=max_count,
        )

    return sorted(refined_boxes, key=lambda box: box[0])


def split_wide_boxes(boxes, min_count: int, max_count: int, aggressive: bool = False):
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    if len(sorted_boxes) >= max_count:
        return sorted_boxes

    widths = sorted(width for _, _, width, _ in sorted_boxes)
    heights = sorted(height for _, _, _, height in sorted_boxes)
    median_width = widths[len(widths) // 2]
    median_height = heights[len(heights) // 2]
    split_threshold = median_width * (1.45 if aggressive else 1.8)

    split_boxes = []
    for index, (x, y, width, height) in enumerate(sorted_boxes):
        remaining_original = len(sorted_boxes) - index - 1
        room_for_extra_box = len(split_boxes) + remaining_original + 1 < max_count
        aspect_ratio = width / height
        should_split = (
            room_for_extra_box
            and len(sorted_boxes) < min_count
            and width >= split_threshold
            and width >= median_width * 1.35
            and height >= median_height * 0.65
            and aspect_ratio > (0.65 if aggressive else 0.85)
        )
        if should_split:
            left_width = width // 2
            right_width = width - left_width
            split_boxes.append((x, y, left_width, height))
            split_boxes.append((x + left_width, y, right_width, height))
        else:
            split_boxes.append((x, y, width, height))

    return sorted(split_boxes, key=lambda box: box[0])


def keep_best_character_boxes(boxes, min_count: int, max_count: int):
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    if len(sorted_boxes) <= max_count:
        return sorted_boxes

    best_group = []
    best_score = -1.0
    for target_count in range(min(max_count, len(sorted_boxes)), min_count - 1, -1):
        for start_index in range(0, len(sorted_boxes) - target_count + 1):
            group = sorted_boxes[start_index:start_index + target_count]
            score = score_refined_group(group)
            if score > best_score:
                best_score = score
                best_group = group

    if best_group:
        return sorted(best_group, key=lambda box: box[0])

    return sorted_boxes[:max_count]


def score_refined_group(group):
    sorted_group = sorted(group, key=lambda box: box[0])
    heights = sorted(height for _, _, _, height in sorted_group)
    widths = sorted(width for _, _, width, _ in sorted_group)
    centers_y = sorted(y + height / 2 for _, y, _, height in sorted_group)
    median_height = heights[len(heights) // 2]
    median_width = widths[len(widths) // 2]
    median_center_y = centers_y[len(centers_y) // 2]

    height_errors = [abs(height - median_height) / max(median_height, 1) for _, _, _, height in sorted_group]
    width_errors = [abs(width - median_width) / max(median_width, 1) for _, _, width, _ in sorted_group]
    y_errors = [abs((y + height / 2) - median_center_y) / max(median_height, 1) for _, y, _, height in sorted_group]
    aspect_errors = [abs((width / height) - 0.45) for _, _, width, height in sorted_group]

    gaps = []
    for previous_box, box in zip(sorted_group, sorted_group[1:]):
        gaps.append(max(0, box[0] - (previous_box[0] + previous_box[2])))
    if gaps:
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2]
        gap_errors = [abs(gap - median_gap) / max(median_height, 1) for gap in gaps]
    else:
        gap_errors = [0]

    group_left = sorted_group[0][0]
    group_right = sorted_group[-1][0] + sorted_group[-1][2]
    group_width = group_right - group_left
    density = len(sorted_group) / max(group_width / max(median_height, 1), 1)

    penalty = (
        sum(height_errors) / len(height_errors) * 0.30
        + sum(width_errors) / len(width_errors) * 0.18
        + sum(y_errors) / len(y_errors) * 0.32
        + sum(aspect_errors) / len(aspect_errors) * 0.10
        + sum(gap_errors) / len(gap_errors) * 0.10
    )
    return len(sorted_group) * 2.0 + density - penalty


def write_result_file(result_path: Path, image_name: str, boxes) -> None:
    lines = [image_name, str(len(boxes))]
    for x, y, width, height in boxes:
        lines.append(f"{x + 1} {y + 1} {width} {height}")
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
