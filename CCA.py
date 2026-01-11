import cv2
import numpy as np

def compute_adaptive_radii_rotated(img, base_radius=5, row_height=100, alpha=10.0, beta=7.0):
    """
    Compute adaptive radii on the image AFTER rotating clockwise 90°.
    Returns radii_per_row for the rotated image coordinate frame and the rotated binary image.
    radii_per_row is a list of (y0, y1, a, b) where y indexes rows in the *rotated* image.
    """
    # enforce binary
    bin_img = (img > 0).astype(np.uint8)

    # rotate clockwise -> this is the same rotation used in CCA function
    rot = cv2.rotate(bin_img, cv2.ROTATE_90_CLOCKWISE)
    Hr, Wr = rot.shape

    radii_per_row = []
    fun = lambda x: alpha * np.exp(-beta * x)  # smoother decay

    for y in range(0, Hr, row_height):
        slice_h = min(row_height, Hr - y)
        row_slice = rot[y:y + slice_h, :]

        # density as fraction of foreground pixels (0..1)
        density = float(row_slice.mean())
        print(f"the th row slice: density = {density}")
        raw_radius = fun(density)

        a = int(round(max(base_radius, raw_radius)))
        b = int(round(max(1, max(1, raw_radius / 3.0), base_radius // 2)))

        # clamp to image extents
        a = min(a, max(1, Wr // 2))
        b = min(b, max(1, slice_h // 2))

        radii_per_row.append((y, y + slice_h, a, b))
        # optional debug:
        # print(f"[rot] Row {y}:{y+slice_h} density={density:.4f} -> a={a}, b={b}")

    return radii_per_row, rot


def make_ellipse_offsets(a, b):
    """Return list of (dy, dx) integer offsets inside an ellipse of radii a (x) and b (y)."""
    offsets = []
    aa = float(max(1, a * a))
    bb = float(max(1, b * b))
    for dy in range(-b, b + 1):
        for dx in range(-a, a + 1):
            if (dx * dx) / aa + (dy * dy) / bb <= 1.0:
                offsets.append((dy, dx))
    return offsets


def connected_components_adaptive_plus_rotated(rot_bin_img, radii_per_row, base_radius=5):
    """
    Run adaptive-ellipse CCA on an image that is already rotated clockwise.
    Returns labels (same shape as rot_bin_img) and number of labels.

    Note: radii_per_row contains (y0, y1, a, b) computed in rotated-frame logic.
    To compensate for the 90° rotation (so the ellipse elongation follows the original
    text-horizontal direction), we use the swapped ellipse (b, a) when creating offsets.
    """
    bin_img = (rot_bin_img > 0).astype(np.uint8)
    H, W = bin_img.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0

    # Precompute ellipse offsets for all (a,b) combos in radii_per_row
    # Store both (a,b) and their swapped (b,a) variants; we'll use (b,a) for growth.
    ellipse_masks = {}
    for _, _, a, b in radii_per_row:
        key_ab = (int(a), int(b))
        key_ba = (int(b), int(a))
        if key_ab not in ellipse_masks:
            ellipse_masks[key_ab] = make_ellipse_offsets(*key_ab)
        if key_ba not in ellipse_masks:
            ellipse_masks[key_ba] = make_ellipse_offsets(*key_ba)

    # Ensure default offsets exists (also keep swapped default)
    default_key = (base_radius, base_radius)
    swapped_default_key = (base_radius, base_radius)  # same if square; kept for clarity
    if default_key not in ellipse_masks:
        ellipse_masks[default_key] = make_ellipse_offsets(*default_key)
    if swapped_default_key not in ellipse_masks:
        ellipse_masks[swapped_default_key] = make_ellipse_offsets(*swapped_default_key)
    # Default offsets we will use (swapped orientation)
    default_offsets = ellipse_masks[swapped_default_key]

    def offsets_for_y(y):
        # find the (a,b) for this rotated-row y and return the *swapped* offsets (b,a)
        for y0, y1, a, b in radii_per_row:
            if y0 <= y < y1:
                return ellipse_masks[(int(b), int(a))]  # swapped here
        return default_offsets

    # flood-fill: mark label when pushing to avoid duplicates
    for y in range(H):
        offsets = offsets_for_y(y)
        for x in range(W):
            if bin_img[y, x] and labels[y, x] == 0:
                current_label += 1
                labels[y, x] = current_label
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    # offsets chosen relative to rotated image coordinates
                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if bin_img[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                stack.append((ny, nx))

    return labels, current_label




def extract_text_components_with_rotation(img_binary, row_height=100, base_radius=10,
                                          v_overlap_thresh=0.35, max_gap=0.7):
    """
    Pipeline that preserves the 'rotation trick' and includes dynamic line grouping:
      1. Rotate input (inside compute_adaptive_radii_rotated)
      2. Compute radii per rotated-row
      3. Run CCA on entire rotated image (adaptive ellipse neighborhoods)
      4. Rotate label map back and extract boxes in original coords
      5. Merge overlapping boxes greedily (keeps your original behavior)
      6. Group merged boxes into lines dynamically (vertical overlap / center heuristic)
      7. Sort boxes left->right within each line and lines top->bottom
    Returns:
      merged_sorted: list of [label, x, y, w, h] in original image coords (reading order)
      labels_global: label map in original image orientation
    """

    # --- internal helpers ---
    def vertical_overlap_ratio(boxA, boxB):
        # box format: (x, y, w, h)
        _, ay, _, ah = boxA
        _, by, _, bh = boxB
        a0, a1 = ay, ay + ah - 1
        b0, b1 = by, by + bh - 1
        inter0 = max(a0, b0)
        inter1 = min(a1, b1)
        inter_h = max(0, inter1 - inter0 + 1)
        min_h = min(ah, bh) if min(ah, bh) > 0 else 1
        return inter_h / min_h

    def group_boxes_into_lines(boxes, v_overlap_thresh_local, max_gap_local):
        """
        Group bounding boxes into text lines dynamically and return list of lines (lists of boxes).
        Each box is [id, x, y, w, h].
        """
        if not boxes:
            return []

        # Sort by top y for deterministic processing
        boxes_sorted = sorted(boxes, key=lambda b: b[2])  # b[2] == y

        lines = []  # each entry: dict with 'boxes', 'y_min', 'y_max', 'centers'
        for b in boxes_sorted:
            _, x, y, w, h = b
            placed = False
            box_center_y = y + h / 2.0

            for ln in lines:
                # compute vertical overlap between this box and the line bbox
                line_y_min = ln['y_min']
                line_y_max = ln['y_max']
                line_h = line_y_max - line_y_min + 1
                inter0 = max(y, line_y_min)
                inter1 = min(y + h - 1, line_y_max)
                inter_h = max(0, inter1 - inter0 + 1)
                denom = min(h, line_h) if min(h, line_h) > 0 else 1
                v_overlap = inter_h / denom

                # center distance heuristic
                line_center_y = (line_y_min + line_y_max) / 2.0
                avg_h = (h + line_h) / 2.0
                center_dist = abs(box_center_y - line_center_y)

                if v_overlap >= v_overlap_thresh_local or center_dist <= max_gap_local * avg_h:
                    # add to this line
                    ln['boxes'].append(b)
                    ln['y_min'] = min(ln['y_min'], y)
                    ln['y_max'] = max(ln['y_max'], y + h - 1)
                    ln['centers'].append(box_center_y)
                    placed = True
                    break

            if not placed:
                lines.append({
                    'boxes': [b],
                    'y_min': y,
                    'y_max': y + h - 1,
                    'centers': [box_center_y]
                })

        # Finalize: sort boxes left->right within each line and sort lines top->bottom by median center
        for ln in lines:
            ln['median_center'] = np.median(ln['centers'])
            ln['boxes'] = sorted(ln['boxes'], key=lambda bb: bb[1])  # bb[1] is x

        lines_sorted = sorted(lines, key=lambda ln: ln['median_center'])
        return [ln['boxes'] for ln in lines_sorted]

    # -----------------------
    # Main pipeline (keeps original behavior up to box extraction)
    # -----------------------
    if img_binary is None:
        raise ValueError("img_binary must be provided")

    # Step 1: compute radii on rotated image (and get rotated binary)
    radii_per_row, rot_bin = compute_adaptive_radii_rotated(img_binary, base_radius=base_radius, row_height=row_height)

    # Step 2: run CCA on rotated image
    rot_labels, nlabels = connected_components_adaptive_plus_rotated(rot_bin, radii_per_row, base_radius=base_radius)

    # Step 3: rotate labels back to original orientation (counter-clockwise)
    labels_global = cv2.rotate(rot_labels, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Step 4: compute bounding boxes for each global label
    boxes = []
    unique = np.unique(labels_global)
    for lab in unique:
        if lab == 0:
            continue
        ys, xs = np.where(labels_global == lab)
        if ys.size == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        w_box, h_box = x1 - x0 + 1, y1 - y0 + 1
        boxes.append([int(lab), x0, y0, w_box, h_box])

    # Step 5: merge overlapping boxes (your original greedy pairwise merge)
    merged, used = [], set()
    for i in range(len(boxes)):
        if i in used:
            continue
        _, xi, yi, wi, hi = boxes[i]
        bx0, by0, bx1, by1 = xi, yi, xi + wi - 1, yi + hi - 1
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            _, xj, yj, wj, hj = boxes[j]
            nbx0, nby0, nbx1, nby1 = xj, yj, xj + wj - 1, yj + hj - 1
            # overlap test
            if not (nbx1 < bx0 or nbx0 > bx1 or nby1 < by0 or nby0 > by1):
                bx0 = min(bx0, nbx0)
                by0 = min(by0, nby0)
                bx1 = max(bx1, nbx1)
                by1 = max(by1, nby1)
                used.add(j)
        merged.append([boxes[i][0], bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1])

    # Step 6: group merged boxes into lines dynamically, then left->right within each line
    lines = group_boxes_into_lines(merged, v_overlap_thresh_local=v_overlap_thresh, max_gap_local=max_gap)
    ordered_boxes = []
    for line in lines:
        for b in line:
            ordered_boxes.append(b)

    # This is the final reading-order list
    merged_sorted = ordered_boxes

    return merged_sorted, labels_global


def visualize_components(img, boxes):
    vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for idx, (bid, x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(idx), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis
