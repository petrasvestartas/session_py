"""2D triangulation via ear clipping with hole merging."""


def _cross_2d(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _signed_area_2d(coords):
    area = 0.0
    n = len(coords) // 2
    for i in range(n):
        j = (i + 1) % n
        area += coords[i*2] * coords[j*2+1] - coords[j*2] * coords[i*2+1]
    return area * 0.5


def _point_in_triangle_2d(px, py, ax, ay, bx, by, cx, cy):
    d1 = _cross_2d(ax, ay, bx, by, px, py)
    d2 = _cross_2d(bx, by, cx, cy, px, py)
    d3 = _cross_2d(cx, cy, ax, ay, px, py)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    d1 = _cross_2d(cx, cy, dx, dy, ax, ay)
    d2 = _cross_2d(cx, cy, dx, dy, bx, by)
    d3 = _cross_2d(ax, ay, bx, by, cx, cy)
    d4 = _cross_2d(ax, ay, bx, by, dx, dy)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _is_visible(coords, polygon, from_v, to_v):
    ax, ay = coords[from_v*2], coords[from_v*2+1]
    bx, by = coords[to_v*2], coords[to_v*2+1]
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        vi, vj = polygon[i], polygon[j]
        if vi == from_v or vi == to_v or vj == from_v or vj == to_v:
            continue
        cx, cy = coords[vi*2], coords[vi*2+1]
        dx, dy = coords[vj*2], coords[vj*2+1]
        if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
            return False
    return True


def _ear_clip(coords, indices):
    triangles = []
    n = len(indices)
    if n < 3:
        return triangles
    poly_coords = []
    for i in range(n):
        poly_coords.append(coords[indices[i]*2])
        poly_coords.append(coords[indices[i]*2+1])
    if _signed_area_2d(poly_coords) < 0.0:
        indices = indices[::-1]
    else:
        indices = list(indices)

    def build_reflex(idx):
        r = [False] * len(idx)
        for i in range(len(idx)):
            p = (i - 1) % len(idx)
            nx = (i + 1) % len(idx)
            if _cross_2d(coords[idx[p]*2], coords[idx[p]*2+1],
                         coords[idx[i]*2], coords[idx[i]*2+1],
                         coords[idx[nx]*2], coords[idx[nx]*2+1]) <= 0.0:
                r[i] = True
        return r

    reflex = build_reflex(indices)
    max_iter = n * n * 2
    it = 0
    while len(indices) > 3 and it < max_iter:
        found = False
        for i in range(len(indices)):
            if reflex[i]:
                continue
            p = (i - 1) % len(indices)
            nx = (i + 1) % len(indices)
            ax, ay = coords[indices[p]*2], coords[indices[p]*2+1]
            bx, by = coords[indices[i]*2], coords[indices[i]*2+1]
            cx, cy = coords[indices[nx]*2], coords[indices[nx]*2+1]
            is_ear = True
            for k in range(len(indices)):
                if k == p or k == i or k == nx:
                    continue
                if not reflex[k]:
                    continue
                px, py = coords[indices[k]*2], coords[indices[k]*2+1]
                if (abs(px-ax) < 1e-12 and abs(py-ay) < 1e-12) or \
                   (abs(px-bx) < 1e-12 and abs(py-by) < 1e-12) or \
                   (abs(px-cx) < 1e-12 and abs(py-cy) < 1e-12):
                    continue
                if _point_in_triangle_2d(px, py, ax, ay, bx, by, cx, cy):
                    is_ear = False
                    break
            if is_ear:
                triangles.append((indices[p], indices[i], indices[nx]))
                indices.pop(i)
                reflex = build_reflex(indices)
                found = True
                break
        if not found:
            removed = False
            for i in range(len(indices)):
                p = (i - 1) % len(indices)
                nx = (i + 1) % len(indices)
                c = _cross_2d(coords[indices[p]*2], coords[indices[p]*2+1],
                              coords[indices[i]*2], coords[indices[i]*2+1],
                              coords[indices[nx]*2], coords[indices[nx]*2+1])
                if abs(c) < 1e-12:
                    triangles.append((indices[p], indices[i], indices[nx]))
                    indices.pop(i)
                    reflex = build_reflex(indices)
                    removed = True
                    break
            if not removed:
                coord_map = {}
                for pos in range(len(indices)):
                    key = (round(coords[indices[pos]*2], 8), round(coords[indices[pos]*2+1], 8))
                    if key in coord_map:
                        split_first = coord_map[key]
                        poly_a = indices[split_first:pos]
                        poly_b = indices[pos:] + indices[:split_first]
                        triangles.extend(_ear_clip(coords, poly_a))
                        triangles.extend(_ear_clip(coords, poly_b))
                        break
                    coord_map[key] = pos
                break
        it += 1
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def _merge_holes(coords, boundary_indices, hole_indices_list):
    if not hole_indices_list:
        return list(boundary_indices)
    sorted_holes = []
    for h, hole in enumerate(hole_indices_list):
        mx = -1e300
        mx_local = 0
        for i, vi in enumerate(hole):
            x = coords[vi*2]
            if x > mx:
                mx = x
                mx_local = i
        sorted_holes.append((h, mx, mx_local))
    sorted_holes.sort(key=lambda t: -t[1])
    merged = list(boundary_indices)
    for h_idx, _, hole_vi in sorted_holes:
        hole = hole_indices_list[h_idx]
        hx = coords[hole[hole_vi]*2]
        hy = coords[hole[hole_vi]*2+1]
        best_ix = 1e300
        best_edge = -1
        for i in range(len(merged)):
            j = (i + 1) % len(merged)
            y0 = coords[merged[i]*2+1]
            y1 = coords[merged[j]*2+1]
            x0 = coords[merged[i]*2]
            x1 = coords[merged[j]*2]
            if (y0 <= hy and y1 > hy) or (y1 <= hy and y0 > hy):
                t = (hy - y0) / (y1 - y0)
                ix = x0 + t * (x1 - x0)
                if ix >= hx and ix < best_ix:
                    best_ix = ix
                    best_edge = i
        if best_edge < 0:
            continue
        be0 = best_edge
        be1 = (best_edge + 1) % len(merged)
        vx0 = coords[merged[be0]*2]
        vx1 = coords[merged[be1]*2]
        candidate = be0 if vx0 >= vx1 else be1
        vis_x = coords[merged[candidate]*2]
        vis_y = coords[merged[candidate]*2+1]
        min_dist = 1e300
        visible = candidate
        for i in range(len(merged)):
            px = coords[merged[i]*2]
            py = coords[merged[i]*2+1]
            if px < hx:
                continue
            if _point_in_triangle_2d(px, py, hx, hy, best_ix, hy, vis_x, vis_y) or \
               (abs(px-vis_x) < 1e-12 and abs(py-vis_y) < 1e-12):
                dx, dy = px - hx, py - hy
                dist = dx*dx + dy*dy
                if dist < min_dist and _is_visible(coords, merged, hole[hole_vi], merged[i]):
                    min_dist = dist
                    visible = i
        if visible == candidate and not _is_visible(coords, merged, hole[hole_vi], merged[candidate]):
            min_dist = 1e300
            for i in range(len(merged)):
                px = coords[merged[i]*2]
                py = coords[merged[i]*2+1]
                dx, dy = px - hx, py - hy
                dist = dx*dx + dy*dy
                if dist < min_dist and _is_visible(coords, merged, hole[hole_vi], merged[i]):
                    min_dist = dist
                    visible = i
        new_merged = merged[:visible+1]
        hole_n = len(hole)
        for i in range(hole_n + 1):
            new_merged.append(hole[(hole_vi + i) % hole_n])
        new_merged.append(merged[visible])
        new_merged.extend(merged[visible+1:])
        merged = new_merged
    return merged


def point_in_polygon_2d(px, py, coords):
    winding = 0
    n = len(coords) // 2
    for i in range(n):
        j = (i + 1) % n
        y0, y1 = coords[i*2+1], coords[j*2+1]
        if y0 <= py:
            if y1 > py:
                if _cross_2d(coords[i*2], y0, coords[j*2], y1, px, py) > 0:
                    winding += 1
        else:
            if y1 <= py:
                if _cross_2d(coords[i*2], y0, coords[j*2], y1, px, py) < 0:
                    winding -= 1
    return winding != 0


def triangulate(outer_pts, hole_pts_list=None):
    """Ear-clip triangulation of 2D polygon with holes.
    outer_pts: list of Point (uses [0],[1] as x,y)
    hole_pts_list: list of lists of Point
    Returns: list of (v0,v1,v2) index tuples"""
    if len(outer_pts) < 3:
        return []
    coords = []
    bn = len(outer_pts)
    if bn > 1 and abs(outer_pts[0][0]-outer_pts[bn-1][0]) < 1e-12 and \
       abs(outer_pts[0][1]-outer_pts[bn-1][1]) < 1e-12:
        bn -= 1
    no_holes = not hole_pts_list
    if no_holes:
        if bn == 3:
            return [(0, 1, 2)]
        if bn == 4:
            quad_convex = True
            qfirst = 0.0
            for qi in range(4):
                c = _cross_2d(outer_pts[qi%4][0], outer_pts[qi%4][1],
                              outer_pts[(qi+1)%4][0], outer_pts[(qi+1)%4][1],
                              outer_pts[(qi+2)%4][0], outer_pts[(qi+2)%4][1])
                if abs(c) < 1e-12:
                    continue
                if qfirst == 0.0:
                    qfirst = c
                elif (c > 0) != (qfirst > 0):
                    quad_convex = False
                    break
            if quad_convex:
                d02 = (outer_pts[0][0]-outer_pts[2][0])**2 + (outer_pts[0][1]-outer_pts[2][1])**2
                d13 = (outer_pts[1][0]-outer_pts[3][0])**2 + (outer_pts[1][1]-outer_pts[3][1])**2
                if d02 <= d13:
                    return [(0, 1, 2), (0, 2, 3)]
                else:
                    return [(0, 1, 3), (1, 2, 3)]
        convex = True
        first_cross = 0.0
        for i in range(bn):
            j = (i + 1) % bn
            k = (i + 2) % bn
            c = _cross_2d(outer_pts[i][0], outer_pts[i][1], outer_pts[j][0], outer_pts[j][1], outer_pts[k][0], outer_pts[k][1])
            if abs(c) < 1e-12:
                continue
            if first_cross == 0.0:
                first_cross = c
            elif (c > 0) != (first_cross > 0):
                convex = False
                break
        if convex and bn >= 5:
            return [(0, i, i + 1) for i in range(1, bn - 1)]
    boundary_indices = []
    for i in range(bn):
        idx = len(coords) // 2
        coords.append(outer_pts[i][0])
        coords.append(outer_pts[i][1])
        boundary_indices.append(idx)
    bcoords = []
    for i in range(bn):
        bcoords.append(coords[boundary_indices[i]*2])
        bcoords.append(coords[boundary_indices[i]*2+1])
    if _signed_area_2d(bcoords) < 0.0:
        boundary_indices.reverse()
    hole_indices_list = []
    if hole_pts_list:
        for hole in hole_pts_list:
            hn = len(hole)
            if hn < 3:
                continue
            if hn > 1 and abs(hole[0][0]-hole[hn-1][0]) < 1e-12 and \
               abs(hole[0][1]-hole[hn-1][1]) < 1e-12:
                hn -= 1
            hole_indices = []
            for i in range(hn):
                idx = len(coords) // 2
                coords.append(hole[i][0])
                coords.append(hole[i][1])
                hole_indices.append(idx)
            hcoords = []
            for i in range(hn):
                hcoords.append(coords[hole_indices[i]*2])
                hcoords.append(coords[hole_indices[i]*2+1])
            if _signed_area_2d(hcoords) > 0.0:
                hole_indices.reverse()
            hole_indices_list.append(hole_indices)
    merged = _merge_holes(coords, boundary_indices, hole_indices_list)
    return _ear_clip(coords, merged)
