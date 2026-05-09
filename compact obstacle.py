import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def make_sphere(center, radius, n=24):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_sphere(ax, center, radius, alpha=0.20):
    x, y, z = make_sphere(center, radius)
    ax.plot_surface(x, y, z, alpha=alpha, linewidth=0)


def draw_box(ax, center, size):
    cx, cy, cz = np.array(center, dtype=float)
    sx, sy, sz = np.array(size, dtype=float) / 2.0
    corners = np.array([
        [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
        [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
        [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
        [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz]
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]], linewidth=1.5)


def remove_duplicate_spheres(spheres, tol=1e-8):
    unique = []
    for c, r in spheres:
        duplicated = False
        for c_old, r_old in unique:
            if np.linalg.norm(c - c_old) < tol:
                duplicated = True
                break
        if not duplicated:
            unique.append((c, r))
    return unique


def generate_variable_radius_spheres_for_segment(start, end,
                                                 half_width, half_height,
                                                 num_spheres=4,
                                                 safety_factor=1.03):
    """
    沿长条分支生成少量、非等半径的包络球。
    两端球较小，中间球较大，以降低保守性。
    """

    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    axis_vec = end - start
    length = np.linalg.norm(axis_vec)

    if length < 1e-12:
        rho = np.sqrt(half_width**2 + half_height**2)
        return [(start, safety_factor * rho)]

    direction = axis_vec / length
    rho = np.sqrt(half_width**2 + half_height**2)

    num_spheres = max(2, int(num_spheres))

    # 在轴线上均匀布置球心
    s_list = np.linspace(0.0, length, num_spheres)

    spheres = []

    for i, s_i in enumerate(s_list):
        center = start + direction * s_i

        # 该球负责覆盖的轴向区间 [a_i, b_i]
        if i == 0:
            a_i = 0.0
            b_i = 0.5 * (s_list[i] + s_list[i+1])
        elif i == num_spheres - 1:
            a_i = 0.5 * (s_list[i-1] + s_list[i])
            b_i = length
        else:
            a_i = 0.5 * (s_list[i-1] + s_list[i])
            b_i = 0.5 * (s_list[i] + s_list[i+1])

        axial_span = max(s_i - a_i, b_i - s_i)

        radius = safety_factor * np.sqrt(rho**2 + axial_span**2)
        spheres.append((center, radius))

    return spheres


# ================================
# 1. 近圆/紧凑障碍物：单球包络
# ================================
compact_center = np.array([0.0, 0.0, 1.0])
compact_size = np.array([1.2, 1.0, 1.1])
compact_radius = 0.5 * np.linalg.norm(compact_size)


# ================================
# 2. 长条障碍物：4个变半径球
# ================================
elongated_start = np.array([3.0, -2.0, 1.0])
elongated_end = np.array([8.0, -2.0, 1.0])
elongated_length = np.linalg.norm(elongated_end - elongated_start)

elongated_half_width = 0.35
elongated_half_height = 0.35

elongated_spheres = generate_variable_radius_spheres_for_segment(
    elongated_start,
    elongated_end,
    half_width=elongated_half_width,
    half_height=elongated_half_height,
    num_spheres=4,
    safety_factor=1.03
)


# ================================
# 3. 凹形障碍物：每个分支4个变半径球
# ================================
concave_half_width = 0.45
concave_half_height = 0.35

concave_segments = [
    (np.array([0.0, 3.0, 1.0]), np.array([0.0, 6.0, 1.0])),
    (np.array([3.0, 3.0, 1.0]), np.array([3.0, 6.0, 1.0])),
    (np.array([0.0, 3.0, 1.0]), np.array([3.0, 3.0, 1.0])),
]

concave_spheres = []
for start, end in concave_segments:
    seg_spheres = generate_variable_radius_spheres_for_segment(
        start,
        end,
        half_width=concave_half_width,
        half_height=concave_half_height,
        num_spheres=4,
        safety_factor=1.03
    )
    concave_spheres.extend(seg_spheres)

concave_spheres = remove_duplicate_spheres(concave_spheres)


# ================================
# 可视化
# ================================
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

# 近圆障碍物
draw_box(ax, compact_center, compact_size)
draw_sphere(ax, compact_center, compact_radius, alpha=0.18)
ax.scatter([compact_center[0]], [compact_center[1]], [compact_center[2]], s=25)
ax.text(compact_center[0] - 0.8,
        compact_center[1] - 0.8,
        compact_center[2] + 1.4,
        "Compact obstacle\nsingle sphere")

# 长条障碍物
draw_box(ax, (elongated_start + elongated_end) / 2,
         [elongated_length, 2*elongated_half_width, 2*elongated_half_height])

for c, r in elongated_spheres:
    draw_sphere(ax, c, r, alpha=0.15)

ax.scatter([c[0] for c, r in elongated_spheres],
           [c[1] for c, r in elongated_spheres],
           [c[2] for c, r in elongated_spheres], s=18)

ax.plot([c[0] for c, r in elongated_spheres],
        [c[1] for c, r in elongated_spheres],
        [c[2] for c, r in elongated_spheres], linewidth=1.2)

ax.text(4.0, -3.0, 2.2,
        "Elongated obstacle\nfew variable-radius spheres")

# 凹形障碍物
for start, end in concave_segments:
    center = (start + end) / 2
    seg_len = np.linalg.norm(end - start)

    if abs(start[0] - end[0]) < 1e-9:
        draw_box(ax, center, [2*concave_half_width, seg_len, 2*concave_half_height])
    else:
        draw_box(ax, center, [seg_len, 2*concave_half_width, 2*concave_half_height])

for c, r in concave_spheres:
    draw_sphere(ax, c, r, alpha=0.13)

ax.scatter([c[0] for c, r in concave_spheres],
           [c[1] for c, r in concave_spheres],
           [c[2] for c, r in concave_spheres], s=15)

ax.text(0.2, 5.6, 2.2,
        "Concave obstacle\ndecomposition + variable-radius spheres")

ax.set_title("Efficient Spherical Envelope with Variable Radii")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(-1.5, 9.0)
ax.set_ylim(-3.2, 7.0)
ax.set_zlim(0.0, 3.4)

plt.tight_layout()
plt.show()


# ================================
# 输出参数
# ================================
print("Compact obstacle:")
print("center =", compact_center)
print("radius =", compact_radius)

print("\nElongated obstacle:")
print("number of spheres =", len(elongated_spheres))
for i, (c, r) in enumerate(elongated_spheres):
    print(f"sphere {i}: center = {c}, radius = {r}")

print("\nConcave obstacle:")
print("number of spheres =", len(concave_spheres))
for i, (c, r) in enumerate(concave_spheres):
    print(f"sphere {i}: center = {c}, radius = {r}")