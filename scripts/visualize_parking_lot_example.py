#!/usr/bin/env python3
"""Open3D visualization for the parking-lot DCReg example.

This script is intentionally kept outside the C++ core. The DCReg executable
exports lightweight artifacts, and Python/Open3D only handles visualization:

  ./DCReg/build/dcreg_parking_lot_example
  python3 scripts/visualize_parking_lot_example.py

Color convention:
  gray-scale intensity - 0.5 m ground-plane sampled full prior map
  orange intensity     - source frame transformed by the initial prediction
  blue intensity       - source frame transformed by the final DCReg result
  amber                - weakest observable physical axis
  red                  - degenerate physical axis, if the DCReg mask marks one
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_EXPORT_DIR = Path("DCReg/dataset/Parking-Lot-example/visualization")
DEFAULT_SCREENSHOT_NAME = "dcreg_parking_lot_view.png"
ROT_LABELS = ("roll", "pitch", "yaw")
TRANS_LABELS = ("x", "y", "z")


def load_open3d():
    try:
        import open3d as o3d  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise SystemExit(
            "Open3D is required for this visualization.\n"
            "Install it with: python3 -m pip install open3d numpy"
        ) from error
    return o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the exported parking-lot DCReg example."
    )
    parser.add_argument(
        "export_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Directory produced by dcreg_parking_lot_example --export_vis.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="PNG path. Defaults to <export_dir>/dcreg_parking_lot_view.png.",
    )
    parser.add_argument(
        "--no_screenshot",
        action="store_true",
        help="Do not save the default screenshot.",
    )
    parser.add_argument(
        "--plain_screenshot",
        action="store_true",
        help="Save the screenshot without the compact 2D information card.",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Save/validate outputs without opening the interactive viewer.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Open3D point size for the interactive viewer.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate exported files and print diagnostics without opening Open3D.",
    )
    parser.add_argument(
        "--verbose_labels",
        action="store_true",
        help="Show detailed per-axis lambda/ratio labels in the 3D viewer.",
    )
    parser.add_argument(
        "--show_settings",
        action="store_true",
        help="Show the Open3D settings panel for debugging.",
    )
    parser.add_argument(
        "--show_open3d_helpers",
        action="store_true",
        help="Show Open3D's built-in axes and ground grid.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return path


# ===== BEGIN CHANGE: intensity-aware PCD loading =====
def read_ascii_pcd_xyz_intensity(path: Path):
    fields = []
    data_type = ""
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            header_lines += 1
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "FIELDS":
                fields = tokens[1:]
            elif tokens[0] == "DATA":
                data_type = tokens[1].lower() if len(tokens) > 1 else ""
                break

    if data_type != "ascii" or not {"x", "y", "z"}.issubset(fields):
        return None, None

    xyz_columns = [fields.index(axis) for axis in ("x", "y", "z")]
    intensity_column = fields.index("intensity") if "intensity" in fields else None
    use_columns = xyz_columns + ([] if intensity_column is None else [intensity_column])
    values = np.loadtxt(path, skiprows=header_lines, usecols=use_columns)
    values = np.atleast_2d(values)
    points = values[:, :3]
    intensity = values[:, 3] if intensity_column is not None else None
    return points, intensity


def intensity_colors(
    intensity: np.ndarray | None, tint: tuple[float, float, float], point_count: int
):
    tint_array = np.asarray(tint, dtype=float)
    if intensity is None or intensity.size == 0:
        return np.tile(tint_array, (point_count, 1))

    finite = intensity[np.isfinite(intensity)]
    if finite.size == 0:
        normalized = np.ones_like(intensity, dtype=float)
    else:
        low, high = np.percentile(finite, [2.0, 98.0])
        if high <= low:
            normalized = np.ones_like(intensity, dtype=float)
        else:
            normalized = np.clip((intensity - low) / (high - low), 0.0, 1.0)

    brightness = 0.18 + 0.82 * normalized
    return np.clip(brightness[:, None] * tint_array[None, :], 0.0, 1.0)


def load_intensity_cloud(o3d, path: Path, tint: tuple[float, float, float]):
    path = require_file(path)
    points, intensity = read_ascii_pcd_xyz_intensity(path)
    if points is None:
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise SystemExit(f"empty point cloud: {path}")
        cloud.paint_uniform_color(np.asarray(tint, dtype=float))
        return cloud

    if points.size == 0:
        raise SystemExit(f"empty point cloud: {path}")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(
        intensity_colors(intensity, tint, points.shape[0])
    )
    return cloud
# ===== END CHANGE: intensity-aware PCD loading =====


def matrix_from_json(values: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (4, 4):
        raise SystemExit(f"expected 4x4 transform, got {matrix.shape}")
    return matrix


def rotation_from_z(direction: np.ndarray) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    norm = np.linalg.norm(cross)
    dot = float(np.dot(z_axis, direction))
    if norm < 1e-10:
        if dot > 0.0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])

    cross /= norm
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * skew @ skew
    )


def make_cylinder_axis(o3d, start: np.ndarray, end: np.ndarray,
                       color: tuple[float, float, float], radius: float):
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        raise ValueError("axis length is too small")
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=24
    )
    mesh.compute_vertex_normals()
    mesh.rotate(rotation_from_z(direction), center=np.zeros(3))
    mesh.translate((start + end) * 0.5)
    mesh.paint_uniform_color(color)
    return mesh


def axis_status(mask: list[bool], axis_index: int, is_rotation: bool) -> str:
    offset = 0 if is_rotation else 3
    return "degenerate" if mask[offset + axis_index] else "observable"


def important_axis_summary(diagnostics: dict) -> str:
    degeneracy = diagnostics["degeneracy"]
    mask = degeneracy["degenerate_mask"]
    rot_lambda = np.asarray(degeneracy["aligned_lambda_rpy"])
    trans_lambda = np.asarray(degeneracy["aligned_lambda_xyz"])
    weak_rot = int(np.argmin(rot_lambda))
    weak_trans = int(np.argmin(trans_lambda))

    lines = []
    for index, is_degenerate in enumerate(mask[:3]):
        if is_degenerate:
            ratio = degeneracy["rot_axis_contribution_ratio"][index][index]
            lines.append(f"{ROT_LABELS[index]} rotation, ratio {ratio:.2f}")
    for index, is_degenerate in enumerate(mask[3:]):
        if is_degenerate:
            ratio = degeneracy["trans_axis_contribution_ratio"][index][index]
            lines.append(f"{TRANS_LABELS[index]} translation, ratio {ratio:.2f}")

    if lines:
        return "Degenerate: " + "; ".join(lines)
    return (
        f"Weakest: {ROT_LABELS[weak_rot]} rotation, "
        f"{TRANS_LABELS[weak_trans]} translation"
    )


def mask_text(diagnostics: dict) -> str:
    return "".join(
        "1" if value else "0"
        for value in diagnostics["degeneracy"]["degenerate_mask"]
    )


def make_axis_geometry(o3d, transform: np.ndarray, diagnostics: dict):
    center = transform[:3, 3]
    rotation = transform[:3, :3]
    mask = diagnostics["degeneracy"]["degenerate_mask"]
    rot_lambda = np.asarray(diagnostics["degeneracy"]["aligned_lambda_rpy"])
    trans_lambda = np.asarray(diagnostics["degeneracy"]["aligned_lambda_xyz"])
    weakest_rot = int(np.argmin(rot_lambda))
    weakest_trans = int(np.argmin(trans_lambda))

    base_colors = ((0.85, 0.10, 0.10), (0.10, 0.65, 0.20), (0.10, 0.25, 0.90))
    weak_color = (1.0, 0.55, 0.0)
    degenerate_color = (1.0, 0.0, 0.0)
    geometries = []
    labels = []

    for axis in range(3):
        color = base_colors[axis]
        if axis == weakest_trans:
            color = weak_color
        if mask[3 + axis]:
            color = degenerate_color
        end = center + 18.0 * rotation[:, axis]
        geometries.append(make_cylinder_axis(o3d, center, end, color, radius=0.35))
        labels.append(
            (
                end,
                f"trans {TRANS_LABELS[axis]}: "
                f"lambda={trans_lambda[axis]:.3f}, "
                "ratio="
                f"{diagnostics['degeneracy']['trans_axis_contribution_ratio'][axis][axis]:.3f}, "
                f"{axis_status(mask, axis, is_rotation=False)}",
            )
        )

    rotation_origin = center + np.array([0.0, 0.0, 3.0])
    for axis in range(3):
        color = base_colors[axis]
        if axis == weakest_rot:
            color = weak_color
        if mask[axis]:
            color = degenerate_color
        end = rotation_origin + 10.0 * rotation[:, axis]
        geometries.append(
            make_cylinder_axis(o3d, rotation_origin, end, color, radius=0.24)
        )
        labels.append(
            (
                end,
                f"rot {ROT_LABELS[axis]}: "
                f"lambda={rot_lambda[axis]:.3f}, "
                "ratio="
                f"{diagnostics['degeneracy']['rot_axis_contribution_ratio'][axis][axis]:.3f}, "
                f"{axis_status(mask, axis, is_rotation=True)}",
            )
        )

    return geometries, labels


def print_matrix(title: str, row_labels: tuple[str, ...], values: list[list[float]]) -> None:
    matrix = np.asarray(values, dtype=float)
    print(title)
    for label, row in zip(row_labels, matrix):
        print(f"  {label:>5s}: " + " ".join(f"{value:8.4f}" for value in row))


def print_notes(diagnostics: dict) -> None:
    degeneracy = diagnostics["degeneracy"]
    mask = degeneracy["degenerate_mask"]
    print("\nDCReg visualization notes")
    print("-------------------------")
    print("This scene uses a single parking-lot LiDAR frame as source and a prior map")
    print("as target. The source is shown at both the initial prediction and final")
    print("DCReg result; the target cloud is the full 0.5 m sampled prior map.")
    print(f"Iterations: {diagnostics['registration']['iterations']}")
    print(f"RMSE: {diagnostics['registration']['rmse']:.6f}")
    print(f"Fitness: {diagnostics['registration']['fitness']:.6f}")
    print(f"Degenerate mask [roll pitch yaw x y z]: {mask}")
    print(f"Schur condition, rotation: {degeneracy['cond_schur_rot']:.6f}")
    print(f"Schur condition, translation: {degeneracy['cond_schur_trans']:.6f}")
    print_matrix(
        "Rotation contribution ratio, rows=rpy cols=aligned_rpy:",
        ROT_LABELS,
        degeneracy["rot_axis_contribution_ratio"],
    )
    print_matrix(
        "Translation contribution ratio, rows=xyz cols=aligned_xyz:",
        TRANS_LABELS,
        degeneracy["trans_axis_contribution_ratio"],
    )
    print("Viewer colors: intensity-tinted target/source clouds on a black background.")
    print("Axis colors: amber=weakest observable axis, red=degenerate axis if present.")
    print("Viewer mode: compact presentation labels by default.")
    print("Use --verbose_labels for per-axis lambda/ratio annotations.\n")


def geometry_bounds(o3d, geometries: list):
    points = []
    for geometry in geometries:
        if hasattr(geometry, "get_axis_aligned_bounding_box"):
            box = geometry.get_axis_aligned_bounding_box()
            points.extend([box.get_min_bound(), box.get_max_bound()])
    if not points:
        return o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
    points = np.asarray(points)
    return o3d.geometry.AxisAlignedBoundingBox(points.min(axis=0), points.max(axis=0))


def focus_bounds(o3d, geometries: list):
    # ===== BEGIN CHANGE: focus camera on registration area =====
    # The target map object is now the full 0.5 m sampled prior map.  Fitting
    # the camera to that full map makes the source scan and degeneracy axes too
    # small for a demo.  Keep the full map loaded, but use source/axis geometry
    # for the default camera and label placement.
    # ===== END CHANGE: focus camera on registration area =====
    focus_geometries = geometries[1:] if len(geometries) > 1 else geometries
    return geometry_bounds(o3d, focus_geometries)


def add_screenshot_card(screenshot: Path, diagnostics: dict) -> None:
    # ===== BEGIN CHANGE: clean 2D presentation card =====
    try:
        from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-outside-toplevel
    except ImportError:
        return

    image = Image.open(screenshot).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
        )
        body_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21
        )
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    panel = (32, 30, 650, 215)
    draw.rounded_rectangle(
        panel,
        radius=18,
        fill=(8, 12, 18, 222),
        outline=(255, 255, 255, 72),
        width=2,
    )
    draw.text(
        (58, 50), "DCReg Parking-Lot", fill=(246, 248, 252, 255), font=title_font
    )
    draw.text(
        (58, 91),
        important_axis_summary(diagnostics),
        fill=(190, 30, 30, 255),
        font=body_font,
    )
    draw.text(
        (58, 126),
        f"mask rpy/xyz {mask_text(diagnostics)} | "
        f"RMSE {diagnostics['registration']['rmse']:.3f}",
        fill=(222, 228, 236, 255),
        font=body_font,
    )

    legend = [
        ((210, 210, 210, 255), "map"),
        ((255, 122, 13, 255), "initial"),
        ((13, 82, 255, 255), "DCReg"),
        ((255, 0, 0, 255), "degenerate axis"),
    ]
    x = 58
    for color, label in legend:
        draw.rounded_rectangle((x, 166, x + 24, 190), radius=5, fill=color)
        draw.text((x + 33, 163), label, fill=(222, 228, 236, 255), font=body_font)
        x += 132 if label != "degenerate axis" else 0

    Image.alpha_composite(image, overlay).convert("RGB").save(screenshot)
    # ===== END CHANGE: clean 2D presentation card =====


def save_screenshot(
    o3d,
    geometries: list,
    point_size: float,
    screenshot: Path,
    diagnostics: dict,
    plain_screenshot: bool,
) -> None:
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name="DCReg parking-lot visualization screenshot",
        width=1600,
        height=950,
        visible=True,
    )
    for geometry in geometries:
        visualizer.add_geometry(geometry)

    render = visualizer.get_render_option()
    render.background_color = np.asarray([0.0, 0.0, 0.0])
    render.point_size = point_size

    view = visualizer.get_view_control()
    bounds = focus_bounds(o3d, geometries)
    extent = np.linalg.norm(bounds.get_extent())
    if extent <= 1e-6:
        extent = 1.0
    # ===== BEGIN CHANGE: top-down screenshot camera =====
    # Open3D's classic Visualizer is used for deterministic PNG export.  It
    # does not support O3DVisualizer's text labels, so the saved image focuses
    # on the color-coded clouds and thick degeneracy axes while the interactive
    # viewer carries the full textual diagnostics.  A strict top-down camera
    # makes the initial/final scan displacement and horizontal degeneracy axes
    # easier to compare in the parking-lot scene.
    view.set_lookat(bounds.get_center())
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, 1.0, 0.0])
    view.set_zoom(min(0.5, max(0.18, extent / 120.0)))
    # ===== END CHANGE: top-down screenshot camera =====
    visualizer.poll_events()
    visualizer.update_renderer()
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    visualizer.capture_screen_image(str(screenshot), do_render=True)
    visualizer.destroy_window()
    if not plain_screenshot:
        add_screenshot_card(screenshot, diagnostics)
    print(f"Saved screenshot: {screenshot}")


def add_viewer_labels(
    visualizer,
    bounds,
    diagnostics: dict,
    axis_labels: list[tuple[np.ndarray, str]],
    verbose_labels: bool,
) -> None:
    # ===== BEGIN CHANGE: compact presentation labels =====
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    anchor = np.array(
        [min_bound[0] + 4.0, max_bound[1] - 4.0, max_bound[2] + 5.0]
    )
    degeneracy = diagnostics["degeneracy"]
    visualizer.add_3d_label(
        anchor,
        "DCReg Parking-Lot\n"
        "intensity map | orange initial | blue DCReg\n"
        f"{important_axis_summary(diagnostics)}\n"
        f"mask rpy/xyz: {mask_text(diagnostics)} | "
        f"RMSE {diagnostics['registration']['rmse']:.3f}",
    )

    if verbose_labels:
        visualizer.add_3d_label(
            anchor + np.array([0.0, -7.0, 0.0]),
            "Contribution diag r/p/y: "
            f"{degeneracy['rot_axis_contribution_ratio'][0][0]:.2f}, "
            f"{degeneracy['rot_axis_contribution_ratio'][1][1]:.2f}, "
            f"{degeneracy['rot_axis_contribution_ratio'][2][2]:.2f}\n"
            "Contribution diag x/y/z: "
            f"{degeneracy['trans_axis_contribution_ratio'][0][0]:.2f}, "
            f"{degeneracy['trans_axis_contribution_ratio'][1][1]:.2f}, "
            f"{degeneracy['trans_axis_contribution_ratio'][2][2]:.2f}",
        )
        for position, text in axis_labels:
            visualizer.add_3d_label(position, text)
    # ===== END CHANGE: compact presentation labels =====


def show_labeled_viewer(
    o3d,
    geometries: list,
    axis_labels: list,
    diagnostics: dict,
    verbose_labels: bool,
    show_settings: bool,
    show_open3d_helpers: bool,
) -> None:
    from open3d.visualization import gui  # pylint: disable=import-outside-toplevel

    app = gui.Application.instance
    app.initialize()
    visualizer = o3d.visualization.O3DVisualizer(
        "DCReg parking-lot visualization", 1600, 950
    )
    # ===== BEGIN CHANGE: Open3D 0.19 viewer compatibility =====
    visualizer.show_settings = show_settings
    visualizer.show_axes = show_open3d_helpers
    visualizer.show_ground = show_open3d_helpers
    # Do not assign `show_skybox`: it is read-only in Open3D 0.19 wheels and
    # raises AttributeError before the interactive viewer opens.
    # ===== END CHANGE: Open3D 0.19 viewer compatibility =====
    visualizer.set_background(np.asarray([0.0, 0.0, 0.0, 1.0]), None)

    names = [
        "target_full_voxel_map_intensity",
        "source_initial_prediction_orange",
        "source_registered_dcreg_blue",
        "axis_translation_x",
        "axis_translation_y",
        "axis_translation_z",
        "axis_rotation_roll",
        "axis_rotation_pitch",
        "axis_rotation_yaw",
    ]
    for name, geometry in zip(names, geometries):
        visualizer.add_geometry(name, geometry)

    bounds = focus_bounds(o3d, geometries)
    center = bounds.get_center()
    # ===== BEGIN CHANGE: top-down viewer camera =====
    extent = np.linalg.norm(bounds.get_extent())
    if extent <= 1e-6:
        extent = 1.0
    eye = center + np.array([0.0, 0.0, 0.35 * extent])
    visualizer.setup_camera(60.0, center, eye, np.array([0.0, 1.0, 0.0]))
    # ===== END CHANGE: top-down viewer camera =====
    add_viewer_labels(
        visualizer, bounds, diagnostics, axis_labels, verbose_labels
    )
    app.add_window(visualizer)
    app.run()


def main() -> int:
    args = parse_args()
    if args.screenshot is None:
        args.screenshot = args.export_dir / DEFAULT_SCREENSHOT_NAME

    diagnostics = read_json(require_file(args.export_dir / "diagnostics.json"))
    print_notes(diagnostics)
    for filename in ("target_map.pcd", "source_initial.pcd",
                     "source_registered.pcd"):
        require_file(args.export_dir / filename)
    if args.dry_run:
        print(f"Dry run passed: {args.export_dir}")
        return 0

    o3d = load_open3d()

    target = load_intensity_cloud(o3d, args.export_dir / "target_map.pcd",
                                  (0.86, 0.86, 0.86))
    source_initial = load_intensity_cloud(
        o3d, args.export_dir / "source_initial.pcd", (1.0, 0.48, 0.05)
    )
    source_registered = load_intensity_cloud(
        o3d, args.export_dir / "source_registered.pcd", (0.05, 0.32, 1.0)
    )
    axis_geometry, axis_labels = make_axis_geometry(
        o3d, matrix_from_json(diagnostics["final_transform"]), diagnostics
    )
    geometries = [target, source_initial, source_registered] + axis_geometry

    if not args.no_screenshot:
        save_screenshot(
            o3d,
            geometries,
            args.point_size,
            args.screenshot,
            diagnostics,
            args.plain_screenshot,
        )
    if not args.no_viewer:
        show_labeled_viewer(
            o3d,
            geometries,
            axis_labels,
            diagnostics,
            args.verbose_labels,
            args.show_settings,
            args.show_open3d_helpers,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
