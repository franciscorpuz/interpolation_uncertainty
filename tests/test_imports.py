"""Verify that all public names are importable from the top-level package."""


def test_top_level_imports():
    from interpolation_uncertainty import (
        read_file,
        show_depth,
        build_coverage_mask,
        filter_blobs_keep_lines,
        filter_squares_keep_lines,
        filter_morphological_operations,
        compute_dominant_angle_radon_transform,
        compute_local_orientation,
        rotate_points,
        update_geotransform,
        refine_cluster_rotation,
        undo_rotation,
        x_distance,
        compute_uncertainty,
        detect_trackline_positions,
        assign_depths_to_lines,
        process_line_pair,
        build_uncertainty_raster,
    )
    assert callable(read_file)
    assert callable(compute_uncertainty)
    assert callable(build_uncertainty_raster)


def test_subpackage_imports():
    from interpolation_uncertainty.io import read_file
    from interpolation_uncertainty.visualization import show_depth
    from interpolation_uncertainty.preprocessing import build_coverage_mask
    from interpolation_uncertainty.methods import compute_uncertainty
    from interpolation_uncertainty.pipeline import detect_trackline_positions
    assert callable(read_file)
    assert callable(show_depth)
    assert callable(build_coverage_mask)
    assert callable(compute_uncertainty)
    assert callable(detect_trackline_positions)
