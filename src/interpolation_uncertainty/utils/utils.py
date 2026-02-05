from PIL.ImageChops import offset

from interpolation_uncertainty.readers.bathymetryDataset import RasterDataset
from interpolation_uncertainty.processors.rasterProcessor import RasterProcessor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.cbook as cbook


def case_1_bp(bathy_data: RasterDataset,
              processor:RasterProcessor,
              method:str) -> cbook.boxplot_stats:
    """
    Compute boxplot of the difference between residuals and uncertainty_estimate
    Case 1 is default, across-track, computing uncertainty estimate using same row
    """

    # Compute uncertainty_estimate from using parameters inside Rasterprocessor
    residual_data = processor.compute_residual_surface(bathy_data)
    uncertainty = processor.estimate_uncertainty(method=method, residual_data=residual_data)

    # Make dimensions of residual and uncertainty the same
    if isinstance(uncertainty, RasterDataset):
        uncertainty = uncertainty[:, :residual_data.shape[1]]
        new_number_cols = uncertainty.shape[1]
        residual = residual_data[:, :new_number_cols]
        
        # Cast to numpy
        residual = np.array(residual)
        uncertainty = np.array(uncertainty)
        
        case_1 = cbook.boxplot_stats(np.abs(residual.ravel()-uncertainty.ravel()), labels=['Case 1'])[0]
        return case_1
        
    elif isinstance(uncertainty, dict):
        for key in uncertainty.keys():
            uncertainty[key] = np.array(uncertainty[key][:, :residual_data.shape[1]])
        new_number_cols = uncertainty[list(uncertainty.keys())[0]].shape[1]
        residual = np.array(residual_data[:, :new_number_cols])
        case_1_list = []
        for key in uncertainty.keys():
            case_1 = cbook.boxplot_stats(np.abs(residual.ravel()-uncertainty[key].ravel()), labels=[f'Case 1 {key}'])[0]
            case_1_list.append(case_1)
        return case_1_list
            
    else:
        print(f"uncertainty type: {type(uncertainty)}")
        raise TypeError("Uncertainty must be either RasterDataset or dict of RasterDataset")


    


def case_2_bp(bathy_data: RasterDataset,
              processor:RasterProcessor,
              method:str,
              base_row:str = 'top') -> cbook.boxplot_stats:
    """
    Compute boxplot of the difference between residuals and uncertainty_estimate
    Case 2 uses across-track data, simulates tiled uncertainty

    """
    linespacing_pixels = int(processor.linespacing_meters / bathy_data.metadata['resolution'])

    # Compute residual using complete coverage, across track data
    residual_data = processor.compute_residual_surface(bathy_data)

    # Subsample resulting residual every n-th linespacing
    if base_row == 'top':
        # Use top row as base and retain every linespacing_row
        row_subsampled_residual = residual_data[::linespacing_pixels, :]
    elif base_row == 'middle':
        # Use middle row as base and retain every linespacing_row
        offset = linespacing_pixels // 2
        row_subsampled_residual = residual_data[offset::linespacing_pixels, :]
    else:
        raise ValueError(f"base_row must be 'top' or 'middle'. input:{base_row}")

    # Expand subsampled residual and use it to compute for the uncertainty estimate
    expanded_residual = np.repeat(row_subsampled_residual, linespacing_pixels, axis=0)
    new_residual = expanded_residual[:residual_data.shape[0],:]
    
    # Compute uncertainty from the expanded residual
    uncertainty = processor.estimate_uncertainty(method=method, residual_data=new_residual)

    if isinstance(uncertainty, RasterDataset):
        # Make dimensions of original residual and computed uncertainty the same
        new_number_cols = uncertainty.shape[1]
        residual = residual_data[:, :new_number_cols]

        # Cast to numpy
        residual = np.array(residual)
        uncertainty = np.array(uncertainty)

        case_2 = cbook.boxplot_stats(np.abs(residual.ravel() - uncertainty.ravel()), labels=[f"Case 2 {base_row}"])[0]
        return case_2
    elif isinstance(uncertainty, dict):
        case_2_list = []
        for key in uncertainty.keys():
            unc = uncertainty[key]
            # Make dimensions of original residual and computed uncertainty the same
            new_number_cols = unc.shape[1]
            residual = residual_data[:, :new_number_cols]

            # Cast to numpy
            residual = np.array(residual)
            unc = np.array(unc)

            case_2 = cbook.boxplot_stats(np.abs(residual.ravel() - unc.ravel()), labels=[f"Case 2 {base_row} {key}"])[0]
            case_2_list.append(case_2)
        return case_2_list
    else:
        print(f"uncertainty type: {type(uncertainty)}")
        raise TypeError("Uncertainty must be either RasterDataset or dict of RasterDataset")


def case_3_bp(bathy_data: RasterDataset,
              processor:RasterProcessor,
              method:str,
              selected_column:str = 'left') -> cbook.boxplot_stats:
    """ Case 3 uses the along track data to estimate the
    across track uncertainty from the complete coverage data
    """

    # Compute normal residual using complete coverage, across track data
    residual_data = processor.compute_residual_surface(bathy_data)
    residual_data_as_strip = processor.matrix2strip(residual_data)


    # Crop original complete coverage data using only rows used in computing the residual
    bathy_data_cropped = bathy_data[:, :residual_data.shape[1]]

    # Update the "original size" metadata to reflect new size
    bathy_data_cropped.orig_shape = bathy_data_cropped.shape

    # Reshape original complete coverage data and extract data from preferred column
    bathy_data_strip = processor.matrix2strip(bathy_data_cropped)
    if selected_column == 'left':
        bathy_column = bathy_data_strip[:, 0]
    elif selected_column == 'right':
        bathy_column = bathy_data_strip[:, -1]
    else:
        raise ValueError(f"base_column must be 'left' or 'right'. input:{selected_column}")


    uncertainty = uncertainty_from_column(bathy_column, processor, method)
    if isinstance(uncertainty, RasterDataset):
        # if base column is right, we need to flip the surface to ensure first element of the column
        # would correspond to the point nearest to the column
        if selected_column == 'right':
            uncertainty = np.flip(uncertainty, axis=-1)
            uncertainty = bathy_data.wrap(uncertainty)

        # Make dimensions of residual_data and uncertainty the same to enable comparison
        linespacing_pixels = int(processor.linespacing_meters / bathy_data.metadata['resolution'])
        offset = linespacing_pixels // 2
        residual = residual_data_as_strip[offset + 1:-offset, :]

        # Cast to numpy
        residual = np.array(residual)
        uncertainty = np.array(uncertainty)

        # Compare uncertainty estimate with the original residual
        case_3 = cbook.boxplot_stats(np.abs(residual.ravel() - uncertainty.ravel()), labels=[f"Case 3 {selected_column}"])[0]
        return case_3
    elif isinstance(uncertainty, dict):
        case_3_list = []
        for key in uncertainty.keys():
            unc = uncertainty[key]
            # if base column is right, we need to flip the surface to ensure first element of the column
            # would correspond to the point nearest to the column
            if selected_column == 'right':
                unc = np.flip(unc, axis=-1)
                unc = bathy_data.wrap(unc)

            # Make dimensions of residual_data and uncertainty the same to enable comparison
            linespacing_pixels = int(processor.linespacing_meters / bathy_data.metadata['resolution'])
            offset = linespacing_pixels // 2
            residual = residual_data_as_strip[offset + 1:-offset, :]

            # Cast to numpy
            residual = np.array(residual)
            unc = np.array(unc)

            # Compare uncertainty estimate with the original residual
            case_3 = cbook.boxplot_stats(np.abs(residual.ravel() - unc.ravel()), labels=[f"Case 3 {selected_column} {key}"])[0]
            case_3_list.append(case_3)
        return case_3_list
    else:
        print(f"uncertainty type: {type(uncertainty)}")
        raise TypeError("Uncertainty must be either RasterDataset or dict of RasterDataset")

def case_4_bp(bathy_data: RasterDataset,
              processor:RasterProcessor,
              method:str,
              bias:float = 0.5) -> cbook.boxplot_stats:

    # limit possible values of bias for the meantime
    assert(bias in [0, 0.25, 0.5, 0.75, 1])

    # Compute normal residual using complete coverage, across track data
    residual_data = processor.compute_residual_surface(bathy_data)
    residual_data_as_strip = processor.matrix2strip(residual_data)
    
    # Make dimensions of residual_data and uncertainty the same to enable comparison
    linespacing_pixels = int(processor.linespacing_meters / bathy_data.metadata['resolution'])
    offset = linespacing_pixels // 2
    residual = residual_data_as_strip[offset + 1:-offset, :]

    # Crop original complete coverage data using only rows used in computing the residual
    bathy_data_cropped = bathy_data[:, :residual_data.shape[1]]

    # Update the "original size" metadata to reflect new size
    bathy_data_cropped.orig_shape = bathy_data_cropped.shape

    # Reshape original complete coverage data and extract data from preferred column
    bathy_data_strip = processor.matrix2strip(bathy_data_cropped)
    left_column = bathy_data_strip[:, 0]
    right_column = bathy_data_strip[:, -1]

    left_uncertainty = uncertainty_from_column(left_column, processor, method)
    right_uncertainty = uncertainty_from_column(right_column, processor, method)
    if isinstance(left_uncertainty, RasterDataset):
        right_uncertainty = np.flip(right_uncertainty, axis=-1)
        right_uncertainty = bathy_data.wrap(right_uncertainty)


        # Using the bias, determine how much of each left and right surface views
        # will be used to compose the proxy surface
        num_cols = int(residual.shape[1] * bias)
        surface_composite_strip = np.concatenate((left_uncertainty[:,:num_cols],
                                        right_uncertainty[:,-num_cols:]), axis=1)
        assert(surface_composite_strip.shape[1] == left_uncertainty.shape[1])
        surface_composite_strip = bathy_data.wrap(surface_composite_strip)
        composite_residual = processor.compute_residual(surface_composite_strip)

        # Compute corresponding uncertainties from the surface views
        uncertainty = processor.estimate_uncertainty(method=method, residual_data=composite_residual, is_strip=True)

        # Cast to numpy
        residual = np.array(residual)
        uncertainty = np.array(uncertainty)

        case_4 = cbook.boxplot_stats(np.abs(residual.ravel() - uncertainty.ravel()), labels=[f"Case 4 bias: {bias}"])[0]
        return case_4
    elif isinstance(left_uncertainty, dict):
        case_4_list = []
        for key in left_uncertainty.keys():
            left_unc = left_uncertainty[key]
            right_unc = right_uncertainty[key]
            right_unc = np.flip(right_unc, axis=-1)
            right_unc = bathy_data.wrap(right_unc)

            # Using the bias, determine how much of each left and right surface views
            # will be used to compose the proxy surface
            num_cols = int(residual.shape[1] * bias)
            surface_composite_strip = np.concatenate((left_unc[:,:num_cols], right_unc[:,-num_cols:]), axis=1)
            assert(surface_composite_strip.shape[1] == left_uncertainty[key].shape[1])
            surface_composite_strip = bathy_data.wrap(surface_composite_strip)
            composite_residual = processor.compute_residual(surface_composite_strip)

            # Compute corresponding uncertainties from the surface views
            uncertainty = processor.estimate_uncertainty(method=method, residual_data=composite_residual, is_strip=True)

            # Cast to numpy
            residual = np.array(residual)
            uncertainty = np.array(uncertainty)

            case_4 = cbook.boxplot_stats(np.abs(residual.ravel() - uncertainty.ravel()), labels=[f"Case 4 bias: {bias} {key}"])[0]
            case_4_list.append(case_4)
        return case_4_list
    
    else:
        print(f"uncertainty type: {type(left_uncertainty)}")
        raise TypeError("Uncertainty must be either RasterDataset or dict of RasterDataset")


def uncertainty_from_column(data_column: RasterDataset, processor:RasterProcessor, method:str):

    linespacing_pixels = int(processor.linespacing_meters / data_column.metadata['resolution'])

    # Create row views of the column where each view has dimension (1,linespacing)
    # sliding_window_view has width linespacing_pixels + 2 to accommodate edge columns
    column_as_strip = sliding_window_view(data_column, linespacing_pixels + 2)
    column_as_strip = data_column.wrap(column_as_strip)

    # Compute uncertainty from the surface view of the along-track column data
    column_residual_data = processor.compute_residual(column_as_strip)
    uncertainty = processor.estimate_uncertainty(method=method, residual_data=column_residual_data, is_strip=True)

    return uncertainty


def uncertainty_comparison(residuals, uncertainties):
    nonzero_idx = np.nonzero(
        (residuals != 0) & (~np.isnan(residuals)) & (uncertainties != 0)
    )
    uncertainty_ratio = np.full(residuals.shape, np.nan)
    uncertainty_ratio[nonzero_idx] = uncertainties[nonzero_idx] / np.abs(residuals[nonzero_idx])
    fail_points = np.nonzero(uncertainty_ratio < 1)
    ur_flat = uncertainty_ratio[nonzero_idx].flatten()
    total_count = len(ur_flat)
    fail_count = len(fail_points[0])
    pass_percentage = 100 - fail_count / total_count * 100
    current_rmse = np.sqrt(np.mean((residuals[nonzero_idx] - uncertainties[nonzero_idx]) ** 2))
    mean_error = np.mean(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
    std_dev = np.std(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
    sharp = np.mean(uncertainties[nonzero_idx])
    corr = np.corrcoef(uncertainties[nonzero_idx], np.abs(residuals[nonzero_idx]))[0, 1] if len(np.abs(residuals[nonzero_idx])) > 1 else np.nan

    return {
        "total_cts": total_count,
        "fail_cts": fail_count,
        "percentage": pass_percentage,
        "rmse": current_rmse,
        "mean": mean_error,
        "std_dev": std_dev,
        "sharp": sharp,
        "corr": corr}, uncertainties[nonzero_idx], np.abs(residuals[nonzero_idx])
    
    
def multi_uncertainty_comparison(
    residuals: np.ndarray,
    uncertainties_dict: dict[str, np.ndarray],
    resolution,
    desired_linespacing_meters=None,
    fn=None,
    plot_grid=(4, 3),
    path=None,
    plot_boxplots=True
):
    """
    Compare multiple uncertainty surfaces against residuals in one figure.

    Parameters
    ----------
    residuals : np.ndarray
        2D array of residual surface.
    uncertainties_dict : dict
        Dictionary of uncertainty name -> uncertainty array.
    resolution : float
        Grid resolution in meters.
    desired_linespacing_meters : float, optional
        Used for labeling titles.
    fn : str, optional
        Surface name for the first title.
    plot_grid : tuple
        (nrows, ncols) for subplot grid.
    """
    
    import matplotlib.pyplot as plt

    def uncertainty_comparison(residuals:np.ndarray, uncertainties:np.ndarray):


        nonzero_idx = np.nonzero(
            (residuals != 0) & (~np.isnan(residuals)) & (uncertainties != 0)
        )
        print(nonzero_idx)
        uncertainty_ratio = np.full(residuals.shape, np.nan)
        uncertainty_ratio[nonzero_idx] = uncertainties[nonzero_idx] / np.abs(residuals[nonzero_idx])
        fail_points = np.nonzero(uncertainty_ratio < 1)
        ur_flat = uncertainty_ratio[nonzero_idx].flatten()
        total_count = len(ur_flat)
        fail_count = len(fail_points[0])
        pass_percentage = 100 - fail_count / total_count * 100
        current_rmse = np.sqrt(np.mean((residuals[nonzero_idx] - uncertainties[nonzero_idx]) ** 2))
        mean_error = np.mean(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
        std_dev = np.std(uncertainties[nonzero_idx] - np.abs(residuals[nonzero_idx]))
        sharp = np.mean(uncertainties[nonzero_idx])
        corr = np.corrcoef(uncertainties[nonzero_idx], np.abs(residuals[nonzero_idx]))[0, 1] if len(np.abs(residuals[nonzero_idx])) > 1 else np.nan

        return pass_percentage, current_rmse, mean_error, std_dev, sharp, corr

    # ---- Create figure ----
    nrows, ncols = plot_grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12), layout="constrained")

    axes = axes.flatten()
    names = list(uncertainties_dict.keys())
    results = []

    for i, (name, uncertainty) in enumerate(uncertainties_dict.items()):
        ax = axes[i]

        # Compute stats
        pass_percentage, rmse, mean_error, std_dev, sharp, corr = uncertainty_comparison(residuals, uncertainty)

        # Append stats for CSV
        results.append({
            "Seabed": fn,
            "Uncertainty Method": name,
            "Line spacing ": desired_linespacing_meters,
            "Pass %": pass_percentage,
            "RMSE": rmse,
            "Bias (Mean Error)": mean_error,
            "Std Dev": std_dev,
            "Sharpness": sharp,
            "Correlation": corr
        })

        # Scatter comparison plot
        nonzero_idx = np.nonzero(
            (residuals != 0) & (~np.isnan(residuals)) & (uncertainty != 0)
        )
        max_unc = np.max(uncertainty[nonzero_idx])
        ax.plot(np.abs(residuals[nonzero_idx]), uncertainty[nonzero_idx], ".", alpha=0.3)
        ax.plot([0, max_unc], [0, max_unc], "r", lw=1)

        ax.set_xlabel("Abs. Residual (m)")
        ax.set_ylabel("Uncertainty (m)")
        ax.set_xlim(0, max_unc)
        ax.set_ylim(0, max_unc)
        ax.grid(True, alpha=0.3)

        # Title with stats
        ax.set_title(
            f"{name}\nPass: {pass_percentage:.1f}%  RMSE: {rmse:.2f}  Bias: {mean_error:.2f}\n  "
            f"Corr: {corr:.2f}, Sharp: {sharp:.2f}",  fontsize=12,
        )

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if fn and desired_linespacing_meters:
        fig.suptitle(
            f"Uncertainty Comparisons for {fn} ({resolution}m grid, {desired_linespacing_meters}m spacing)",
            fontsize=14,
        )
    else:
        fig.suptitle("Uncertainty Comparisons", fontsize=14)
    # outpath = f'{path}_uncertainty_comparisons.png'
    # plt.savefig(outpath, bbox_inches='tight')
    plt.show()


    # # ---- Export to CSV ----
    # if path:
    #     df = pd.DataFrame(results)
    #     outpath = f'{path}_stats.csv'
    #     df.to_csv(f'{outpath}', index=False)
    #     print(f"Statistics exported to {outpath}")

    # # ---- Optional: Combined Boxplot of Residuals vs Uncertainties ----

    # if plot_boxplots:
    #     # Collect depth
    #     depth = []
    #     labels = []

    #     # Residuals (absolute values)
    #     res_vals = np.abs(residuals[(residuals != 0) & (~np.isnan(residuals))]).flatten()
    #     depth.append(res_vals)
    #     labels.append("Abs.  Residuals")

    #     # Each uncertainty
    #     for name, uncertainty in uncertainties_dict.items():
    #         unc_vals = uncertainty[(uncertainty != 0) & (~np.isnan(uncertainty))].flatten()
    #         depth.append(unc_vals)
    #         labels.append(name)

    #     # Combined boxplot
    #     plt.figure(figsize=(10, 5))
    #     plt.boxplot(depth, patch_artist=True, labels=labels,
    #                 boxprops=dict(facecolor='lightgray', alpha=0.7),
    #                 medianprops=dict(color='red', linewidth=1.5))
    #     plt.title(f"Uncertainty Boxplots for {fn} ({resolution}m grid, {desired_linespacing_meters}m spacing)")
    #     plt.ylabel("Uncertainty (m)")
    #     plt.grid(alpha=0.3)
    #     plt.xticks(rotation=30)
    #     outpath = f'{path}_uncertainty_boxplots.png'
    #     plt.savefig(outpath, bbox_inches='tight')
    #     plt.show()

    # def make_square(raster_data: RasterBathymetry) -> RasterBathymetry:
    #     new_dim = np.min(raster_data.data.shape)
    #     raster_data.depth  = raster_data.depth[:new_dim, :new_dim]
    #     return raster_data