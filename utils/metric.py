

from medpy.metric.binary import dc, hd95, sensitivity, specificity


def calculate_metrics_with_debug(pred_mask, true_mask, region_name, inf_value=373.13):
    """
    Calculate metrics using medpy with debugging information
    """
    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()

    # First check for empty masks and set initial values
    if pred_sum == 0 and true_sum == 0:
        print(f"\nDebug - {region_name}: Both masks empty - adding score of 1.0 to results")
        return 1.0, 0.0, 1.0, 1.0  # Perfect scores for all metrics
    elif pred_sum == 0 and true_sum > 0:
        print(f"Debug - {region_name}: Prediction mask empty (truth has content) - HD95 set to {inf_value}")
        print(f"Prediction voxels: {pred_sum}, Ground truth voxels: {true_sum}")
        return 0.0, inf_value, 0.0, 0.0  # Zero scores and max HD95
    elif true_sum == 0 and pred_sum > 0:
        print(f"Debug - {region_name}: Ground truth mask empty (pred has content) - HD95 set to {inf_value}")
        print(f"Prediction voxels: {pred_sum}, Ground truth voxels: {true_sum}")
        return 0.0, inf_value, 0.0, 0.0  # Zero scores and max HD95

    # Calculate metrics using medpy
    dice = dc(pred_mask, true_mask)
    sens = sensitivity(pred_mask, true_mask)
    spec = specificity(pred_mask, true_mask)

    # HD95 calculation with debugging
    try:
        hausdorff = hd95(pred_mask, true_mask)
        if hausdorff == float('inf'):
            hausdorff = inf_value
            print(f"Debug - {region_name}: HD95 calculation returned inf - setting to {inf_value}")
            print(f"Prediction voxels: {pred_sum}, Ground truth voxels: {true_sum}")
    except Exception as e:
        hausdorff = inf_value
        print(f"Debug - {region_name}: HD95 calculation error: {str(e)}")
        print(f"Prediction voxels: {pred_sum}, Ground truth voxels: {true_sum}")

    return dice, hausdorff, sens, spec