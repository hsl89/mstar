"""
Used to calculate data drop coefficients
Given a drop consisting of multiple arrow files, calculate the
per-file weighting of that drop to achieve the target ratio
"""
import hydra
import numpy as np
import datasets


def calculate_within_drop_ratios(example_counts: dict) -> dict:
    """
    Parameters
    ----------
    example_counts : dict
        A {filepath: example_count} mapping containing the within-drop example counts

    Returns
    -------
    dict :
        A {filename: ratio} dict where ratio is the ratio of examples in
        that file divided by the total number of examples of all files in the list
    """
    total_examples = sum(example_counts.values())

    per_file_example_ratios = {x: y / total_examples for x, y in example_counts.items()}

    assert np.allclose(
        sum((per_file_example_ratios[key] for key in per_file_example_ratios.keys())),
        1.0,
    )

    return per_file_example_ratios


def compute_per_file_ratios_across_drops(
    within_drop_ratios: dict, target_per_drop_ratios: dict
) -> dict:
    """
    Parameters
    ----------
    within_drop_ratios: dict
        A {drop: {filename: within_drop_ratio}} mapping that covers all drops
        and all filenames within a drop
    target_per_drop_ratios: dict
        A {drop: target_ratio} mapping that specifies the target ratio of each drop

    Returns
    -------
    dict :
        A {filename: ratio} dict where ratio is the mixing ratio of each file necessary to
        achieve the desired target ratio across drops with equal sampling within each drop
    """
    per_file_ratios = {}
    for drop in within_drop_ratios:
        per_file_ratios[drop]={}
        for filename in within_drop_ratios[drop]:
            per_file_ratios[drop][filename] = (
                target_per_drop_ratios[drop] * within_drop_ratios[drop][filename]
            )

    return per_file_ratios


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="sampled_byte_drop_4_stage_2_t5.yaml",
)
def main(cfg):
    """
    Calculate per-file ratios that are needed to achieve the desired per-drop ratio
    """

    # checksum on target ratios, should add to 1
    checksum = sum((getattr(cfg, drop_name).target_ratio for drop_name in cfg))
    assert np.allclose(
        checksum, 1.0
    ), f"Checksum {checksum} on target ratios should be 1.0"

    # calculate the number of examples in each file
    # assumes we load from arrow files
    per_file_example_counts = {}
    for drop in cfg:
        per_file_example_counts[drop] = {
            file_name: len(datasets.arrow_dataset.Dataset.from_file(file_name))
            for file_name in cfg[drop].filepaths
        }

    # calculate within-drop ratios for each drop
    within_drop_ratios = {
        drop: calculate_within_drop_ratios(per_file_example_counts[drop])
        for drop in cfg
    }

    # use within-drop ratio and the target ratios to compute per-file ratios
    target_per_drop_ratios = {drop: cfg[drop].target_ratio for drop in cfg}

    per_file_ratios = compute_per_file_ratios_across_drops(
        within_drop_ratios=within_drop_ratios,
        target_per_drop_ratios=target_per_drop_ratios,
    )

    checksum = sum((per_file_ratios[drop][filename] for drop in per_file_ratios for filename in per_file_ratios[drop]))
    assert np.allclose(
        checksum, 1.0
    ), f"Checksum {checksum} on per-file ratios should be 1.0"

    print("Per file ratios \n")

    as_list = []    
    for drop in per_file_ratios:
        for filename in per_file_ratios[drop]:
            val = per_file_ratios[drop][filename]
            as_list.append(val)
            print(filename, val)

    # also useful to print in list format for copy-paste into config
    print(as_list)

if __name__ == "__main__":
    main()
