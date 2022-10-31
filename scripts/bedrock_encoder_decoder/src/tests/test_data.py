from data.calculate_coefficients import calculate_within_drop_ratios, compute_per_file_ratios_across_drops 
import pytest

def test_calculate_data_coefficients():

    example_counts = {
        "drop_1": {"file_1":10, "file_2":10},
        "drop_2": {"file_1":5, "file_2":15}
    }
    
    expected_within_drop_ratios = {
        "drop_1": {"file_1":0.5, "file_2":0.5},
        "drop_2": {"file_1":0.25, "file_2":.75}
    }
    
    target_ratios = {"drop_1":0.25, "drop_2":0.75}

    expected_per_file_ratios = {
        "drop_1": {"file_1":0.5*0.25, "file_2":0.5*0.25},
        "drop_2": {"file_1":0.25*0.75, "file_2":.75*0.75}
    }

    within_drop_ratios = {drop:calculate_within_drop_ratios(example_counts[drop]) for drop in example_counts}
    
    assert within_drop_ratios==expected_within_drop_ratios

    per_file_ratios = compute_per_file_ratios_across_drops(within_drop_ratios=within_drop_ratios,target_per_drop_ratios=target_ratios) 
    assert per_file_ratios==expected_per_file_ratios
