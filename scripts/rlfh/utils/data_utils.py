def compute_indices(world_size, current_rank, dataset_length):
    num_test_samples_per_rank = int(dataset_length / world_size)
    if current_rank < world_size-1:
        indices = list(range(num_test_samples_per_rank*current_rank, num_test_samples_per_rank*(current_rank+1)))
    else:
        indices = list(range(num_test_samples_per_rank*current_rank, dataset_length))
    return indices