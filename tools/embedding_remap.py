"""
Tool for shuffling and reinitializing embedding vectors for vocab changes.

Usage:
SOURCE_INDICES="$(seq -s' ' 32099 -1 32000) $(seq -s' ' 32100 33919)"
TARGET_INDICES="$(seq -s' ' 34175 -1 32256)"
REINIT_INDICES="$(seq -s' ' 32000 32255)"
python3 embedding_remap.py --file pytorch_model.bin --output_file patched_pytorch_model.bin \
        --embedding_keys lm_head.weight shared.weight encoder.embed_tokens.weight decoder.embed_tokens.weight \
        --source_indices $SOURCE_INDICES --target_indices $TARGET_INDICES --reinit_indices $REINIT_INDICES
"""

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Tool for shuffling and reinitializing embedding vectors'
                    'for vocab changes. It copies from source indices to '
                    'target indices first, and then use the mean embedding'
                    'vector plus a normal distribution noise to reinitialize'
                    ' the vectors.')
    parser.add_argument(
        '--file', required=True,
        help='Path to the Pytorch .bin file.')
    parser.add_argument(
        '--output_file', required=True,
        help='Path for saving the patched Pytorch .bin file.')
    parser.add_argument(
        '--embedding_keys', required=True, nargs='+', type=str,
        help='Dict key for accessing the embedding weights.')
    parser.add_argument(
        '--source_indices', nargs='+', type=int, default=[],
        help='Indices to copy from.')
    parser.add_argument(
        '--target_indices', nargs='+', type=int, default=[],
        help='Indices to copy to.')
    parser.add_argument(
        '--reinit_indices', nargs='+', type=int, default=[],
        help='Indices to reinitialize.')
    parser.add_argument(
        '--std', type=float, default=1e-2,
        help='Indices to reinitialize.')
    args = parser.parse_args()
    return args


def main(args):
    embedding_keys = args.embedding_keys
    source_indices = args.source_indices
    target_indices = args.target_indices
    reinit_indices = args.reinit_indices
    state_dict = torch.load(args.file, map_location='cpu')
    std = args.std
    output_file = args.output_file

    for embedding_key in embedding_keys:
        embedding_weight = state_dict[embedding_key].data.clone()
        vocab_size, emb_dim = embedding_weight.shape

        # check indices
        # indices in range
        assert all(0 <= idx < vocab_size for idx in source_indices), [
            idx for idx in source_indices if idx < 0 or idx >= vocab_size]
        assert all(0 <= idx < vocab_size for idx in target_indices), [
            idx for idx in target_indices if idx < 0 or idx >= vocab_size]
        assert all(0 <= idx < vocab_size for idx in reinit_indices), [
            idx for idx in reinit_indices if idx < 0 or idx >= vocab_size]
        # source and target indices match in length
        assert len(source_indices) == len(target_indices)

        # for each embedding weight, get mean vector
        mean_vec = embedding_weight.data.mean(dim=0).clone()

        # remap embedding indices
        if source_indices:
            print(
                f'Remapping {len(source_indices)} vectors in {embedding_key}')
            embedding_weight[torch.tensor(target_indices)] = \
                embedding_weight[torch.tensor(source_indices)]

        # reinitialize indicies if any
        if reinit_indices:
            print(
                f'Reinitializing {len(reinit_indices)} vectors in '
                '{embedding_key}')
            noise = torch.empty(len(reinit_indices), emb_dim)
            embedding_weight[torch.tensor(reinit_indices)] = \
                torch.nn.init.normal_(noise, std=std) + mean_vec

        state_dict[embedding_key] = embedding_weight

    print(f'Writing patched weights to {output_file}')
    torch.save(state_dict, output_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
