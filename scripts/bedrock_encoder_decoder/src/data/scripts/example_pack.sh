#!/bin/bash
INPUT_FILES=('/mnt/pile_mnt/pile/val.arrow' '/mnt/pile_mnt/pile/test.arrow' '/mnt/pile_mnt/pile/train.arrow')

EXAMPLE_PACK_LENGTH=2600


for file in ${INPUT_FILES[@]};
do python example_packing.py --data_file $file --example_pack_length $EXAMPLE_PACK_LENGTH;
done

