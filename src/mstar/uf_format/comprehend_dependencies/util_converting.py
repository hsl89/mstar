def tags_to_iobes(tags):
    if not tags:
        return

    if len(tags) == 1:
        tags[0] = 'S-' + tags[0]
        return

    tags[0] = 'B-' + tags[0]
    tags[-1] = 'E-' + tags[-1]
    for i, tag in enumerate(tags[1:-1]):
        tags[i+1] = 'I-' + tag

def convert_encoding(tags, src_encoding, tgt_encoding):
    '''
    Convert the tags encoding from sr_encoding to tgt_encoding.

    :param tags list of tags in src_encoding 
    :param src_encoding can only be "iobes" or "iob" or "none"
    :param tgt_encoding can only be "iobes" or "iob"

    return new list of tags in tgt_encoding
    '''
    if src_encoding == tgt_encoding:
        return tags

    if src_encoding == "iobes" and tgt_encoding == "iob":
        new_tags = []
        for tag in tags:
            if tag == "O":
                new_tags.append(tag)
                continue

            tag_splits = tag.split("-", maxsplit=1)
            prefix = tag_splits[0]
            ent_type = tag_splits[1]
            if prefix == "S":
                new_tags.append("{}-{}".format("B", ent_type))
            elif prefix == "E":
                new_tags.append("{}-{}".format("I", ent_type))
            else:
                new_tags.append(tag)

        return new_tags

    elif src_encoding == "iob" and tgt_encoding == "iobes":
        new_tags = []
        for tag_index, tag in enumerate(tags):
            is_not_end = ((tag_index + 1 < len(tags)) and
                          ((tags[tag_index + 1][0] == "I") and
                           (tags[tag_index + 1][1:] == tag[1:])))
            if tag == "O":
                new_tags.append(tag)
            elif tag[0] == "B":
                # check whether it is singleton
                if is_not_end:
                    new_tags.append(tag)
                else:
                    new_tags.append("S" + tag[1:])
            elif tag[0] == "I":
                # check whether it"s at the beginning
                if ((tag_index == 0) or (tags[tag_index - 1] == "O")
                        or (tags[tag_index - 1][1:] != tag[1:])):
                    # check whether it is singleton
                    if is_not_end:
                        new_tags.append("B" + tag[1:])
                    else:
                        new_tags.append("S" + tag[1:])
                elif (tags[tag_index - 1][0] == "B") or \
                        (tags[tag_index - 1][0] == "I"):
                    # check whether it"s at the end
                    if is_not_end:
                        new_tags.append("I" + tag[1:])
                    else:
                        new_tags.append("E" + tag[1:])
                else:
                    raise Exception("Case not covered exists: %s" % str(tags))
            else:
                raise Exception("Case not covered exists: %s" % str(tags))

        return new_tags
    elif src_encoding == "none" and tgt_encoding == "iobes":
        # This is the case where no IOB tag is available for each token, which means we only have
        # token level labels. In this case we will merge continous tags that have the same type into one 
        # label span
        new_tags = []
        cur_tag_span = []
        for tag in tags:
            if tag != 'O':
                if cur_tag_span and cur_tag_span[-1] != tag:
                    tags_to_iobes(cur_tag_span)
                    new_tags += cur_tag_span
                    cur_tag_span = []

                cur_tag_span.append(tag)

            else:
                tags_to_iobes(cur_tag_span)
                new_tags += cur_tag_span
                cur_tag_span = []
                new_tags.append('O')

        tags_to_iobes(cur_tag_span)
        new_tags += cur_tag_span

        assert len(new_tags) == len(tags)
        return new_tags

    else:
        raise Exception("Current encoding converting not supported!")
