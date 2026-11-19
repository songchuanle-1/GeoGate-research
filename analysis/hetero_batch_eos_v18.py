"""Pure EOS-trim helpers for heterogeneous batching validation."""


def trim_after_eos(output_ids, eos_token_id):
    if eos_token_id is None:
        return list(output_ids)
    try:
        end = output_ids.index(eos_token_id)
    except ValueError:
        return list(output_ids)
    return list(output_ids[:end + 1])


def raw_mismatch_is_eos_padding_only(
    batch_output_ids,
    individual_output_ids,
    eos_token_id,
    pad_token_id,
):
    if eos_token_id is None or pad_token_id is None:
        return False
    if trim_after_eos(batch_output_ids, eos_token_id) != trim_after_eos(
        individual_output_ids,
        eos_token_id,
    ):
        return False
    if len(batch_output_ids) < len(individual_output_ids):
        return False
    if (
        batch_output_ids[:len(individual_output_ids)]
        != individual_output_ids
    ):
        return False
    return all(
        token == pad_token_id
        for token in batch_output_ids[len(individual_output_ids):]
    )
