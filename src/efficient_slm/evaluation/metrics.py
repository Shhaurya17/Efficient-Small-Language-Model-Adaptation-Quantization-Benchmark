def calculate_delta_sft(base_score, sft_score):
    return sft_score - base_score


def calculate_delta_quant(sft_score, quant_score):
    return quant_score - sft_score


def calculate_efficiency(vram_gb, score):
    if not vram_gb:
        return None
    return score / vram_gb


def calculate_quality_per_param(trainable_params, score):
    if not trainable_params:
        return None
    return score / (trainable_params / 1e6)
