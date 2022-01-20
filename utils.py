

class FuncNamingExample(object):
    def __init__(self, idx, non_leaf_tokens, masked_leaf_tokens, ud_pos, dfg_to_dfg, dfg_to_code):
        self.idx = idx
        self.non_leaf_tokens = non_leaf_tokens
        self.masked_leaf_tokens = masked_leaf_tokens
        self.ud_pos = ud_pos
        self.dfg_to_dfg = dfg_to_dfg
        self.dfg_to_code = dfg_to_code


class FuncNamingFeature(object):
    def __init__(self, example_id, source_ids, position_idx, rel_pos, source_mask, target_ids, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.rel_pos = rel_pos
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask


