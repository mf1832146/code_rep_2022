import pickle
import os
from functools import partial
import time
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from _utils import build_relative_position, connect_db

logger = logging.getLogger(__name__)

seq_rel_pos = build_relative_position()


def load_and_cache_gen_data_from_db(args, pool, tokenizer, split_tag):
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + data_tag)

    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        logger.info("Create cache data into %s", cache_fn)

        return_items = {'_id': 1}
        conditions = {'partition': split_tag, 'lang': args.sub_task, 'is_ok': 1}

        results = connect_db().codes.find(conditions, return_items)
        examples = pool.map(partial(get_func_naming_feature, tokenizer=tokenizer, args=args),
                            tqdm(list(results), total=results.count()))
        data = FuncNamingDataset(examples)
        if args.local_rank in [-1, 0]:
            torch.save(data, cache_fn)
    return data


class FuncNamingDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        attn_mask = example.rel_pos > 0
        return (torch.tensor(example.source_ids),
                torch.tensor(example.source_mask),
                torch.tensor(example.position_idx),
                torch.tensor(attn_mask),
                torch.tensor(example.rel_pos),
                torch.tensor(example.target_ids),
                torch.tensor(example.target_mask),
                torch.tensor(example.gold_ids))


class FuncNamingFeature(object):
    def __init__(self, example_id, source_ids, position_idx, rel_pos, source_mask, target_ids, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.rel_pos = rel_pos
        self.source_mask = source_mask
        self.target_ids = target_ids
        self.target_mask = target_mask


def get_func_naming_feature(result, tokenizer, args):
    return_items = {'code_index': 1, 'non_leaf_tokens': 1, 'masked_leaf_tokens': 1,
                    'split_ud_pos': 1, 'dfg_to_dfg': 1, 'dfg_to_code': 1,
                    'func_name': 1}
    item = connect_db().codes.find({'_id': result['_id']}, return_items)[0]

    example_id = item['code_index']
    non_leaf_tokens = item['non_leaf_tokens']
    leaf_tokens = item['masked_leaf_tokens']
    ud_pos = pickle.loads(item['split_ud_pos'])
    dfg_to_dfg = item['dfg_to_dfg']
    dfg_to_code = item['dfg_to_code']
    func_name = item['func_name']

    if '.' in func_name:
        func_name = func_name.split('.')[-1]

    source_len = args.max_ast_len if args.use_ast else args.max_code_len

    if args.use_dfg:
        source_len += args.max_dfg_len

    rel_pos = np.zeros((source_len, source_len), dtype=np.long)

    """truncating"""
    if args.use_ast:
        origin_non_leaf_len = len(non_leaf_tokens)
        max_source_len = args.max_ast_len - 2
        if len(leaf_tokens) > max_source_len:
            leaf_tokens = leaf_tokens[:max_source_len]
            non_leaf_tokens = []
        else:
            non_leaf_tokens = non_leaf_tokens[:max_source_len - len(leaf_tokens)]
    else:
        max_source_len = args.max_code_len - 2
        leaf_tokens = leaf_tokens[:max_source_len]

    input_tokens = non_leaf_tokens + leaf_tokens if args.use_ast else leaf_tokens

    source_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    if args.use_ast:
        non_leaf_len = len(non_leaf_tokens)
        leaf_len = len(leaf_tokens)
        length = len([tokenizer.cls_token])
        # non leaf -> non leaf
        rel_pos[length:non_leaf_len+length, length:non_leaf_len+length] = ud_pos[:non_leaf_len, :non_leaf_len]
        # leaf -> non_leaf
        rel_pos[length+non_leaf_len:length+non_leaf_len+leaf_len, length:non_leaf_len+length] = \
            ud_pos[origin_non_leaf_len:origin_non_leaf_len+leaf_len,
                                      :non_leaf_len]
        # leaf -> leaf
        rel_pos[length+non_leaf_len:length+non_leaf_len+leaf_len, length+non_leaf_len:length+non_leaf_len+leaf_len] \
            = ud_pos[origin_non_leaf_len:origin_non_leaf_len+leaf_len, origin_non_leaf_len:origin_non_leaf_len+leaf_len]
        # non_leaf -> leaf
        rel_pos[length:non_leaf_len+length, length+non_leaf_len:length+non_leaf_len+leaf_len] \
            = ud_pos[:non_leaf_len, origin_non_leaf_len:origin_non_leaf_len+leaf_len]

    if args.use_code:
        leaf_token_len = len(leaf_tokens)
        rel_leaf_pos = seq_rel_pos[:leaf_token_len, :leaf_token_len]
        if args.use_ast:
            rel_leaf_pos += args.max_rel_pos  # 偏移
        length = len([tokenizer.cls_token]) + len(non_leaf_tokens)
        print(leaf_token_len)
        print(rel_leaf_pos.size())
        print(rel_pos.size())
        print(rel_pos[length:length+leaf_token_len, length:length+leaf_token_len].size())
        rel_pos[length:length+leaf_token_len, length:length+leaf_token_len] = rel_leaf_pos

    if args.use_dfg:
        source_len = len(source_ids)
        source_ids += [tokenizer.unk_token_id] * len(dfg_to_dfg)
        length = len([tokenizer.cls_token]) + len(non_leaf_tokens) if args.use_ast else len([tokenizer.cls_token])

        # dfg_to_code
        for i, (a, b) in enumerate(dfg_to_code):
            rel_pos[i+source_len, a+length:b+length] = 1  # 1 means self
            rel_pos[a+length:b+length, i+source_len] = 1

        # dfg to dfg
        for i, item in enumerate(dfg_to_dfg):
            for j in item:
                rel_pos[(i + source_len, j + source_len)] = 1

    # special tokens attend to all tokens
    end_s_pos = len([tokenizer.cls_token]) + len(leaf_tokens)
    if args.use_ast:
        end_s_pos += non_leaf_len
    rel_pos[0, :len(source_ids)] = 1
    rel_pos[end_s_pos, :len(source_ids)] = 1

    max_source_len = 0

    position_idx = [-3]  # start pos

    if args.use_ast:
        max_source_len += args.max_ast_len
    else:
        max_source_len += args.max_code_len
    if args.use_dfg:
        max_source_len += args.max_dfg_len

    padding_length = max_source_len - len(source_ids)

    if args.use_ast:
        position_idx += [-3] * len(non_leaf_tokens)
    if args.use_code:
        position_idx += [i + tokenizer.pad_token_id + 1 for i in range(len(leaf_tokens))]
    else:
        position_idx += [-2] * len(leaf_tokens)
    position_idx += [-3]
    if args.use_dfg:
        position_idx += [-1] * len(dfg_to_dfg)

    position_idx += [tokenizer.pad_token_id] * (max_source_len - len(position_idx))
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask = [1] * (len(source_tokens))
    source_mask += [0] * padding_length

    target_tokens = tokenizer.tokenize(func_name)[:args.max_target_length - 2]
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    target_mask = [1] * len(target_ids)
    padding_length = 7 - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    feature = FuncNamingFeature(example_id,
                                source_ids,
                                position_idx,
                                rel_pos,
                                source_mask,
                                target_ids,
                                target_mask)

    return feature


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)











