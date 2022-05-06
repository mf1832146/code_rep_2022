

# 生成关系矩阵
import argparse
import multiprocessing
from functools import partial
import time
import numpy as np
import pickle
import h5py

from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from _utils import ast2seq, connect_db, get_ud2pos, clean_ast, remove_node, sub_tree
from config import add_args
from utils import get_func_naming_feature, FuncNamingDataset

ud2pos = get_ud2pos(64)


tokenizer = Tokenizer.from_file('tokenizer/deberta_tokenizer.json')
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                    unk_token='<unk>',
                                    bos_token="<s>",
                                    eos_token="</s>",
                                    cls_token='<s>',
                                    sep_token='</s>',
                                    pad_token='<pad>',
                                    mask_token='<mask>')
db = connect_db().codes


def deal_with_ast(item):
    ast = pickle.loads(item['ast'])
    dfg = pickle.loads(item['dfg'])
    index2code = pickle.loads(item['index_to_code'])

    clean_ast(ast)
    sub_tree(ast, 256)  # args.max_ast_len
    remove_node(ast, index2code)

    func_name = item['func_name']
    if func_name == '':
        return None

    if '.' in func_name:
        func_name = func_name.split('.')[-1]

    _id = item['_id']

    non_leaf_tokens, leaf_tokens, ud_pos = ast2seq(ast, index2code, ud2pos)
    # leaf tokens 分词
    split_leaf_tokens = []
    masked_leaf_tokens = []
    for idx, x in enumerate(leaf_tokens):
        if idx != 0:
            origin_tokens = tokenizer.tokenize('@ ' + x)[1:]
            if x == func_name:
                masked_func_tokens = ['<mask>'] * len(origin_tokens)  # [mask_func_tokens]
                masked_leaf_tokens.append(masked_func_tokens)
            else:
                masked_leaf_tokens.append(origin_tokens)
            split_leaf_tokens.append(origin_tokens)
        else:
            split_leaf_tokens.append(tokenizer.tokenize(x))
            masked_leaf_tokens.append(tokenizer.tokenize(x))

    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(split_leaf_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(split_leaf_tokens[i]))
    split_leaf_tokens = [y for x in split_leaf_tokens for y in x]
    masked_leaf_tokens = [y for x in masked_leaf_tokens for y in x]

    non_leaf_len = len(non_leaf_tokens)
    leaf_len = len(leaf_tokens)
    s_leaf_len = len(split_leaf_tokens)

    split_ud_pos = np.zeros((non_leaf_len + s_leaf_len, non_leaf_len + s_leaf_len), dtype=np.float64)

    # 叶节点拆分映射
    for i in range(non_leaf_len + leaf_len):
        for j in range(non_leaf_len + leaf_len):
            if i >= non_leaf_len:
                a_i, b_i = ori2cur_pos[i-non_leaf_len]
                a_i += non_leaf_len
                b_i += non_leaf_len
            else:
                a_i, b_i = i, i+1
            if j >= non_leaf_len:
                a_j, b_j = ori2cur_pos[j-non_leaf_len]
                a_j += non_leaf_len
                b_j += non_leaf_len
            else:
                a_j, b_j = j, j+1

            split_ud_pos[a_i: b_i, a_j: b_j] = ud_pos[i][j]

    dfg = [d for d in dfg if d[1] in ori2cur_pos]

    # reindex 记录code token idx 到 dfg index
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    # 将dfg的指向节点位置映射为dfg的位置
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    # 记录dfg对应的code第几个到第几个index
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]

    # 写入数据库
    r = {
        'code_index': item['code_index'],
        'func_name': item['func_name'],
        'non_leaf_tokens': non_leaf_tokens,
        'leaf_tokens': split_leaf_tokens,
        'masked_leaf_tokens': masked_leaf_tokens,
        'split_ud_pos': split_ud_pos,
        'dfg_to_dfg': dfg_to_dfg,
        'dfg_to_code': dfg_to_code,
    }
    return r


def save_h5(file_name, data, times=0):
    chunk_size = 1000
    if times == 0:
        h5f = h5py.File(file_name, 'w')
        dataset = h5f.create_dataset("up_pos", (chunk_size, 320, 320),
                                     maxshape=(None, 320, 320),
                                     dtype='float32')
    else:
        h5f = h5py.File(file_name, 'a')
        dataset = h5f['up_pos']

    dataset.resize([(times + 1) * chunk_size, 320, 320])

    dataset[times * chunk_size: times * chunk_size + data.shape[0]] = data
    h5f.close()


def load_h5(file_name):
    h5f = h5py.File(file_name, 'r')
    data = h5f['up_pos']
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    print('generate rel pos for lang : ', args.lang)

    split_tags = ['train', 'test', 'valid']

    time_costs = [[], [], []]
    for i, split_tag in enumerate(split_tags):
        conditions = {'partition': split_tag, 'lang': args.lang, 'build_ast': 1}
        return_items = {'code_index': 1, 'ast': 1, 'func_name': 1, 'dfg': 1, 'index_to_code': 1}
        results = connect_db().codes.find(conditions, return_items)

        start_time = time.time()
        processed_results = []

        for result in tqdm(results, total=results.count(), desc='generate rel pos'):
            processed_result = deal_with_ast(result)
            if processed_result is not None:
                processed_results.append(processed_result)

        ud_pos_gen_time = time.time()

        examples = []
        rel_pos_list = []

        cache_fn = './cache_file/{}_examples.pkl'.format(split_tag)
        cache_rel_pos = './cache_file/{}_rel_pos.h5'.format(split_tag)

        times = 0

        for j, result in tqdm(enumerate(processed_results), total=len(processed_results),
                              desc='convert to features.'):
            example = get_func_naming_feature(result, tokenizer, args)
            rel_pos = example.rel_pos
            example.rel_pos = None  # 减少存储
            examples.append(example)

            rel_pos_list.append(rel_pos)
            if len(rel_pos_list) == 1000 or j == len(processed_results) - 1:
                save_h5(cache_rel_pos, np.array(rel_pos_list, dtype=float), times=times)
                times += 1
                rel_pos_list = []

        pickle.dump(examples, open(cache_fn, 'wb'))

        end_time = time.time()

        time_costs[i].append(ud_pos_gen_time - start_time)
        time_costs[i].append(end_time - ud_pos_gen_time)
        time_costs[i].append(end_time - start_time)

    print(time_costs)








