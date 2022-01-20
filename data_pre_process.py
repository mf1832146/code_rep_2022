

# 生成关系矩阵
import multiprocessing

import numpy as np
import pickle

from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from _utils import ast2seq, connect_db, get_ud2pos


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


def deal_with_ast(item):
    ast = pickle.loads(item['ast'])
    dfg = pickle.loads(item['dfg'])
    index2code = pickle.loads(item['index_to_code'])
    func_name = item['func_name']
    if func_name == '':
        return
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
    db = connect_db().codes
    add_values = {
        'non_leaf_tokens': non_leaf_tokens,
        'leaf_tokens': split_leaf_tokens,
        'masked_leaf_tokens': masked_leaf_tokens,
        'split_ud_pos': pickle.dumps(split_ud_pos),
        'dfg_to_dfg': dfg_to_dfg,
        'dfg_to_code': dfg_to_code,
        'is_ok': 1
    }
    try:
        db.update_one({'_id': _id}, {'$set': add_values})
    except:
        db.update_one({'_id': _id}, {'$set':{'is_ok': 0}})


if __name__ == '__main__':
    pool = multiprocessing.Pool(10)
    conditions = {'lang': 'javascript', 'build_ast': 1}
    return_items = {'code_index': 1, 'ast': 1, 'func_name': 1, 'dfg': 1, 'index_to_code': 1}
    results = connect_db().codes.find(conditions, return_items)

    pool.map(deal_with_ast, tqdm(list(results), 'generate rel pos'))




