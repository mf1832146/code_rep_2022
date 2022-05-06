import math
import torch
import numpy as np
from pymongo import MongoClient


def build_relative_position(query_size=600, key_size=600, max_relative_positions=64):
    max_relative_positions = int(max_relative_positions / 2)
    q_ids = torch.arange(query_size, dtype=torch.long, device='cpu')
    k_ids = torch.arange(key_size, dtype=torch.long, device='cpu')
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    rel_pos_ids += max_relative_positions + 1
    rel_pos_ids = torch.clamp(rel_pos_ids, 1, 2 * max_relative_positions + 1)
    return rel_pos_ids.numpy()


def ast2seq(root_node, index2code, ud2pos):
    clean_ast(root_node)
    sub_tree(root_node, 512)  # args.max_ast_len
    remove_node(root_node, index2code)
    non_leaves, leaves = [], []
    id2pos = {}
    traverse_ast(root_node, non_leaves, leaves, id2pos)
    for k in id2pos.keys():
        if id2pos[k] < 0:
            id2pos[k] = len(non_leaves) - id2pos[k] - 1
    ud_mask = get_ud_masks(non_leaves + leaves, id2pos, 10000)
    leaf_tokens = [index2code[node['position_id']][1] for node in leaves]
    non_leaf_tokens = [node['type'] for node in non_leaves]
    ud_mask = clip_ud_mask(ud_mask, ud2pos, 64)  # args.max_rel_pos
    return non_leaf_tokens, leaf_tokens, ud_mask


def sub_tree(root_node, max_ast_len, i=0):
    root_node['id'] = i
    i = i + 1
    if i > max_ast_len:
        return -1
    else:
        for j, child in enumerate(root_node['children']):
            i = sub_tree(child, max_ast_len, i)
            if i == -1:
                root_node['children'] = root_node['children'][:j]
                return -2
            if i == -2:
                root_node['children'] = root_node['children'][:j + 1]
                return i
        return i


def remove_node(root_node, index2code):
    if len(root_node['children']) == 0:
        if root_node['position_id'] not in index2code.keys():
            root_node['is_remove'] = True
    else:
        for child in root_node['children']:
            remove_node(child, index2code)
        is_remove = True
        for child in root_node['children']:
            if 'is_remove' not in child:
                is_remove = False
        if is_remove:
            root_node['is_remove'] = True
        else:
            root_node['children'] = [child for child in root_node['children'] if 'is_remove' not in child]


def clean_ast(root_node):
    if root_node['type'] == 'string':
        root_node['children'] = []
    if root_node['type'] == 'comment':
        root_node['is_delete'] = True
        if len(root_node['children']) > 0:
            print('error')
    else:
        for child in root_node['children']:
            clean_ast(child)
        root_node['children'] = [child for child in root_node['children'] if 'is_delete' not in child]


def traverse_ast(root_node, non_leaf_nodes, leaf_nodes, id2pos):
    if len(root_node['children']) == 0:
        leaf_nodes.append(root_node)
        id2pos[root_node['id']] = - len(leaf_nodes)
    else:
        id2pos[root_node['id']] = len(non_leaf_nodes)
        non_leaf_nodes.append(root_node)
        for child in root_node['children']:
            traverse_ast(child, non_leaf_nodes, leaf_nodes, id2pos)


def get_ancestors(dp, id2pos):
    ancestors = {0: []}
    node2parent = {0: 0}
    levels = {0: 0}
    for i, node in enumerate(dp):
        if "children" in node:
            cur_level = levels[i]
            for child in node["children"]:
                child_pos = id2pos[child['id']]
                node2parent[child_pos] = i
                levels[child_pos] = cur_level + 1
        ancestors[i] = [i] + ancestors[node2parent[i]]
    return ancestors, levels


def get_ud2pos(max_rel_pos):
    ud2pos = {'<self>': 1}
    max_u_op = int(math.sqrt(max_rel_pos))
    pos = 2  # self
    for i in range(max_u_op):
        for j in range(max_u_op):
            if i == 0 and j == 0:
                continue
            else:
                ud2pos[str(i + 0.001 * j)] = pos
                pos += 1
    return ud2pos


def clip_ud_mask(ud_mask, ud2pos, max_rel_pos):
    l = len(ud_mask)
    clipped_ud = np.zeros((l, l), dtype=np.int)
    for i, item in enumerate(ud_mask):
        path_rel = item.split()
        for j, path_len in enumerate(path_rel):
            if path_len in ud2pos:
                clipped_ud[i, j] = ud2pos[path_len]
            else:
                clipped_ud[i, j] = max_rel_pos
    return clipped_ud


def get_ud_masks(dp, id2pos, max_len):
    def get_path(i, j):
        if i == j:
            return "<self>"
        if i - j >= max_len:
            return "0"
        anc_i = set(ancestors[i])
        for node in ancestors[j][-(levels[i] + 1):]:
            if node in anc_i:
                up_n = levels[i] - levels[node]
                down_n = levels[j] - levels[node]
                return str(up_n + 0.001 * down_n)

    ancestors, levels = get_ancestors(dp, id2pos)
    path_rel = []
    for i in range(len(dp)):
        path_rel.append(" ".join([get_path(i, j) for j in range(len(dp))]))
    return path_rel


def connect_db():
    # 172.29.7.221  127.0.0.1
    client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
    return client.code_search_net
