import pickle
import sys
from pymongo import MongoClient
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

sys.setrecursionlimit(1000000)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

word_tokens = []

client = MongoClient('172.29.7.221', 27017, username='admin', password='123456')
codes = client.code_search_net.codes
query = codes.find({'partition': 'train', 'build_ast': 1})

non_leaf_nodes = set()


def get_non_leaf_node(node):
    li = []
    if len(node['children']) > 0:
        li.append(node['type'])
        for child in node['children']:
            li += get_non_leaf_node(child)
    return li


for result in tqdm(query, total=query.count()):
    word_tokens.extend(result['code_tokens'])
    root_node = pickle.loads(result['ast'])
    for non_leaf_node in get_non_leaf_node(root_node):
        non_leaf_nodes.add(non_leaf_node)
    word_tokens.extend(result)
    word_tokens.extend(result['docstring_tokens'])

special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"] + list(non_leaf_nodes)
trainer = trainers.BpeTrainer(vocab_size=50000, min_frequency=3, special_tokens=special_tokens)

tokenizer.train_from_iterator(word_tokens, trainer=trainer, length=len(word_tokens))
tokenizer.save('./deberta_tokenizer.json', pretty=True)

test_seq = ['func', 'Seek', 'SecondaryTree', 'trueorfalse']
non_leaf = ['update_expression', 'update_expression', 'program']

tokenizer = Tokenizer.from_file('./deberta_tokenizer.json')
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                    unk_token='<unk>',
                                    bos_token="<s>",
                                    eos_token="</s>",
                                    cls_token='<s>',
                                    sep_token='</s>',
                                    pad_token='<pad>',
                                    mask_token='<mask>')

tokens = [tokenizer.tokenize(' @' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(test_seq)]
tokens = [y for x in tokens for y in x]
print(tokens)
source_ids = tokenizer.convert_tokens_to_ids(tokens)
non_leaf_ids = tokenizer.convert_tokens_to_ids(non_leaf)
print(source_ids)
print(non_leaf_ids)
