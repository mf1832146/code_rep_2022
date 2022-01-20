#!/usr/bin/env python
import os
import argparse


def get_cmd(task, sub_task, model_tag, gpu, data_num, bs, lr, max_ast_len,
            max_code_len, max_dfg_len, max_rel_pos, use_ast, use_code, use_dfg,
            target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn):
    cmd_str = 'bash exp_with_args.sh %s %s %s %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %s %s %s' % \
              (task, sub_task, model_tag, gpu, data_num, bs, lr,
               max_ast_len, max_code_len, max_dfg_len, max_rel_pos, use_ast, use_code, use_dfg,
               target_length, patience, epoch,
               warmup, model_dir, summary_dir, res_fn)
    return cmd_str


def get_args_by_task_model(task):
    if task == 'method_name':
        max_ast_len = 512
        max_code_len = 256
        max_dfg_len = 64
        max_rel_pos = 64

        trg_len = 7
        epoch = 200
        patience = 5

        bs = 32
    lr = 5
    return bs, lr, max_ast_len, max_code_len, max_dfg_len, max_rel_pos, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, max_ast_len, max_code_len, max_dfg_len, max_rel_pos, trg_len, patience, epoch =\
        get_args_by_task_model(args.task)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, max_ast_len=max_ast_len,
                      max_code_len=max_code_len, max_dfg_len=max_dfg_len,
                      max_rel_pos=max_rel_pos,
                      use_ast=args.use_ast, use_code=args.use_code, use_dfg=args.use_dfg,
                      target_length=trg_len,
                      patience=patience, epoch=epoch, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag))
    print('%s\n' % cmd_str)
    os.system(cmd_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='deberta',
                        choices=['deberta'])
    parser.add_argument("--task", type=str, default='method_name', choices=['method_name'])
    parser.add_argument("--sub_task", type=str, default='ruby')
    parser.add_argument("--use_ast", action='store_true')
    parser.add_argument("--use_code", action='store_true')
    parser.add_argument("--use_dfg", action='store_true')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=str, default='0', help='index of the gpu to use in a cluster')
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    run_one_exp(args)
