import os
import math
import logging
import argparse

from tqdm import tqdm

from config import add_args, set_seed, set_dist
import multiprocessing
import time

from model import build_or_load_gen_model
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import get_elapse_time, load_and_cache_gen_data_from_db
from collections import OrderedDict
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def eval_ppl_epoch(args, eval_dataloader, model):
    # Start evaluating model
    logger.info(" " + "**** Run ppl evaluation ****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, position_idx, attn_mask, rel_pos, target_ids, target_mask, _ = batch
        with torch.no_grad():
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               position_idx=position_idx, attn_mask=attn_mask,
                               rel_pos=rel_pos,
                               target_ids=target_ids, target_mask=target_mask)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

        eval_loss += loss.item()
        batch_num += 1
    print(f'eval: {loss}')
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_acc_epoch(args, eval_dataloader, model, split_tag):
    logger.info("  ***** Running acc evaluation on {} data*****".format(split_tag))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    total_acc_num = 0
    total_num = 0

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval acc for {} set".format(split_tag)):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, position_idx, attn_mask, rel_pos, target_ids, target_mask, gold_ids = batch

        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask,
                          position_idx=position_idx, attn_mask=attn_mask,
                          rel_pos=rel_pos)
            # top_preds = [pred[0] for pred in preds]  # [batch_size, 7]
            top_preds = preds[:, 0, :]

            top_preds = top_preds * target_mask  # 对应位置进行mask
            total_num += torch.sum(target_mask[:, 1:]).cpu().numpy()
            pad_num = torch.sum(target_mask[:, 1:] == 0).cpu().numpy()
            acc = torch.sum(top_preds[:, 1:] == gold_ids[:, 1:]).cpu().numpy()
            total_acc_num += acc - pad_num

    print('total acc num', total_acc_num)
    acc = round(total_acc_num / total_num, 5) * 100
    logger.info("***** Eval results ***** Acc = %s", str(acc))

    return acc


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(args.cpu_cont)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')
    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_data = load_and_cache_gen_data_from_db(args, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step = 0
        best_ppl = 1e6
        best_acc = -1
        not_loss_dec_cnt = 0
        not_acc_inc_cnt = 0 if args.do_eval_acc else 1e6
        # global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        # not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc='Training')
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)

                source_ids, source_mask, position_idx, attn_mask, rel_pos, target_ids, target_mask, _ = batch

                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   position_idx=position_idx, attn_mask=attn_mask,
                                   rel_pos=rel_pos,
                                   target_ids=target_ids, target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_data = dev_dataset['dev_loss']
                else:
                    eval_data = load_and_cache_gen_data_from_db(args, pool, tokenizer, 'valid')
                    eval_sample = SequentialSampler(eval_data)
                    eval_data = DataLoader(eval_data, sampler=eval_sample, batch_size=args.eval_batch_size,
                                           num_workers=4, pin_memory=True)
                    dev_dataset['dev_loss'] = eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, model)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_acc_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_acc_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_acc_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_acc:
                    eval_data = dev_dataset['dev_loss']
                    result = eval_acc_epoch(args, eval_data, model, 'dev')
                    dev_acc = result
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_acc', dev_acc, cur_epoch)
                    if dev_acc > best_acc:
                        not_acc_inc_cnt = 0
                        logger.info("  [%d] Best acc: %.2f",
                                    cur_epoch, dev_acc)
                        logger.info("  " + "*" * 20)
                        best_acc = dev_acc
                        fa.write("[%d] Best acc changed into %.2f\n" % (
                            cur_epoch, best_acc))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best acc model into %s", output_model_file)
                    else:
                        not_acc_inc_cnt += 1
                        logger.info("Acc does not increase for %d epochs", not_acc_inc_cnt)
                        fa.write(
                            "[%d] Best acc (%.2f) does not drop changed for %d epochs, cur acc: %.2f \n" % (
                                cur_epoch, best_acc, not_acc_inc_cnt, dev_acc))
                        if all([x > args.patience for x in [not_acc_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_acc_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_acc_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  ATTENTION!!!!!!!!!!!!
        file = os.path.join(args.output_dir, 'checkpoint-best-ppl/pytorch_model.bin')
        logger.info("Reload model from {}".format(file))
        state_dict = torch.load(file)
        new_state_dict = OrderedDict()
        if hasattr(model, 'module'):
            for k, v in state_dict.items():
                name = 'module.' + k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load(file),False)
        eval_data = load_and_cache_gen_data_from_db(args, pool, tokenizer, 'test')
        eval_sample = SequentialSampler(eval_data)
        eval_data = DataLoader(eval_data, sampler=eval_sample, batch_size=args.eval_batch_size,
                               num_workers=4, pin_memory=True)
        result = eval_acc_epoch(args, eval_data, model, 'test')
        test_acc = result
        result_str = "[ppl] acc: %.2f\n" % test_acc
        logger.info(result_str)
        fa.write(result_str)
        if args.res_fn:
            with open(args.res_fn, 'a+') as f:
                f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
