import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
import os
import json
import time
import torch.nn.functional as F
from preprocess import LCSTSProcessor, convert_examples_to_features, create_dataset, shuffle_src
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_FILE = 'train_eval_extract_w_noise.txt'
INPUT_FILE = 'train_keywords.txt'

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data path. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", 
                    default=None, 
                    type=str, 
                    required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")

# Opitional paramete
parser.add_argument("--GPU_index",
                    default='-1',
                    type=str,
                    help="Designate the GPU index that you desire to use. Should be str. -1 for using all available GPUs.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
parser.add_argument("--max_src_len",
                    default=256,
                    type=int,
                    help="Max sequence length for source text. Sequences will be truncated or padded to this length")
parser.add_argument("--max_tgt_len",
                    default=60,
                    type=int,
                    help="Max sequence length for target text. Sequences will be truncated or padded to this length")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--decoder_config",
                    default=None,
                    type=str,
                    help="Configuration file for decoder. Must be in JSON format.")
parser.add_argument("--print_every",
                    default=100,
                    type=int,
                    help="Print loss every k steps.")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
# parser.add_argument('--draft_only',
#                     action='store_true',
#                     help="Only use stage 1 to generate drafts.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")




if __name__ == "__main__":
    args = parser.parse_args()

    if args.GPU_index != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_index
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    assert args.train_batch_size % n_gpu == 0
    logger.info(f'Using device:{device}, n_gpu:{n_gpu}')

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_path = os.path.join(args.output_dir, time.strftime('model_%m-%d-%H:%M:%S', time.localtime()))
        os.mkdir(model_path)
        logger.info(f'Saving model to {model_path}.')

    # if args.decoder_config is not None:
    #     with open(args.decoder_config, 'r') as f:
    #         decoder_config = json.load(f)
    # else:
    #     with open(os.path.join(args.bert_model, 'bert_config.json'), 'r') as f:
    #         bert_config = json.load(f)
    #         decoder_config = {}
    #         decoder_config['len_max_seq'] = args.max_tgt_len
    #         decoder_config['d_word_vec'] = bert_config['hidden_size']
    #         decoder_config['n_layers'] = 8
    #         decoder_config['n_head'] = 12
    #         decoder_config['d_k'] = 64
    #         decoder_config['d_v'] = 64
    #         decoder_config['d_model'] = bert_config['hidden_size']
    #         decoder_config['d_inner'] = bert_config['hidden_size']
    #         decoder_config['vocab_size'] = bert_config['vocab_size']
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
                
    # train data preprocess
    processor = LCSTSProcessor()
    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    logger.info('Loading train examples...')
    if not os.path.exists(os.path.join(args.data_dir, TRAIN_FILE)):
        raise ValueError(f'train.csv does not exist.')
    train_examples = processor.get_examples(os.path.join(args.data_dir, TRAIN_FILE))
    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    logger.info('Converting train examples to features...')
    train_features = convert_examples_to_features(train_examples, args.max_src_len, args.max_tgt_len, tokenizer)
    example = train_examples[0]
    example_feature = train_features[0]
    logger.info("*** Example ***")
    logger.info("guid: %s" % (example.guid))
    logger.info("src text: %s" % example.src)
    logger.info("src_ids: %s" % " ".join([str(x) for x in example_feature.src_ids]))
    logger.info("src_mask: %s" % " ".join([str(x) for x in example_feature.src_mask]))
    logger.info("input keywords: %s" % example.keyword)
    logger.info("key_ids: %s" % " ".join([str(x) for x in example_feature.key_ids]))
    logger.info("tgt text: %s" % example.tgt)
    logger.info("tgt_ids: %s" % " ".join([str(x) for x in example_feature.tgt_ids]))
    logger.info("tgt_mask: %s" % " ".join([str(x) for x in example_feature.tgt_mask]))
    logger.info('Building dataloader...')
    train_data = create_dataset(train_features, args.train_batch_size)
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    # eval data preprocess
    if not os.path.exists(os.path.join(args.data_dir, 'eval_bart.csv')):
        logger.info('No eval data found in data directory. Eval will not be performed.')
        eval_dataloader = None
    else:
        logger.info('Loading eval dataset...')
        eval_examples = processor.get_examples(os.path.join(args.data_dir, "eval_extract_w_noise.txt"))
        logger.info('Converting eval examples to features...')
        eval_features = convert_examples_to_features(eval_examples, args.max_src_len, args.max_tgt_len, tokenizer)
        eval_data = create_dataset(eval_features, args.train_batch_size)
        # eval_sampler = RandomSampler(eval_data)
        # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size, drop_last=True)


    # model
    # model = BertAbsSum(args.bert_model, decoder_config, device)
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=args.learning_rate,)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_train_optimization_steps*0.1,
        num_training_steps=num_train_optimization_steps,
    )

    
    # training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    global_step = 0
    # with open('loss.txt','w') as fout:
    for i in range(int(args.num_train_epochs)):
        # do training
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        shuffled_train_data = shuffle_src(train_data)
        for step, batch in enumerate(tqdm(shuffled_train_data, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            # print("src", len(batch[0][0]), "src_mask", len(batch[1][0]), "decoder", len(batch[2][0]), "labels", len(batch[3][0]))
            output = model(input_ids=batch[0], attention_mask=batch[1], decoder_input_ids = batch[2], labels=batch[3], return_dict=True)
            loss = output.loss
            # print(loss)
            # print(output.logits.shape)
            # exit(1)
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            if (step + 1) % args.print_every == 0:
                logger.info(f'Epoch {i}, step {step}, loss {loss.item()}.')
                logger.info(f'Source: {" ".join(tokenizer.convert_ids_to_tokens(batch[0][0].cpu().numpy())).replace("Ġ","")}')
                logger.info(f'Ground: {" ".join(tokenizer.convert_ids_to_tokens(batch[2][0].cpu().numpy())).replace("Ġ","")}')
                logger.info(f'Generated: {" ".join(tokenizer.convert_ids_to_tokens(output.logits[0].max(-1)[1].cpu().numpy())).replace("Ġ","")}')
                # fout.write(str(loss.item())+"\n")
        # do evaluation
        if args.output_dir is not None:
            state_dict = model.module.state_dict() if n_gpu > 1 else model.state_dict()
            torch.save(state_dict, os.path.join(model_path, 'BartAbsSum_{}.bin'.format(i)))
            logger.info('Model saved')
        if eval_data is not None:
            model.eval()
            # batch = next(iter(eval_dataloader))
            batch = eval_data[i]
            batch = tuple(t.to(device) for t in batch)
            # beam_decode
            if n_gpu > 1:
                pred, _ = model.module.beam_decode(batch[0], batch[1], 3, 3)
            else:
                summary_ids = model.generate(input_ids=batch[0], attention_mask=batch[1], decoder_input_ids=batch[4], num_beams=5, max_length=100, early_stopping=True)
            logger.info(f'Source: {" ".join(tokenizer.convert_ids_to_tokens(batch[0][0].cpu().numpy())).replace("Ġ","")}')
            logger.info(f'Beam Generated: {tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).replace("Ġ","")}')
            logger.info(f'Ground Truth: {" ".join(tokenizer.convert_ids_to_tokens(batch[2][0].cpu().numpy())).replace("Ġ","")}')
            # if n_gpu > 1:
            #     pred = model.module.greedy_decode(batch[0], batch[1])
            # else:
            #     pred = model.greedy_decode(batch[0], batch[1])
            # logger.info(f'Beam Generated: {tokenizer.convert_ids_to_tokens(pred[0].cpu().numpy())}')
        logger.info(f'Epoch {i} finished.')
    # with open(os.path.join(args.bert_model, 'bert_config.json'), 'r') as f:
    #     bert_config = json.load(f)
    # config = {'bert_config': bert_config, 'decoder_config': decoder_config}
    # with open(os.path.join(model_path, 'config.json'), 'w') as f:
    #     json.dump(config, f)
    logger.info('Training finished')


