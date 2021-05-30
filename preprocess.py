import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import csv
import os
import logging
from utils import convert_to_unicode
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm, trange
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, src, tgt=None, keyword=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            src: string. The untokenized text of the target sequence.
            tgt: (Optional) string. The untokenized text of the target.
        """
        self.guid = guid
        self.src = src
        self.tgt = tgt
        self.keyword = keyword

class InputFeatures():
    """A single set of features of data."""

    def __init__(self, src_ids, src_mask, tgt_ids, tgt_mask, key_ids):
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.tgt_ids = tgt_ids
        self.tgt_mask = tgt_mask 
        self.key_ids = key_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a txt file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line)
            return lines

    def _read_data(cls, input_file):
        """Reads a txt file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line)
            return lines

class LCSTSProcessor(DataProcessor):
    """Processor for the LCSTS data set."""

    def get_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_data(data_path))

    def _create_examples(self, lines):
        examples = [] 
        for i,line in enumerate(lines):
            # lines: id, summary, text
            data = line.split('\t')
            guid = data[0]
            src = convert_to_unicode(data[3])
            tgt = convert_to_unicode(data[2])
            keyword = convert_to_unicode(data[1])
            examples.append(InputExample(guid=guid, src=src, tgt=tgt, keyword=keyword))
            # if i>100:
            #     break
        return examples

def convert_examples_to_features(examples, src_max_seq_length, tgt_max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc='examples')):
        src_tokens = tokenizer.tokenize(example.src)
        tgt_tokens = tokenizer.tokenize(example.tgt)
        key_tokens = tokenizer.tokenize(example.keyword)
        if len(src_tokens) > src_max_seq_length - 2:
            src_tokens = src_tokens[:(src_max_seq_length - 2)]
        if len(tgt_tokens) > tgt_max_seq_length - 1:
            tgt_tokens = tgt_tokens[:(tgt_max_seq_length - 1)]
        src_tokens = ["<s>"] + src_tokens + ["</s>"]
        tgt_tokens = tgt_tokens + ["</s>"]
        key_tokens = key_tokens + ["</s>"]
        # no need to generate segment ids here because if we do not provide
        # bert model will generate dafault all-zero ids for us
        # and we regard single text as one sentence

        src_ids = tokenizer.convert_tokens_to_ids(src_tokens)
        tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)
        key_ids = tokenizer.convert_tokens_to_ids(key_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        src_mask = [1] * len(src_ids)
        tgt_mask = [1] * len(tgt_ids)
        key_mask = [1] * len(key_ids)
        # Zero-pad up to the sequence length.
        src_padding = [0] * (src_max_seq_length - len(src_ids))
        tgt_padding = [0] * (tgt_max_seq_length - len(tgt_ids))
        src_ids += src_padding
        src_mask += src_padding
        tgt_ids += tgt_padding
        tgt_mask += tgt_padding

        assert len(src_ids) == src_max_seq_length
        assert len(tgt_ids) == tgt_max_seq_length

        features.append(InputFeatures(src_ids=src_ids,
                                      src_mask=src_mask,
                                      tgt_ids=tgt_ids,
                                      tgt_mask=tgt_mask,
                                      key_ids=key_ids))
    return features

def create_shuffle_dataset(all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask, all_key_ids, batch_size):
    key_id_len = torch.tensor([len(k) for k in all_key_ids])
    # print(key_id_len.shape)
    key_len_sort = torch.argsort(key_id_len)
    sort_src_ids, sort_src_mask, sort_tgt_ids, sort_tgt_mask, sort_key_ids = [], [], [], [], []
    cur_len = 0
    last_key_len = 0
    tmp_src_ids, tmp_src_mask, tmp_tgt_ids, tmp_tgt_mask, tmp_key_ids = [], [], [], [], []
    for i in range(len(key_id_len)):
        key_len = key_id_len[key_len_sort[i]]
        # print(key_len, last_key_len, cur_len)
        if cur_len % batch_size == 0:
            last_key_len = key_len
            if len(tmp_src_ids) != 0:
                sort_src_ids.append(tmp_src_ids)
                sort_src_mask.append(tmp_src_mask)
                sort_tgt_ids.append(tmp_tgt_ids)
                sort_tgt_mask.append(tmp_tgt_mask)
                sort_key_ids.append(tmp_key_ids)
                # print(tmp_key_ids)
                tmp_src_ids, tmp_src_mask, tmp_tgt_ids, tmp_tgt_mask, tmp_key_ids = [], [], [], [], []
        if key_len != last_key_len:
            cur_len -= (cur_len%batch_size)
            last_key_len = key_len
            tmp_src_ids, tmp_src_mask, tmp_tgt_ids, tmp_tgt_mask, tmp_key_ids = [], [], [], [], []
        tmp_src_ids.append(all_src_ids[key_len_sort[i]])
        tmp_src_mask.append(all_src_mask[key_len_sort[i]])
        tmp_tgt_ids.append(all_tgt_ids[key_len_sort[i]])
        tmp_tgt_mask.append(all_tgt_mask[key_len_sort[i]])
        tmp_key_ids.append(all_key_ids[key_len_sort[i]])
        cur_len += 1
            
            
    print(len(sort_key_ids))
    shuffle_id = [x for x in range(len(sort_key_ids))]
    random.shuffle(shuffle_id)
    shuffle_src_ids, shuffle_src_mask, shuffle_tgt_ids, shuffle_tgt_mask, shuffle_key_ids = [], [], [], [], []
    train_data=[]
    for i in shuffle_id:
        # shuffle_src_ids.append(torch.tensor(sort_src_ids[i], dtype=torch.long))
        # shuffle_src_mask.append(torch.tensor(sort_src_mask[i], dtype=torch.long))
        # shuffle_tgt_ids.append(torch.tensor(sort_tgt_ids[i], dtype=torch.long))
        # shuffle_tgt_mask.append(torch.tensor(sort_tgt_mask[i], dtype=torch.long))
        # shuffle_key_ids.append(torch.tensor(sort_key_ids[i], dtype=torch.long))
        train_data.append((torch.tensor(sort_src_ids[i], dtype=torch.long), 
            torch.tensor(sort_src_mask[i], dtype=torch.long), 
            torch.cat((torch.tensor(sort_key_ids[i], dtype=torch.long),torch.tensor(sort_tgt_ids[i], dtype=torch.long)), dim=1)[:, :-1], 
            torch.cat((torch.full_like(torch.tensor(sort_key_ids[i], dtype=torch.long), -100),torch.tensor(sort_tgt_ids[i], dtype=torch.long)), dim=1)[:, 1:], 
            torch.tensor(sort_key_ids[i], dtype=torch.long)))
    return train_data

def shuffle_src(train_data):
    new_train_data = []
    print("shuffling document order!")
    for batch in tqdm(train_data, total=len(train_data)):
        src = batch[0]
        new_src = []
        src_max_seq_length = len(src[0])
        for instance in src:
            # print("instance", instance)
            sep_index = (instance == 2).nonzero(as_tuple=False)[:,0]
            segment_index = [(sep_index[i-1]+1, sep_index[i]+1) if i > 0 else (torch.tensor(1), sep_index[i]+1) for i,sep in enumerate(sep_index)]
            shuffle_id = [x for x in range(len(segment_index))]
            random.shuffle(shuffle_id)
            tmp = torch.cat([instance[segment_index[i][0]:segment_index[i][1]] for i in shuffle_id])
            tmp = torch.cat((tmp, torch.zeros(src_max_seq_length-len(tmp)-1, dtype=torch.long)))
            # print("tmp",tmp)
            tmp = torch.cat((torch.tensor([0]), tmp))
            new_src.append(tmp.unsqueeze(0))
        new_src = torch.cat(new_src, dim = 0)
        new_batch = (new_src, batch[1], batch[2], batch[3], batch[4])
        new_train_data.append(new_batch)
    return new_train_data

def create_dataset(features, batch_size):
    all_src_ids = [f.src_ids for f in features]
    all_src_mask = [f.src_mask for f in features]
    all_tgt_ids = [f.tgt_ids for f in features]
    all_tgt_mask =[f.tgt_mask for f in features]
    all_key_ids = [f.key_ids for f in features]
    train_data = create_shuffle_dataset(all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask, all_key_ids, batch_size)
    # train_data = TensorDataset(all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask, all_key_ids)
    return train_data

def convert_one_example(text, src_max_seq_length, tokenizer):
    src_tokens = tokenizer.tokenize(text)
    if len(src_tokens) > src_max_seq_length - 2:
        src_tokens = src_tokens[:(src_max_seq_length - 2)]
    src_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]

    src_ids = tokenizer.convert_tokens_to_ids(src_tokens)

    src_mask = [1] * len(src_ids)
    src_padding = [0] * (src_max_seq_length - len(src_ids))
    src_ids += src_padding
    src_mask += src_padding

    return torch.tensor([src_ids]), torch.tensor([src_mask])

if __name__ == "__main__":
    # processor = LCSTSProcessor()
    tokenizer = BertTokenizer.from_pretrained(os.path.join('pretrained_model', 'vocab.txt'))    # examples = processor.get_train_examples('data/processed_data')
    # features = convert_examples_to_features(examples, 130, 20, tokenizer)

    print(convert_one_example('新京报讯（记者 梁辰）5月20日，针对外媒报道因美国政府将华为列入实体名单，而谷歌已暂停与华为部分业务往来的消息，谷歌中国通过邮件回复记者称，“我们正在遵守这一命令，并审查其影响”。', 130, tokenizer))
    
    