import numpy as np
import re
from tqdm import tqdm
from nltk import sent_tokenize
import random
from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def clean_str(string):
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def create_noisy(text):
    tmp = text.split(' ')
    mask_num = int(int(len(tmp) * 0.15) + 0.5)
    mask_index = random.sample([x for x in range(len(tmp))],k=mask_num)
    mask_index.sort()
    for index in mask_index:
        tmp[index] = '[MASK]'
    new_text = ' '.join(tmp)
    inputs = tokenizer(new_text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    ind = torch.topk(F.softmax(outputs.logits,dim=1), 1)[1]
    new_words = tokenizer.convert_ids_to_tokens(ind[0])
    new_mask_index = (inputs['input_ids'][0]==103).nonzero(as_tuple=False)
    # print(text)
    # print(new_text)
    for i, index in enumerate(mask_index):
        tmp[index] = new_words[new_mask_index[i]].replace('#','')
    # print(' '.join(tmp))
    return ' '.join(tmp)
    


def classify(text):
    categories = np.zeros(len(topics))
    text = clean_str(text)
    for w in text.split(' '):
        if w in word2topic:
            categories[word2topic[w]] += 1
    categories /= np.sum(categories)
    topic_words = {}
    result_string = ""
    for i, t in enumerate(categories):
        if t > 0.1:
            topic_words[i] = []
            result_string += f"({i},{t}): {topic_name[i]}, "
            topic_words[i].append(topic_name[i])
            for w in text.split(' '):
                if w in word2topic and word2topic[w] == i:
                    result_string += f"{w}, "
                    topic_words[i].append(w)
    for i in topic_words:
        topic_words[i] = list(set(topic_words[i]))
    # print(result_string)
                    
    dist = {i:categories[i] for i in range(len(categories)) if categories[i] > 0.1}
    return topic_words

def classify_doc(text, summary_class):
    categories = np.zeros(len(topics))
    text_tmp = clean_str(text)
    for w in text_tmp.split(' '):
        if w in word2topic:
            categories[word2topic[w]] += 1
    categories /= np.sum(categories)
    topic_words = {}
    result_string = ""
    doc_num = {}
    for i in summary_class:
        tmp_list = []
        if categories[i] > 0:
            for j,d in enumerate(text.split('</s>')):
                doc = clean_str(d)
                for w in doc.split(' '):
                    if w in word2topic and w in summary_class[i]:
                        tmp_list.append(w)
                        if w not in doc_num:
                            doc_num[w] = []
                        doc_num[w].append(j)
            if len(tmp_list) > 0:
                tmp_list = list(set(tmp_list))
                # print([(w, doc_num[w]) for w in tmp_list])
                topic_words[i] = [w for w in tmp_list if len(set(doc_num[w])) > 1]
                if len(topic_words[i]) > 0 and (topic_name[i] not in topic_words[i]):
                    tmp = topic_words[i]
                    topic_words[i] = [topic_name[i]]
                    topic_words[i].extend(tmp)
                if len(topic_words[i]) > 3:
                    topic_words[i] = topic_words[i][:3]

    # print(topic_words)
                    
    # dist = {i:categories[i] for i in range(len(categories)) if categories[i] > 0.1}
    return topic_words

def select_summary(summary, doc_class):
    sents = sent_tokenize(clean_str(summary))
    topic_summ = {}
    for topic in doc_class:
        summ = ""
        if len(doc_class[topic]) > 0:
            for sent in sents:
                if any([w in sent.split(' ') for w in doc_class[topic]]):
                    summ += sent + ' '
        if len(summ) > 0:
            topic_summ[topic] = summ
            # print("summary", topic, summ)
    return topic_summ

def select_docs(docs, doc_class, summary):
    topic_docs = {}
    for topic in doc_class:
        if len(doc_class[topic]) > 0 and topic in summary:
            tmp_doc = []
            for d in docs.split("</s>"):  
                summ = ""
                sents = sent_tokenize(d.strip())           
                for sent in sents:
                    if any([w in clean_str(sent).split(' ') for w in doc_class[topic]]):
                        summ += sent + ' '
                if summ != "":
                    tmp_doc.append(summ)
            if len(tmp_doc) != 0:
                rand_int = random.randint(0,len(tmp_doc))
                new_tmp = []
                new_tmp.extend(tmp_doc[0:rand_int])
                new_summary = create_noisy(summary[topic])
                new_tmp.append(new_summary)
                new_tmp.extend(tmp_doc[rand_int:])
                # print(new_tmp)
                new_tmp = '</s>'.join(new_tmp)
                topic_docs[topic] = new_tmp
                # print("docs", topic, tmp_doc)
    return topic_docs




file_name = 'data/processed_data/yelp_big/topics.txt'
topics = []
word2topic = {}
topic_name = []
with open(file_name) as f:
    for i,line in enumerate(f):
        tmp = line.split(' ')
        for w in tmp[3:]:
            topics.append(w.replace(',',''))
            word2topic[w.replace(',','')] = i
        topic_name.append(tmp[3].replace(',',''))
        


corpus_file = 'data/processed_data/yelp_big/train_bart.csv'
output_file = 'data/processed_data/yelp_big/train_extract_w_noise.txt'
corpus = []
class_num_count = {}
doc_length_total = 0
max_doc_length = 0
cutoff_num = 0
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with open(output_file,'w') as fout:
    with open(corpus_file) as f:
        index = 0
        summary_id = 0
        for line in tqdm(f, total=11000):
            # print(str(index+1))
            summary = line.split('\t')[1]
            docs = line.split('\t')[2]
            summary_class = classify(summary)
            doc_class = classify_doc(docs, summary_class)
            class_num = len(doc_class)
            if class_num not in class_num_count:
                class_num_count[class_num] = 0
            class_num_count[class_num] += 1
            # fout.write(','.join([','.join(doc_class[topic]) for topic in doc_class]))
            # fout.write('\n')
            sel_summary = select_summary(summary, doc_class)
            sel_docs = select_docs(docs, doc_class, sel_summary)
            if len(sel_summary) > 1:
                for topic in sel_summary:
                    fout.write('\t'.join([line.split('\t')[0], ','.join(doc_class[topic]), sel_summary[topic].strip(), sel_docs[topic]]))
                    fout.write('\n')
                    summary_id += 1
                    doc_length_total += len(sel_docs[topic].split(' '))
                    if max_doc_length < len(sel_docs[topic].split(' ')):
                        max_doc_length = len(sel_docs[topic].split(' '))
                    if len(sel_docs[topic].split(' ')) > 256:
                        cutoff_num += 1
            index += 1
            # if index > 50:
            #     break

print("average doc length: ", doc_length_total/(summary_id+1))
print("max doc length: ", max_doc_length)
print("cutoff at 256: ", cutoff_num/(summary_id+1))
            
        
    
        
