import numpy as np
import re
from tqdm import tqdm
from nltk import sent_tokenize

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
        if t > 0.0:
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

def classify_doc(text):
    categories = np.zeros(len(topics))
    text_tmp = clean_str(text)
    for w in text_tmp.split(' '):
        if w in word2topic:
            categories[word2topic[w]] += 1
    categories /= np.sum(categories)
    topic_words = {}
    result_string = ""
    doc_num = {}
    for i in range(len(categories)):
        result_string += f"({i},{categories[i]}): {topic_name[i]}, "
        if categories[i] > 0.05:
            tmp_list = []
            for j,d in enumerate(text.split('</s>')):
                doc = clean_str(d)
                for w in doc.split(' '):
                    if w in word2topic and w in topics[i]:
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

    # print(result_string)
    # print(topic_words)
    # exit(1)
                    
    # dist = {i:categories[i] for i in range(len(categories)) if categories[i] > 0.1}
    return topic_words


def select_docs(docs, doc_class):
    topic_docs = {}
    for topic in doc_class:
        if len(doc_class[topic]) > 0:
            tmp_doc = []
            for d in docs.split("</s>"):  
                summ = ""
                sents = sent_tokenize(d.strip())           
                for sent in sents:
                    if any([w in clean_str(sent).split(' ') for w in doc_class[topic]]):
                        summ += sent + ' '
                if summ != "":
                    tmp_doc.append(summ)
            tmp_doc = '</s>'.join(tmp_doc)
            if len(tmp_doc) != 0:
                topic_docs[topic] = tmp_doc
                # print("docs", topic, tmp_doc)
    return topic_docs




file_name = '/shared/data2/jiaxinh3/summarization/BERT-Transformer-for-Summarization/data/processed_data/yelp_big/cate/topics.txt'
topics = {}
word2topic = {}
topic_name = []
with open(file_name) as f:
    for i,line in enumerate(f):
        tmp = line.split(' ')
        topics[i] = []
        for w in tmp[3:]:
            topics[i].append(w.replace(',',''))
            word2topic[w.replace(',','')] = i
        topic_name.append(tmp[3].replace(',',''))
        


corpus_file = '/shared/data2/jiaxinh3/summarization/BERT-Transformer-for-Summarization/data/processed_data/yelp/yelp_bart.csv'
output_file = '/shared/data2/jiaxinh3/summarization/BERT-Transformer-for-Summarization/data/processed_data/yelp/yelp_try_0.05.txt'
corpus = []
class_num_count = {}
doc_length_total = 0
max_doc_length = 0
cutoff_num = 0
with open(output_file,'w') as fout:
    with open(corpus_file) as f:
        index = 0
        summary_id = 0
        for line in tqdm(f, total=200):
            # print(str(index+1))
            summary = line.split('\t')[1]
            docs = line.split('\t')[2]
            doc_class = classify_doc(docs)
            # class_num = len(doc_class)
            # if class_num not in class_num_count:
            #     class_num_count[class_num] = 0
            # class_num_count[class_num] += 1
            # fout.write(','.join([','.join(doc_class[topic]) for topic in doc_class]))
            # fout.write('\n')
            sel_docs = select_docs(docs, doc_class)
            for topic in doc_class:
                if len(doc_class[topic]) == 0:
                    continue
                try:
                    fout.write('\t'.join([line.split('\t')[0], ','.join(doc_class[topic]), sel_docs[topic]]))
                    fout.write('\n')
                except KeyError:
                    print(doc_class)
                    print(sel_docs)
                summary_id += 1
                doc_length_total += len(sel_docs[topic].split(' '))
                if max_doc_length < len(sel_docs[topic].split(' ')):
                    max_doc_length = len(sel_docs[topic].split(' '))
                if len(sel_docs[topic].split(' ')) > 256:
                    cutoff_num += 1
            index += 1
            # if index > 10:
            #     break

print("average doc length: ", doc_length_total/(summary_id+1))
print("max doc length: ", max_doc_length)
print("cutoff at 256: ", cutoff_num/(summary_id+1))
            
        
    
        