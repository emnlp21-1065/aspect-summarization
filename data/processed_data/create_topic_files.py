import os

file_path = 'yelp_big/cate/'
with open(os.path.join(file_path, 'topics.txt')) as f:
    for line in f:
        tmp = line.split(' ')
        topic_id = tmp[1]
        topic_word = tmp[3:]
        with open(os.path.join(file_path, 'topics', topic_id+'.txt'), 'w') as fout:
            for w in topic_word:
                fout.write(w.replace(',','')+'\n')