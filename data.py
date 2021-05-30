import re
import pandas as pd
from random import shuffle


if __name__ == "__main__":
    pattern = re.compile(r'''<doc id=(\d+)><summary>(.+)</summary><short_text>(.+)</short_text></doc>''', re.M)

    with open('data/raw_data/yelp/yelp_bart.csv', encoding='utf-8') as f:
        text = ''.join(f.readlines())
    matches = re.findall(pattern, text)#[:11000]
    shuffle(matches)
    train_matches = matches#[:11000]
    df = pd.DataFrame(train_matches)
    df.to_csv('data/processed_data/yelp/yelp_bart.csv', sep='\t', header=False, index=False)
    # eval_matches = matches[11000:]
    # df = pd.DataFrame(eval_matches)
    # df.to_csv('data/processed_data/yelp_big/eval_bart.csv', sep='\t', header=False, index=False)


    