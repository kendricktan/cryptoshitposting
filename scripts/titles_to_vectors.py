import argparse
import os
import pathlib
import json
import math
import numpy as np

from tqdm import tqdm
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# Folder Settings
ROOT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
SUBREDDIT_FOLDER = os.path.join(DATA_FOLDER, 'subreddits')


def get_vectors_from_json(filename):
    with open(filename, 'r') as f:
        data_json = json.load(f)    

    titles = reduce(lambda x, y: x + [y['title']], data_json, [])
    scores = reduce(lambda x, y: x + [int(y['ups'])], data_json, [])
    scores_avg = sum(scores) / len(scores)
    scores_median = scores[int(math.floor(len(scores)/2))]

    # Vectorize
    vectorizer = TfidfVectorizer(min_df=2, stop_words='english', max_features=30,
                                 strip_accents='unicode', lowercase=True, ngram_range=(1, 2),
                                 norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)

    X = vectorizer.fit_transform(titles)

    flatten = lambda l: [item for sublist in l for item in sublist]

    return (flatten(X.todense().tolist()), scores_avg, scores_median)


if __name__ == '__main__':
    # Args settings
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--peroid", type=str, default='all', choices=[
                        'all', 'year', 'month', 'week', 'day', 'hour'], help="Scrapped subreddit data folder path (subreddits/<peroid>)")
    args = parser.parse_args()

    # Folder settings
    PEROID_TYPE = args.peroid
    FOLDER_PATH = os.path.join(SUBREDDIT_FOLDER, PEROID_TYPE)

    VECTORIZED_PATH = os.path.join(os.path.join(ROOT_FOLDER, 'src'), 'assets')

    if not os.path.exists(VECTORIZED_PATH):
        os.makedirs(VECTORIZED_PATH)

    # Read coinmarketcap dump
    with open(os.path.join(DATA_FOLDER, 'cmc_dump.json'), 'r') as f:
        cmc_dump = json.load(f)

    # Format cmc data
    cmc_data = reduce(lambda x, y: {**x, y['id']: {**y}}, cmc_dump, {})

    # Get all files in the folder
    _path = pathlib.Path(FOLDER_PATH)
    l = {}
    vectors = []
    vectors_coin = []
    for json_filepath in tqdm(_path.iterdir()):
        # Get vector and median/average score
        v, s_avg, s_median = get_vectors_from_json(json_filepath)

        coin_name = json_filepath.name.replace('.json', '')

        try:
            # Only care about subreddits with
            # A large enough community to have
            # 50 posts (then it has potential)
            if len(v) < 1500:
                continue

            l[coin_name] = {
                'score_average': s_avg,
                'score_median': s_median,
                'coinmarketcap_stats': cmc_data[coin_name]
            }
            
            vectors.append(v)
            vectors_coin.append(coin_name)

        except Exception as e:
            pass

    # Fit tsne    
    v_np = np.array(vectors)    
    x_tsne = TSNE().fit_transform(v_np).tolist()

    # Overwrite l
    mpl_x = []
    mpl_y = []
    mpl_c = []

    for i in range(len(x_tsne)):
        c = vectors_coin[i]
        coord = x_tsne[i]

        l[c]['coord'] = coord

        # top 10
        _rank = int(l[c]['coinmarketcap_stats']['rank'])

        mpl_x.append(x_tsne[i][0])
        mpl_y.append(x_tsne[i][1])        

        if _rank < 75:
            mpl_c.append('g')
        
        elif _rank < 125:
            mpl_c.append('y')
        
        else:
            mpl_c.append('r')

    plt.scatter(np.array(mpl_x), np.array(mpl_y), c=np.array(mpl_c))
    plt.show()

    # Write to file
    with open(os.path.join(VECTORIZED_PATH, 'crypto_vectors_{}.json'.format(PEROID_TYPE)), 'w') as f:
        json.dump(l, f)
