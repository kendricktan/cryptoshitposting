import argparse
import os
import pathlib
import json
import math
import numpy as np

from tqdm import tqdm
from functools import reduce
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# Folder Settings
ROOT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
SUBREDDIT_FOLDER = os.path.join(DATA_FOLDER, 'subreddits')


def visualize_scatter(data_2d, label_ids, id_to_label_dict, figsize=(20, 20)):
    plt.figure(figsize=figsize)
    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.show()


def get_data_from_json_file(filename):
    '''
    '''
    with open(filename, 'r') as f:
        data_json = json.load(f)

    titles = reduce(lambda x, y: x + [y['title']], data_json, [])
    scores = reduce(lambda x, y: x + [int(y['ups'])], data_json, [])

    try:
        scores_avg = sum(scores) / len(scores)
    except:
        scores_avg = 0

    try:
        scores_median = scores[int(math.floor(len(scores)/2))]
    except:
        scores_median = 0

    return titles, scores, scores_avg, scores_median


def vectorize_titles(titles):
    # Vectorize
    vectorizer = TfidfVectorizer(min_df=2, stop_words='english',
                                 strip_accents='unicode', lowercase=True, ngram_range=(1, 2),
                                 norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)

    X = vectorizer.fit_transform(titles)

    return X


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

    titles = []
    labels = []

    for json_filepath in tqdm(_path.iterdir()):
        # Get vector and median/average score
        t, s, s_avg, s_median = get_data_from_json_file(json_filepath)

        coin_name = json_filepath.name.replace('.json', '')

        try:
            # Only want subreddits which
            # have 25 or more posts
            if len(s) < 25:
                continue

            # If median score for reddit
            # post is < 75, ignore, not enough
            # content quality
            if s_median < 75:
                continue

            l[coin_name] = {
                'score_average': s_avg,
                'score_median': s_median,
                'coinmarketcap_stats': cmc_data[coin_name]
            }

            # Extend to titles
            titles.extend(t)

            # Our labels
            metric = float(cmc_data[coin_name]['market_cap_usd'])

            if metric >= 1_000_000_000:
                label_custom = '1Bil++'
            elif metric >= 250_000_000:
                label_custom = '250Mil++'
            elif metric >= 50_000_000:
                label_custom = '50Mil++'
            else:
                label_custom = '<50Mil'

            label = list(map(lambda x: label_custom, s))
            labels.extend(label)

        except Exception as e:
            pass

    # Plotting stuff
    label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels])

    # Vectorize our titles
    v = vectorize_titles(titles)
    v_np = np.array(v.todense().tolist())

    # Fit through PCA / TSNE
    pca_result = PCA().fit_transform(v_np)
    visualize_scatter(pca_result, label_ids, id_to_label_dict)
