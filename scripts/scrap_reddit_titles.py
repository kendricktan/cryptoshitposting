import argparse
import json
import requests
import os

from tqdm import tqdm
from functools import reduce

# Folder Settings
ROOT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
SUBREDDIT_FOLDER = os.path.join(DATA_FOLDER, 'subreddits')

if not os.path.exists(SUBREDDIT_FOLDER):
    os.makedirs(SUBREDDIT_FOLDER)


def get_top_25_titles(subreddit_name, peroid_type='all'):
    '''
    Gets the title (and upvotes) for the top 50 titles within a subreddit
    
    :param subreddit_name: string for subreddit name (e.g. bitcoin, zensys, zec)
    :param peroid_type: string type for top type (e.g. all, month, week, day)
    '''

    # Base strings
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36'
    }
    URL = 'https://www.reddit.com/r/{}/top/.json?count=20&sort=top&t={}'.format(
        subreddit_name, peroid_type)

    # Limited to 25 posts per GET requests
    r_json = requests.get(URL, headers=HEADERS).json()

    # List of
    titles = []

    try:
        # lambda function
        def f(x, y):
            return x + [{'title': y['data']['title'], 'ups': y['data']['ups']}]

        titles = reduce(f, r_json['data']['children'], titles)

        return titles

    except Exception as e:
        print(e)
        return titles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--peroid", type=str, default='all', choices=[
                        'all', 'year', 'month', 'week', 'day', 'hour'], help="Scrap subreddit by top of <x> peroid (Default: all)")
    args = parser.parse_args()

    PEROID_TYPE = args.peroid
    FOLDER_PATH = os.path.join(SUBREDDIT_FOLDER, PEROID_TYPE)

    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)

    with open(os.path.join(DATA_FOLDER, 'subreddits.json'), 'r') as f:
        subreddit_json = json.load(f)

    for idx, coin in tqdm(enumerate(subreddit_json)):
        subreddit_name = subreddit_json[coin]
        d = get_top_50_titles(subreddit_name, PEROID_TYPE)

        with open(os.path.join(FOLDER_PATH, '{}.json'.format(coin)), 'w') as f:
            json.dump(d, f, indent=4)
