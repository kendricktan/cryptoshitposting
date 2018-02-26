import requests

from functools import reduce


def get_top_50_titles(subreddit_name):
    '''
    :param subreddit_name: string for subreddit name (e.g. bitcoin, zensys, zec)
    '''    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36'
    }

    URL = 'https://www.reddit.com/r/{}/top/.json?count=20&sort=top&t=all'.format(subreddit_name)

    # Limited to 25 posts per GET requests
    r_json = requests.get(URL, headers=HEADERS).json()    

    titles = []

    try:        
        titles = reduce(lambda x, y: x + [y['data']['title']], r_json['data']['children'], titles)

        # We do it here. In the event that it fails, just return the top 25
        after_key = r_json['data']['after']    
        r_json2 = requests.get(URL + '&after={}'.format(after_key), headers=HEADERS).json()
        titles = reduce(lambda x, y: x + [y['data']['title']], r_json2['data']['children'], titles)
        return titles

    except:
        return titles
