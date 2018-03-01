# Statistical Proof of Shitposting on Crypto Subreddits

Only Python 3.x is supported

# Install requirements
```
pip install -r requirements.txt
```

## Steps to preprocessing data
### 1. Collecting data (scrapping subreddit titles and score)
```bash
# To scrap top titles from `all`
python scripts/scrap_reddit_titles.py

# To scrap top titles from `month` (Can choose from 'day', 'week', 'month', 'year', 'all')
# Located in 'ROOT_FOLDER/data/subreddits/<period>/*.json'
python scripts/scrap_reddit_titles.py -p month
```

### 2. Vectorize the collected data
```bash
# To process and visualize the top of all time subreddit titles
python scripts/viz_titles.py

# To process and visualize the the scrapped 'month' titles (Choose from 'day', 'week', 'month', 'year', 'all')
python scripts/viz_titles.py -p month

# For help
python scripts/viz_titles.py --help
usage: viz_titles.py [-h] [-pc POST_COUNT] [-xr X_RANGE X_RANGE]
                     [-yr Y_RANGE Y_RANGE] [-smm SCORE_MEDIAN_MIN]
                     [-p {all,year,month,week,day,hour}]

optional arguments:
  -h, --help            show this help message and exit
  -pc POST_COUNT, --post-count POST_COUNT
                        Minimum post count to be accepted into preprocessing.
  -xr X_RANGE X_RANGE, --x-range X_RANGE X_RANGE
                        Prints out pca coordinates that is within this range
                        on the x axis
  -yr Y_RANGE Y_RANGE, --y-range Y_RANGE Y_RANGE
                        Prints out pca coordinates that is within this range
                        on the y axis
  -smm SCORE_MEDIAN_MIN, --score-median-min SCORE_MEDIAN_MIN
                        Only accepts reddit posts with >=<value>. Assume
                        hivemind.
  -p {all,year,month,week,day,hour}, --peroid {all,year,month,week,day,hour}
                        Scrapped subreddit data folder path
                        (subreddits/<peroid>)
```