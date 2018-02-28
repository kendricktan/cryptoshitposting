# Statistical Proof of Shitposting on Crypto Subreddits

# Install requirements
```
pip install -r requirements.txt
```

## Steps to preprocessing data
### 1. Collecting data (scrapping subreddit titles and score)
```bash
# To scrap top titles from `all`
python scrap_reddit_titles.py

# To scrap top titles from `month` (Can choose from 'day', 'week', 'month', 'year', 'all')
# Located in 'ROOT_FOLDER/data/subreddits/all/<period>/...json'
python scrap_reddit_titles.py -p month
```

### 2. Vectorize the collected data
```bash
# To convert the scrapped 'top' titles
python titles_to_vectors.py

# To convert the scrapped 'month' titles (Chooes from 'day', 'week', 'month', 'year', 'all')
# Located in 'ROOT_FOLDER/src/assets/vectors_<period>.json'
python titles_to_vectors.py -p month
```