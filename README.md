# Statistical Proof of Shitposting on Crypto Subreddits

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
```