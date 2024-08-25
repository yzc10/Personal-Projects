GET_EVENT_LABELS_PROMPT = """
# Task:
For each of the following article headlines, determine if it refers to an actual piece of news or not. 
Specify the is_news parameter as "yes" or "no".

# Criteria for is_news:
- is_news="yes": Headlines that refer to actual events that took place very recently/at the point of the article's release.
    - Events can include global events, market- or economy-wide events, company-specific events or updates (e.g. new partnership launch, new product launch, key personnel movement), etc.
    - Events can also include influential media mentions, e.g. if Elon Musk posts about a company.
    - Events do not include commentary or analysis about general industry/market trends, e.g. "AI wave", "Industry 3.0", etc.
- is_news="no": Headlines that refer to some sort of commentary/analysis.

# Output format:
- The output should STRICTLY be a list of dictionaries, where each dictionary corresponds to a headline.
    - Each dictionary should contain four fields: title, stock, article_idx and is_news
    - title refers to the main text of the headline that is not in parentheses.
    - stock is the ticker symbol of the stock to which the headline pertains. It is the string value in the first parentheses, where all letters are in upper case. e.g. MSFT."
    - article_idx is the index of the article. It is the integer value in the second parentheses.
    - Important: Use double quotes for all keys and values in each dictionary. E.g. "title":"Abc def", "stock":"AAPL"

# Sample Headline:
- Top 5 stocks to sell this year (MSFT) (1)
- Apple releases new Apple Watch (AAPL) (2)
- Why we should reconsider the Airbnb stock (SPY) (3)
- Airbnb CEO leaves for Booking.com (ABNB) (5)
- Exxon Mobil announces new partnership with Tesla (XOM) (8)
- Stock Analysis: Unlocking the potential of the Pepsi in 2025 (PEP) (9)
- Analysts tell us that Industry 3.0 is well under way (SPY) (10)

# Sample classification results:
- For title = Top 5 stocks to sell this year, is_news = "no"
- For title = Apple releases new Apple Watch, is_news = "yes"
- For title = Why we should reconsider the Airbnb stock, is_news = "no"
- For title = Airbnb CEO leaves for Booking.com, is_news = "yes"
- For title = Exxon Mobil announces new partnership with Tesla, is_news = "yes"
- For title = Stock Analysis: Unlocking the potential of the Pepsi in 2025 (PEP), is_news = "no"
- For title = Analysts tell us that Industry 3.0 is well under way, is_news = "no"

# Headlines:
{articles_list}

# Output:
"""