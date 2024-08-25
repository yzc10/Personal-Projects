GET_EVENTS_PROMPT = """
# Task: 
For the following stock, list the most significant events that have or may have affected the stock's price from Jan 2020 to Dec 2023.
Adhere to the following:
- Make sure to list 25-30 events or more. Include global or macro events (e.g. pandemics, recessions, wars, etc.), stock-specific news as well as all earnings releases

# How to format your output:
{format_instructions}

# Stock:
[{stock_ticker}] {company_name}
"""