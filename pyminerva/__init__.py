# Copyright 2021-2023 jeongmin kang, jarvisNim@GitHub
# See LICENSE for details.

__author__ = "jarvisNim in GitHub"
__version__ = "0.0.2"


from .economics import (
    get_trend,
    get_country_growth,
    get_market_growth,
    get_business_growth,
)
from .option import (
    get_crypto_historical_data,
    get_crypto_information,
    get_crypto_recent_data,
    get_cryptos,
    get_cryptos_dict,
    get_cryptos_list,
    get_cryptos_overview,
    search_cryptos,
)
from .currency import (
    get_available_currencies,
    get_currency_cross_historical_data,
    get_currency_cross_information,
    get_currency_cross_recent_data,
    get_currency_crosses,
    get_currency_crosses_dict,
    get_currency_crosses_list,
    get_currency_crosses_overview,
    search_currency_crosses,
)
from .etfs import (
    get_etf_countries,
    get_etf_historical_data,
    get_etf_information,
    get_etf_recent_data,
    get_etfs,
    get_etfs_dict,
    get_etfs_list,
    get_etfs_overview,
    search_etfs,
)
from .indices import (
    get_index_countries,
    get_index_historical_data,
    get_index_information,
    get_index_recent_data,
    get_indices,
    get_indices_dict,
    get_indices_list,
    get_indices_overview,
    search_indices,
)
from .news import economic_calendar
from .stocks import (
    get_stock_company_profile,
    get_stock_countries,
    get_stock_dividends,
    get_stock_financial_summary,
    get_stock_historical_data,
    get_stock_information,
    get_stock_recent_data,
    get_stocks,
    get_stocks_dict,
    get_stocks_list,
    get_stocks_overview,
    search_stocks,
)
from .technical import moving_averages, pivot_points, technical_indicators

# from .search import search_events




# class PyMinerva:
#     def __init__(self):
#         pass

#     @staticmethod
#     def lower(string: str):
#         """
#         Converts a string to lowercase
#         """
#         return string.lower()

#     @staticmethod
#     def upper(string: str):
#         """
#         Converts a string to uppercase
#         """
#         return string.upper()

#     @staticmethod
#     def title(string: str):
#         """
#         Converts a string to titlecase
#         """
#         return string.title()