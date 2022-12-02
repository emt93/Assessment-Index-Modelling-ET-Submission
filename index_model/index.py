import datetime as dt
import pandas as pd
import copy


class IndexModel:
    def __init__(self) -> None:
        # Ingest CSV stock price data into a dataframe and set Date column to datetime
        self.stock_prices_df = pd.read_csv('data_sources/stock_prices.csv')
        self.stock_prices_df['Date'] = \
            pd.to_datetime(self.stock_prices_df['Date'], errors="raise", format="%d/%m/%Y").dt.date
        # Create Year-Month column for given Dates
        self.stock_prices_df['Yr_Mth'] = pd.to_datetime(self.stock_prices_df['Date']).dt.to_period('M')
        self.stock_prices_df.sort_values(by='Date', inplace=True)
        self.index_prices_output = None
        pass

    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None:
        # Using stock_prices_df, create weights dataframe based on the following index rules, take the last date of
        # each month, since each stock has the same number of shares outstanding take the price as market cap,
        # and attribute 50%/25%/25% weights to the three largest stocks respectively for the following month
        def first_date_weights(mth_df):
            last_day_mth = mth_df.head(1).squeeze()
            last_day_mth.drop(['Date', 'Yr_Mth'], inplace=True)
            # Sort to find the three largest stocks by market cap (i.e., price)
            sorted_stk_last_day = last_day_mth.sort_values(ascending=False)
            if sorted_stk_last_day.isna().all():
                # Set weights to zero if all values are nan
                mth_df[sorted_stk_last_day.index] = 0
            else:
                # Replace top three with 50%/25%/25%
                list_weights = [0.5, 0.25, 0.25] + [0] * (len(sorted_stk_last_day) - 3)
                weights_mth = pd.Series(list_weights, index=sorted_stk_last_day.index)
                mth_df[weights_mth.index] = weights_mth.values
            return mth_df

        weights_df = copy.deepcopy(self.stock_prices_df)
        # Shift the values for the stock prices by a day so that the first month date is the last month end's prices
        stk_list_cols = weights_df.columns.drop(['Date', 'Yr_Mth'])
        weights_df[stk_list_cols] = weights_df[stk_list_cols].shift(1)
        # Apply groupby function to the last day of each month the index rule
        weights_df = weights_df.groupby(by=['Yr_Mth'], group_keys=False).apply(first_date_weights)

        # Truncate stock_prices_df and weights_df to start date and end date
        mask_dates = (end_date >= self.stock_prices_df['Date']) & (self.stock_prices_df['Date'] >= start_date)
        masked_stk_prc_df = self.stock_prices_df
        masked_weights_df = weights_df

        # Calculate the stock returns from each stock price
        stock_returns = masked_stk_prc_df[stk_list_cols].pct_change(periods=1)
        # Calculate the index return by doing the sumproduct the weights by the respective stock returns by date,
        # shift index weights by a day since Index is recalculated after the close of the first business day so
        # there is a lag on the weights where the first days' return is based on the prior month's dates
        masked_stk_prc_df['Index_Return'] = (masked_weights_df[stk_list_cols].shift(1).fillna(0) *
                                             stock_returns).sum(axis=1)

        # Find Index Price by taking the cumulative returns and normalizing to 100pts initial index value
        masked_stk_prc_df['Index_Level'] = (1 + masked_stk_prc_df.loc[mask_dates, 'Index_Return']).cumprod()
        first_idx_price = masked_stk_prc_df.loc[mask_dates, 'Index_Level'].head(1).values
        masked_stk_prc_df['Index_Level'] = ((masked_stk_prc_df['Index_Level'] / first_idx_price) * 100)
        self.index_prices_output = masked_stk_prc_df.loc[mask_dates, ['Date', 'Index_Level']]
        pass

    def export_values(self, file_name: str) -> None:
        # Create the export file
        file_path = f"data_sources/{file_name}"
        self.index_prices_output.to_csv(file_path, index=False)
        pass
