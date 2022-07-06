
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def get_data(ticker, start_date):
    """  Temporary function since yahoo source has stopped working.
    """
    ticker_data = pd.read_csv('2590360.csv', parse_dates=['date'])
    ticker_data['Adj Close'] = ticker_data['close']*ticker_data['cumulative_split']
    ticker_data.set_index('date', inplace=True)
    return ticker_data[['close','Adj Close']]

def get_data_1(ticker, start_date):
    """ Function to get dataframe. date should be index and "Adj Close" column is only column of interest
    """
    from pandas_datareader import data
    ticker_data = data.DataReader(ticker, data_source='yahoo', start=start_date)
    return ticker_data


def choose_correct_N(adj_close_df, max_cluster=6):
    """ Choose the optimal clustering. If the inertia is not 50% better than we choose that N.
    """
    kmeans_list = {}
    for i in range(1, max_cluster):
        kmeans_list[i] = KMeans(n_clusters=i).fit(adj_close_df)
    print([(key, value.inertia_) for key, value in kmeans_list.items()])

    cluster = 1
    while cluster < (max_cluster - 1) and kmeans_list[cluster].inertia_/kmeans_list[cluster+1].inertia_ > 1.5:
        cluster += 1
    print('Correct cluster = %d'%cluster)
    return kmeans_list[cluster]


def make_adjustment_to_support(correct_kmeans, daily_std):
    """ Merge the levels if they are too close. The too close is dependent on the daily_std provided.
    """
    support_info = list()
    centroids = correct_kmeans.cluster_centers_[correct_kmeans.cluster_centers_[:,1].argsort()]
    cutoff_change = daily_std*5

    index = 0
    start_day = 0
    prices = []
    while index < centroids.shape[0]-1:
        current_price = centroids[index, 0]
        next_price = centroids[index+1, 0]
        if abs(current_price - next_price)/current_price > cutoff_change:
            prices.append(current_price)
            support_info.append((start_day, np.mean(prices)))
            prices = []
            start_day = (centroids[index, 1] + centroids[index+1, 1])/2.
        else:
            prices.append(current_price)
        index += 1
    prices.append(centroids[index, 0])
    support_info.append((start_day, np.mean(prices)))
    return support_info


def find_support_info(ticker, ticker_data):
    adj_close = ticker_data['Adj Close']
    adj_close_df = pd.DataFrame({'adj_close': adj_close, 'days': (adj_close.index-adj_close.index[0]).days})
    correct_kmeans = choose_correct_N(adj_close_df, max_cluster=6)
    daily_std = np.log(adj_close_df.adj_close/adj_close_df.adj_close.shift(1)).std()
    support_info = make_adjustment_to_support(correct_kmeans, daily_std)
    first_date = ticker_data.index[0].date()
    support_lines = list()
    last_price = None
    print(support_info)
    for start_day, price  in support_info:
        start_date = first_date + datetime.timedelta(days=int(start_day))
        if last_price is not None:
            support_lines.append((start_date, last_price))
        support_lines.append((start_date, price))
        last_price = price
    if last_price is not None:
        support_lines.append((ticker_data.index[-1].date(), last_price))
    return support_lines


def plot_ticker(ticker, ticker_data, support_info):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, figsize=(14, 7))
    ticker_data['Adj Close'].plot(ax=ax, grid=True)
    dates, support_levels = zip(*support_info)
    ax.plot(dates, support_levels, color='red')
    plt.show()
     

def main():
    ticker = 'GLD'
    start_date = datetime.date.today() - datetime.timedelta(days=3*365)
    ticker_data = get_data(ticker, start_date)
    support_info = find_support_info(ticker, ticker_data)
    plot_ticker(ticker, ticker_data, support_info)


if __name__ == '__main__':
    main()
