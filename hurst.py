import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# https://en.wikipedia.org/wiki/Hurst_exponent
def get_returns(ticker='^GSPC', start='2015-12-31', end='2021-06-16'):
    import yfinance as yf
    raw_data = yf.download(ticker, start, end)['Close']
    prices = np.array(raw_data)[1:]
    returns = np.array(raw_data)[1:]/np.array(raw_data)[:-1] - 1
    return returns[-1024:]

def get_randomwalk(factor=0.0):
    ts = np.random.randn(1200)
    ts[1:] = ts[1:]+factor*ts[:-1]
    return ts[-1024:]

def subsample_iter(N, min_size=4):
    div_no=1
    size = int(N/div_no)
    while size >= min_size:
        for i in range(div_no):
            yield int(N*i/div_no), int(N*(i+1)/div_no)
        div_no *= 2
        size = int(N/div_no)

#class RollingHurst(object):
#    def __init__(self, window_size):
#        self.window_size = window_size
    
def calculate_hurst(data):
    N = len(data)
    all_R_S = dict()
    for start_index, end_index in subsample_iter(N, min_size=16):
        n = end_index - start_index
        X = data[start_index:end_index]
        m = np.average(X)
        Y = X - m
        Z = np.cumsum(Y)
        R = Z.max() - Z.min()
        S = np.std(X)
        R_S = R/S   
        all_R_S.setdefault(n,list()).append(R_S)

    sorted_n = sorted(all_R_S.keys())
    average_R_S = [np.average(all_R_S[n]) for n in sorted_n]

    log_n = np.log(np.array(sorted_n))
    log_R_S = np.log(np.array(average_R_S))

    reg = sm.OLS(log_R_S, sm.add_constant(log_n))
    res = reg.fit()
    hurst = res.params[1]
    print('Hurst Exponent=', hurst)
    return 
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(range(N), np.cumsum(data), '-')
    for n, list_of_R_S in all_R_S.items():
        ax2.plot([n,]*len(list_of_R_S), list_of_R_S, 'o', color='blue')

    ax2.plot(sorted_n, average_R_S, '*', color='red')
    ax2.plot(sorted_n, np.exp(res.params[0])*np.array(sorted_n)**res.params[1],'-', color='green')
    plt.show()

def calculate_hurst2(input_ts, lags_to_test=20):
    tau = []
    lagvec = []  
    #  Step through the different lags  
    for lag in range(2, lags_to_test):  
        #  produce price difference with lag  
        pp = np.subtract(input_ts[lag:], input_ts[:-lag])  
        #  Write the different lags into a vector  
        lagvec.append(lag)  
        #  Calculate the variance of the differnce vector  
        tau.append(np.sqrt(np.var(pp)))  
    #  linear fit to double-log graph (gives power)  
    m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)  
    # calculate hurst  
    hurst = m[0]*2
    print(hurst)

    
if __name__=='__main__':
    #data = get_returns()
    data = get_randomwalk(factor=-1.0)
    calculate_hurst(data)
    calculate_hurst2(data.cumsum())

