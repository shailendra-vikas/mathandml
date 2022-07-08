
import numpy as np
from sklearn.linear_model import LinearRegression
# https://en.wikipedia.org/wiki/Hurst_exponent


#def get_returns(ticker='^GSPC', start='2015-12-31', end='2021-06-16'):
#    import yfinance as yf
#    raw_data = yf.download(ticker, start, end)['Close']
#    prices = np.array(raw_data)[1:]
#    returns = np.array(raw_data)[1:]/np.array(raw_data)[:-1] - 1
#    return returns[-1024:]

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

class HurstExponent(object):
    """ 
          |-  R(n)   -|        H
        E |  -------  |  = C n       
          |-  S(n)   -|

        Where 
        R(n) is the range of the first n cumulative deviations from the mean
        S(n) is the series (sum) of the first n standard deviations
        E is the expected value
        n is the time span of the observation
        C is a constant

        Finally 
        Log(E) = H * Log(n) + Log(c)
    """
    def __init__(self, data, min_n = 16):
        self.min_n = min_n
        self.data = data
        self.linear_regressor = LinearRegression()

    def _calculate_Expected_R_S(self, data):
        """ Return a dictionary with n as keys and list of range_over_stddev.
        """
        N = len(data)
        all_R_S = dict()
        for start_index, end_index in subsample_iter(N, min_size=16):
            n = end_index - start_index
            curr_data = data[start_index:end_index]
            demean_curr_data = curr_data - np.average(curr_data)
            cum_demean_curr_data = np.cumsum(demean_curr_data)
            range_curr_data = cum_demean_curr_data.max() - cum_demean_curr_data.min()
            std_curr_data = np.std(demean_curr_data)
            range_over_std = range_curr_data/std_curr_data
            all_R_S.setdefault(n,list()).append(range_over_std)
        return all_R_S
    
    def _regress_linear(self, all_R_S):
        """ Return a regressor for log(n) and log(E(R/S))
        """
        sorted_n  = np.array(sorted(all_R_S.keys()))
        log_average_R_S = np.array([np.log(np.average(all_R_S[n])) for n in sorted_n])
        log_n = np.log(sorted_n)

        reg  = self.linear_regressor.fit(log_n.reshape(-1,1), log_average_R_S)
        return reg

    def calculate_hurst(self, data=None):
        """ return the hurst exponent.
        """
        data = self.data if data is None else data
        all_R_S = self._calculate_Expected_R_S(data)
        reg = self._regress_linear(all_R_S)
        return reg.coef_[0]

    def plot(self, data=None):
        data = self.data if data is None else data
        all_R_S = self._calculate_Expected_R_S(data)
        reg = self._regress_linear(all_R_S)
        sorted_n = sorted(all_R_S.keys())
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(1,1)
        for n, list_of_R_S in all_R_S.items():
            ax1.plot([n,]*len(list_of_R_S), list_of_R_S, 'o', color='blue')
        ax1.plot(sorted_n, [np.average(all_R_S[n_]) for n_ in sorted_n], '*', color='red')
        
        n_predict = np.logspace(np.log10(min(sorted_n)),np.log10(max(sorted_n)),101)
        ax1.plot(n_predict, np.exp(self.linear_regressor.predict(np.log(n_predict).reshape(-1,1))),'-', color='green')
        ax1.set_ylabel("R/S")
        ax1.set_xlabel("n")
        ax1.set_title(f'Hurse Exponent is {reg.coef_[0]:.2f}')
        plt.show()



    
if __name__=='__main__':
    #data = get_returns()
    data = get_randomwalk(factor=0.0)
    hurst_exponent = HurstExponent(data=data)
    print(hurst_exponent.calculate_hurst())

    #hurst_exponent.plot()
    #calculate_hurst(data)
    
