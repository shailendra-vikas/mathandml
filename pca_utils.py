
import numpy as np
import pandas as pd

from statsmodels.multivariate.pca import PCA
from numpy.linalg import LinAlgError


def drop_first_eigenvector(data_df, min_count=100):
    """ Make a rolling pca factorization and drop the most significat factor and regenerate the data without that eigenvector
        Assuming data_df is DataFrame of daily_return with zero mean and same variabce with index as the date.
    """

    count = 0
    pca_data = []
    for index, row in data_df.iterrows():
        count += 1
        if count <= MIN_COUNT:
            pca_data.append((np.nan,)*4)
            continue
        X = data_df.iloc[count-MIN_COUNT-1:count-1].copy()
        try:
            pc = PCA(X, standardize=False, demean=False ) # Assuming the daily_returns has have mean zero and volatility similar for each columns
        except LinAlgError:
            pca_data.append((np.nan,)*4)
            continue
        inv_coeff  = np.linalg.inv(pc.coeff) # called loading
        score = np.dot(row, inv_coeff)
        row_pcadj  =  np.dot(score[1:], pc.coeff[1:]) # mapping back to data space but droping scores and eigenvalues of first(biggest) factor
        pca_data.append(row_pcadj.tolist())

    pca_adj_df = pd.DataFrame.from_records(pca_data, columns = data_df.columns, index= data_df.index)
    return pca_adj_df

