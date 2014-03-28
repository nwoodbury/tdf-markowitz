import tdfTrader as trader
import pandas as pd
import math
#import random
import pulp

host = 'http://localhost:3000'
agentid = '5334c4a06b983ef57102948b'
apikey = 'qmemevlbyesdhbciwjshvaswloeqybrn'

mu = 1.5


def __ap_row(x):
    """
    Utility used to convert the all histories to a dataframe of bid prices.

    @param x {pandas.Series} A row in the DataFrame retrieved by an
        AllHistories query.

    @returns {pandas.Series} The row where each element contains only the
        bid price.
    """
    return x.apply(lambda y: y['bid'])


def stdev(s, means):
    """
    Used for computing the standard deviation for a column of time-series
    returns for a particular series.

    @param s {pandas.Series} The column for a single security showing the
        returns of that security over time.
    @param means {pandas.Series} A series of the mean of returns for all
        securities.

    @returns {number} The standard deviation of the returns for the given
        security.
    """
    mean = means[s.name]
    summed = 0
    count = 0
    for rtn in s:
        summed = (rtn - mean) ** 2
        count += 1

    return math.sqrt(summed / count)


def get_stock_data(symbols):
    """
    Given a list of symbols, queries TDF for the historical prices of those
    symbols, returning data for the historical returns, means of returns, and
    standard deviation of returns for each security.

    @param symbols {list} A list of strings of symbols to query.

    @pre each symbol in symbols must be tracked by the TDF host being queried.

    @returns {pandas.DataFrame} A dataframe where the columns are symbols
        and each row is the return seen at a particular time indicated
        by the row. Sorted from oldest to most recent.
    @returns {pandas.Series} A series where each label is a symbol and the
        value is the mean of returns seen by that security.
    @returns {pandas.Series} A series where each label is a symbol and the
        value is the standard deviation of returns seen by that security.
    """
    histories = pd.DataFrame(trader.allHistories(host))
    histories = histories.transpose()[symbols]
    #histories = histories[0:20]
    histories = histories.apply(lambda x: __ap_row(x), axis=1)

    curr = histories[1:]
    prev = histories[:-1]

    prev = prev.transpose()
    prev.columns = list(curr.axes[0])
    prev = prev.transpose()

    returns = (curr - prev) / prev
    means = returns.mean()

    stdevs = returns.apply(lambda x: stdev(x, means), axis=0)

    return [returns, means, stdevs]


def get_mad_lp(returns, means):
    """
    Formulates the MAD model as a pulp linear program.

    @param returns {pandas.DataFrame} A dataframe where the columns are symbols
        and each row is the return seen at a particular time indicated
        by the row. Sorted from oldest to most recent.
    @param means {pandas.Series} A series where each label is a symbol and the
        value is the mean of returns seen by that security.

    @returns {pulp.LpProblem} The pulp lp problem with variables, an objective,
        and constraints added.
    """
    problem = pulp.LpProblem('Markowitz Portfolio Optimization',
                             pulp.LpMaximize)
    symbols = list(returns.axes[1])
    x = pulp.LpVariable.dicts('x', symbols, lowBound=0)
    y = pulp.LpVariable.dicts('y', range(0, len(returns.axes[0])), lowBound=0)

    # Objective: Max sum_i (mean(i)*x_i) - mu/T * sum_j y_j
    problem += (sum(means[i] * x[i] for i in x) -
                sum(y[j] * mu / len(y) for j in y)), 'objective'

    # Equality constraint: sum_i x_i = 1
    problem += sum(x[i] for i in x) == 1

    # Time constraint: sum_j (mean(i) - return(i, j))x_i <= y_j for all j
    for j in range(0, len(y)):
        row = returns.ix[j]
        problem += sum((means[i] - row[i]) * x[i] for i in x) <= y[j]

    return problem


if __name__ == '__main__':
    symbols = ['GOOG', 'AAPL', 'NFLX', 'MSFT', 'GM', 'FE', 'FDX', 'DAL',
               'COV', 'CCE']

    # Step 1: Collect Stock Data
    returns, means, stdevs = get_stock_data(symbols)

    # Step 2: Formulate the MAD Linear Program
    problem = get_mad_lp(returns, means)

    # Step 3: Solve the MAD LP and collect results
    print problem
