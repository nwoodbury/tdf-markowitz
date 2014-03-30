import tdfTrader as trader
import pandas as pd
import math
#import random
import pulp

# TDF Information
host = 'http://localhost:3000'
agentid = '5334c4a06b983ef57102948b'
apikey = 'qmemevlbyesdhbciwjshvaswloeqybrn'
symbols = ['GOOG', 'AAPL', 'NFLX', 'MSFT', 'GM', 'FE', 'FDX', 'DAL',
           'COV', 'CCE']

torun = 'test'


def is_equal(x, y, tol=0.001):
    """
    Utility for assertions. Returns true if x == y within tol tolerance.

    @param x {number} The actual value.
    @param y {number} The expected value.
    @param tol {number, default = 0.0001} The test tolerance.

    @returns {boolean}
    """
    return x <= y + tol and x >= y - tol


def test_portfolio(portfolio, answers, mu):
    failed = False
    for (security, quantity) in portfolio.iteritems():
        expected = answers[mu].get(security, 0)
        if not is_equal(quantity, expected):
            print 'Trade on mu=%.1f: %s=%.3f, expected %.3f' % (mu, security,
                                                                quantity,
                                                                expected)
            failed = True

    return failed


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
        securities.unit test

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


def get_stock_data(src='tdf'):
    """
    Given a list of symbols, queries TDF for the historical prices of those
    symbols, returning data for the historical returns, means of returns, and
    standard deviation of returns for each security.

    @param src {string in {'tdf', 'test'}} The source of the data. If 'tdf',
        grabs historical prices from the TDF server listed at the top of
        the page. If 'test', grabs data from the CS 412 finance homework
        problem.

    @returns {pandas.DataFrame} A dataframe where the columns are symbols
        and each row is the return seen at a particular time indicated
        by the row. Sorted from oldest to most recent.
    @returns {pandas.Series} A series where each label is a symbol and the
        value is the mean of returns seen by that security.
    @returns {pandas.Series} A series where each label is a symbol and the
        value is the standard deviation of returns seen by that security.
    """
    if src == 'tdf':
        histories = pd.DataFrame(trader.allHistories(host))
        histories = histories.transpose()[symbols]
        #histories = histories[0:20]
        histories = histories.apply(lambda x: __ap_row(x), axis=1)
    else:
        histories = pd.read_csv('testdata.csv')

    curr = histories[1:]
    prev = histories[:-1]

    prev = prev.transpose()
    prev.columns = list(curr.axes[0])
    prev = prev.transpose()

    returns = (curr - prev) / prev
    means = returns.mean()

    stdevs = returns.apply(lambda x: stdev(x, means), axis=0)

    return [returns, means, stdevs]


def get_mad_lp(returns, means, mu):
    """
    Formulates the MAD model as a pulp linear program.

    @param returns {pandas.DataFrame} A dataframe where the columns are symbols
        and each row is the return seen at a particular time indicated
        by the row. Sorted from oldest to most recent.
    @param means {pandas.Series} A series where each label is a symbol and the
        value is the mean of returns seen by that security.
    @param mu {number} The risk aversion factor.

    @returns {pulp.LpProblem} The pulp lp problem with variables, an objective,
        and constraints added.
    """
    problem = pulp.LpProblem('Markowitz Portfolio Optimization',
                             pulp.LpMaximize)
    symbols = list(returns.axes[1])
    x = pulp.LpVariable.dicts('x', symbols, lowBound=0)
    y = pulp.LpVariable.dicts('y', range(0, len(returns.axes[0])), lowBound=0)

    # Objective: Max sum_i (mean(i)*x_i) - mu/float T * sum_j y_j
    T = float(len(y))
    mudT = mu / T
    problem += (sum(means[i] * x[i] for i in x) -
                sum(y[j] * mudT for j in y)), 'objective'

    # Equality constraint: sum_i x_i = 1
    problem += sum(x[i] for i in x) == 1

    # Time constraint: sum_j (mean(i) - return(i, j))x_i <= y_j for all j
    for j in range(0, len(y)):
        row = returns.iloc[j]
        problem += sum((means[i] - row[i]) * x[i] for i in x) <= y[j]

    return problem, x


def solve_problem(problem, x):
    problem.solve()
    portfolio = {i: x[i].value() for i in x}
    return portfolio


if __name__ == '__main__':
    if (torun == 'test'):
        mu = 0

        # Step 1: Collect Stock Data
        returns, means, stdevs = get_stock_data(src='test')

        # Step 2: Formulate the MAD Linear Program
        problem, x = get_mad_lp(returns, means, mu)

        # Step 3: Solve the MAD LP and collect results
        portfolio = solve_problem(problem, x)

        # Step 4: Test Results
        answers = {
            0 : {'NASDAQ': 1}
        }
        failed = test_portfolio(portfolio, answers, mu)
        if failed:
            print '\nTESTS ON MU=%.1f FAILED!\n' % mu

        print '\nDONE TESTING\n'
