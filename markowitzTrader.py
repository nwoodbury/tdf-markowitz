import tdfTrader as trader
import pandas as pd
import math
#import random
import pulp

# TDF Information
host = 'http://localhost:3000'
agentid = '5334c4a06b983ef57102948b'
apikey = 'qmemevlbyesdhbciwjshvaswloeqybrn'
symbols = ['AAPL', 'MSFT', 'YHOO', 'MU',
           'XOM', 'GE', 'CVX',
           'C', 'BAC', 'JPM']

torun = 'tdf-static'


def is_equal(x, y, tol=0.01):
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
    if src == 'tdf' or src == 'tdf-static':
        if src == 'tdf':
            histories = pd.DataFrame(trader.allHistories(host))
            histories = histories.transpose()[symbols]
            #histories = histories[0:20]
            histories = histories.apply(lambda x: __ap_row(x), axis=1)
        else:
            histories = pd.read_csv('biddata.csv', parse_dates=True,
                                    index_col=0)

        curr = histories[1:]
        prev = histories[:-1]

        prev = prev.transpose()
        prev.columns = list(curr.axes[0])
        prev = prev.transpose()

        returns = (curr - prev) / prev
    elif src == 'test':
        returns = pd.read_csv('testdata.csv')
        histories = None

    means = returns.mean()

    stdevs = returns.apply(lambda x: stdev(x, means), axis=0)

    return [returns, means, stdevs, histories]


def get_mad_lp(returns, means, mu, lb=0):
    """
    Formulates the MAD model as a pulp linear program.

    @param returns {pandas.DataFrame} A dataframe where the columns are symbols
        and each row is the return seen at a particular time indicated
        by the row. Sorted from oldest to most recent.
    @param means {pandas.Series} A series where each label is a symbol and the
        value is the mean of returns seen by that security.
    @param mu {number} The risk aversion factor.
    @param lb {number 0 <= x < 1} The percentage of the portfolio that must be
        kept in cash.

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
    problem += sum(x[i] for i in x) == 1 - lb

    # Time constraint: sum_j (mean(i) - return(i, j))x_i <= y_j for all j
    for j in range(0, len(y)):
        row = returns.iloc[j]
        problem += sum((means[i] - row[i]) * x[i] for i in x) <= y[j]

    return problem, x


def solve_problem(problem, x):
    problem.solve()
    portfolio = {i: x[i].value() for i in x}
    return portfolio


def fetch_current_status():
    """
    Fetches and returns the agent's status from TDF

    @returns {dict} The agent's status. Each key represents a symbol, and
        the value is another dict which gives the bid price (key=price),
        value=price*quantity (key=value), and quantity owned (key=quantity).
        In addition, two special keys are included, namely uninvested_cash
        which is the amount of cash that hasn't yet been invested, and
        total_value, which is the sum of uninvested_cash and the values
        of all securities.
    """
    status = trader.status(host, agentid, apikey)
    return status


def compute_current_composition(status):
    """
    Computes the percentage of the total portfolio represented by each
    owned security. For each security, this is defined by value / total_value.

    @param status {dict} The agent's status as returned by
        fetch_current_status().

    @returns {dict} The composition represented by each security, where the
        key is the security and the value is the composition.
    """
    current_composition = {}
    for (security, sec_status) in status.iteritems():
        if security == 'total_value' or security == 'uninvested_cash':
            continue
        current_composition[security] = (sec_status['value'] /
                                         status['total_value'])
    return current_composition


def get_composition_change_percent(portfolio, curr_composition):
    """
    Computes the change in computation from present to the computed optimal
    (in percent portfolio value).

    @param portfolio {dict} The computed optimal portfolio.
    @param curr_composition {dict} The current portfolio.

    @returns {dict} The change in portfolio percent.
    """
    delta_portfolio = {}
    for security in symbols:
        delta = portfolio.get(security, 0) - curr_composition.get(security, 0)
        if (math.fabs(delta) > .001):
            delta_portfolio[security] = delta
    return delta_portfolio


def convert_to_shares(portfolio, values, portfolio_value, lb=0):
    """
    Converts a portfolio of percentages to a number of shares of each security
    in the portfolio that should be owned after the purchase.

    @param portfolio {dict} The computed optimal portfolio.
    @param values {pandas.Series} The present ask values of all securities.
    @param portfolio_value {number} The present value of the portfolio.
    @param lb {number} The percent of the portfolio that should remain as cash

    @returns {pandas.Series} The shares of each security that should be owned.
    """
    portfolio = portfolio_value * (1 - lb) * pd.Series(portfolio)
    shares = {}
    for symbol in list(portfolio.axes[0]):
        shares[symbol] = int(math.floor(portfolio[symbol] / values[symbol]))
    return pd.Series(shares)


if __name__ == '__main__':
    if (torun == 'test'):
        for mu in [0, 0.1, 1, 2, 4, 8, 1024]:
            print 'Testing mu = %.1f' % mu

            # Step 1: Collect Stock Data
            returns, means, stdevs = get_stock_data(src='test')

            # Step 2: Formulate the MAD Linear Program
            problem, x = get_mad_lp(returns, means, mu)

            # Step 3: Solve the MAD LP and collect results
            portfolio = solve_problem(problem, x)

            # Step 4: Test Results
            answers = {
                0: {'S&P 500': 1},
                0.1: {'S&P 500': 1},
                1: {
                    'S&P 500': 0.537860,
                    'EAFE': 0.101952,
                    'Gold': 0.360189
                },
                2: {
                    'T-Bills': 0.596322,
                    'S&P 500': 0.050069,
                    'Lehman Bros': 0.221304,
                    'EAFE': 0.071737,
                    'Gold': 0.065004
                },
                4: {
                    'T-Bills': 0.636618,
                    'S&P 500': 0.044743,
                    'NASDAQ': 0.003321,
                    'Lehman Bros': 0.196823,
                    'EAFE': 0.059269,
                    'Gold': 0.059227
                },
                8: {
                    'T-Bills': 0.805118,
                    'S&P 500': 0.025491,
                    'NASDAQ': 0.030647,
                    'Lehman Bros': 0.083281,
                    'EAFE': 0.005346,
                    'Gold': 0.050117
                },
                1024: {
                    'T-Bills': 0.911269,
                    'S&P 500': 0.010245,
                    'Lehman Bros': 0.045950,
                    'EAFE': 0.016692,
                    'Gold': 0.015844
                }
            }
            failed = test_portfolio(portfolio, answers, mu)
            if failed:
                print '\nTESTS ON MU=%.1f FAILED!\n' % mu

        print '\nDONE TESTING\n'
    elif torun == 'tdf-static':
        # Make a decision based on queried data

        # Step 1: Collect Stock Data
        returns, means, stdevs, histories = get_stock_data(src=torun)

        # Step 2: Formulate the MAD Linear Program
        lb = 0.05
        problem, x = get_mad_lp(returns, means, 1, lb=lb)

        # Step 3: Solve the MAD LP and collect results
        portfolio = solve_problem(problem, x)

        # Step 4: Convert to shares
        shares = convert_to_shares(portfolio, histories.ix[-1, :], 100000, lb)
        shares.to_csv('markowitzshares.csv')

    elif torun == 'tdf':
        # Trade on TDF using the given authentication

        # Step 1: Collect Stock Data
        returns, means, stdevs = get_stock_data(src=torun)

        # Step 2: Formulate the MAD Linear Program
        problem, x = get_mad_lp(returns, means, 1, lb=0.05)

        # Step 3: Solve the MAD LP and collect results
        portfolio = solve_problem(problem, x)

        # Step 4: Evaluate Current Portfolio
        status = fetch_current_status()
        curr_composition = compute_current_composition(status)
        delta_composition = get_composition_change_percent(portfolio,
                                                           curr_composition)
        print delta_composition
