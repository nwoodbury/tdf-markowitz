# TODO Once TDF v2 is built, sync with new queries, add robustness and
# error handling.
import requests


def allHistories(host, options=None):
    query = '%s/allhistories' % host

    r = requests.get(query)
    return r.json()


def currentStatus(host, options=None):
    query = '%s/currentstatus' % host

    r = requests.get(query)
    return r.json()


def symbols(host, n=10):
    query = '%s/history?n=%i' % (host, n)

    r = requests.get(query)
    return r.json()


def status(host, agentid, apikey):
    query = '%s/agents/%s/composition?apikey=%s' % (host, agentid, apikey)

    r = requests.get(query)
    return r.json()


def trade(host, agentid, apikey, trade):
    query = '%s/agents/trade/%s?apikey=%s' % (host, agentid, apikey)

    for (symbol, quantity) in trade.iteritems():
        query = '%s&%s=%i' % (query, symbol, quantity)

    try:
        r = requests.get(query).json()
        if 'error' in r:
            return False
        else:
            return True
    except:
        return False
