import tdfTrader as trader

host = 'http://localhost:3000'
agentid = '5334c0626b983ef571027139'
apikey = 'ewnemdmxsgnqerscxwbntosmoaxxirhw'


if __name__ == '__main__':

    print 'Running...'

    print 'All Histories (length): %s' % len(trader.allHistories(host))
    print 'Current Status (length): %s' % len(trader.currentStatus(host))
    print 'Tradable Symbols (length): %s' % len(trader.symbols(host))

    print ''
    print 'Agent Details...'

    agent_status = trader.status(host, agentid, apikey)
    print agent_status

    success = trader.trade(host, agentid, apikey, {'GOOG': 10, 'NFLX': 15})
    print 'Basic buy, result should be True: %s' % success
    success = trader.trade(host, agentid, apikey, {'BOGUS': -16})
    print 'Bad trade, result should be False: %s' % success
    success = trader.trade(host, agentid, apikey, {'GOOG': -10})
    print 'Basic sell, result should be True: %s' % success
