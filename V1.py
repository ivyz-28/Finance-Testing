# Import Section
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# Define start and end dates
start_date = "2021-06-30"
end_date = datetime.now().strftime('%Y-%m-%d')

# Create market class
class Market:
    def __init__(self, market = "US", start = start_date, end = end_date):
        self.mticker_us = "^GSPC"
        self.rfticker_us = "^TNX"
        
        self.start = start
        self.end = end
        
        self.mdata = yf.download(self.mticker_us, start = self.start, end = self.end, interval = '1mo')
        if isinstance(self.mdata.columns, pd.MultiIndex):
            self.mdata.columns = self.mdata.columns.get_level_values(0)
        self.rfinfo = yf.Ticker(self.rfticker_us)

        self.mreturns = np.log(self.mdata['Close']/self.mdata['Close'].shift(1)).dropna()
        return

    def rfr(self):
        # Returns the risk free rate of return
        self.rfr_value = self.rfinfo.info['previousClose'] / 100
        return self.rfr_value
    def mr(self):
        # Returns expected market return (S&P500 Proxy)
        self.total_return = (self.mdata['Close'].iloc[-1] / self.mdata['Close'].iloc[0]) - 1
        years = len(self.mdata) / 12
        self.mr_value = (1 + self.total_return) ** (1 / years) - 1
        return self.mr_value

market_data = Market()

# Create asset class
class Asset:
    def __init__(self, ticker, market, start = start_date, end = end_date, min_months = 12):
        self.ticker = ticker
        self.market = market
        self.start = start
        self.end = end

        self.tinfo = yf.Ticker(self.ticker)
        self.data = yf.download(self.ticker, start=self.start, end=self.end, interval='1mo')
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)

        if self.data.empty:
            raise ValueError(f"No price data returned for '{self.ticker}'.")

        if len(self.data) < min_months:
            print(f"Warning: '{self.ticker}' has only {len(self.data)} months of data "
                  f"(listed {self.data.index[0].date()}). Portfolio-level stats involving "
                  f"this asset may be unreliable or fail.")

        self.current_price = self.data['Close'].iloc[-1]
        self.returns = np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()
        return
    
    def __str__(self):
        # Show the ticker of the stock when the asset is called
        return self.ticker
        
    def beta(self):
        # Returns Beta value of a stock
        self.combined = pd.concat([self.returns, self.market.mreturns], axis = 1, sort = False).dropna()
        self.combined.columns = [self.ticker, 'Market']

        covariance = self.combined[self.ticker].cov(self.combined['Market'])
        variance = self.combined['Market'].var()
        self.beta_value = covariance / (variance)
        return self.beta_value
    
    def volatility(self):
        # Returns Volatility of a stock
        self.volatility_annual = self.returns.std() * np.sqrt(12)
        return self.volatility_annual

    def returns_list(self):
        # Returns total return, annual return, and expected return of a stock
        self.total_return = (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0])-1

        years = len(self.data) / 12
        self.annual_return = (1 + self.total_return) ** (1 / years) - 1

        self.expected_return = self.market.rfr() + (self.market.mr() - self.market.rfr()) * self.beta()
        return [self.total_return, self.annual_return, self.expected_return]
    
    def sr(self):
        # Returns Sharpe Ratio of a stock
        self.sr_value = (self.returns_list()[1] - self.market.rfr()) / self.volatility()
        return self.sr_value
    
    def pe(self):
        # Returns trailing and forward P/E Ratio
        self.tpe = self.tinfo.info.get('trailingPE','N/A')
        self.fpe = self.tinfo.info.get('forwardPE','N/A')
        return [self.tpe, self.fpe]
    def oupriced(self):
        # Determines if an asset is over or under priced
        self.pb = self.tinfo.info.get('priceToBook','N/A')
        return self.pb

# Create portfolio class
class Portfolio:
    def __init__(self,positions):
        # Positions: Set of lists, [0] = ticker, [1] = Amount of shares, [2] = Average share price
        grouped = {}
        for ticker, shares, cost in positions:
            if ticker in grouped:
                existing_shares, existing_cost = grouped[ticker]
                total_shares = existing_shares + shares
                
                avg_cost = (existing_shares * existing_cost + shares * cost) / total_shares
                grouped[ticker] = (total_shares, avg_cost)
            else:
                grouped[ticker] = (shares, cost)

        if len(grouped) < len(positions):
            dupes = [t for t in grouped if sum(1 for p in positions if p[0] == t) > 1]
            print(f"Grouped duplicate tickers: {dupes}")

        self.positions = [(ticker, shares, cost) for ticker, (shares, cost) in grouped.items()]
        self.assets = []
        for position in self.positions:
            temp_class = Asset(position[0],market_data)
            self.assets.append((temp_class,temp_class.current_price*position[1]))
        return
    
    def value(self):
        # Returns the current value of a portfolio
        self.portfolio_value = 0
        for asset in self.assets:
            self.portfolio_value += asset[1]
        return self.portfolio_value
    
    def weightings(self):
        # Returns the weights of each asset in a portfolio
        self.weighting_value = []
        self.portfolio_value = self.value()
        for asset in self.assets:
            self.weighting_value.append((asset[0], asset[1]/self.portfolio_value))
        return self.weighting_value
    
    def beta(self):
        # Returns the beta value of the portfolio
        self.portfolio_beta_value = 0
        for weights in self.weightings():
            self.portfolio_beta_value += weights[1]*weights[0].beta()
        return self.portfolio_beta_value

    def returns(self):
        # Returns the returns of the portfolio
        self.buy_in_cost = 0
        for position in self.positions:
            self.buy_in_cost += position[1]*position[2]
        self.unrealizedpl = self.value() - self.buy_in_cost
        return self.unrealizedpl

    def _cov_matrix(self):
        # Generates var-covar matrix
        weights_data = self.weightings()
        returns_df = pd.concat(
            [w[0].returns.rename(str(w[0])) for w in weights_data],
            axis=1,
            sort = False
        ).dropna()

        min_obs = 12  # at least a year of monthly data
        if len(returns_df) < min_obs:
            short_assets = [
                str(w[0]) for w in weights_data 
                if w[0].data.index[0] > returns_df.index.min()
            ]
            raise ValueError(
                f"Only {len(returns_df)} overlapping observations after alignment "
                f"(need ≥{min_obs}). Likely culprits with short history: {short_assets}. "
                f"Consider excluding these or using a shorter analysis window for the rest."
            )
        cov = returns_df.cov().values * 12

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e6:
            print(f"Warning: covariance matrix is near-singular "
                  f"(condition number={cond_number:.2e})."
                  f"Results from methods using matrix inversion (Kelly, eff_frontier) "
                  f"may be numerically unreliable. Consider removing near-duplicate assets."
                  )
        return cov

    def volatility(self):
        # Returns the volatility of a portfolio
        weights_data = self.weightings()
        weight_vector = np.array([w[1] for w in weights_data])
        
        self.portfolio_vol = np.sqrt(weight_vector @ self._cov_matrix() @ weight_vector)
        return self.portfolio_vol

    def sr(self):
        # Returns the sharpe ratio of a portfolio
        weights_data = self.weightings()

        portfolio_return = sum(w[1]*(getattr(w[0], 'expected_return', None) or w[0].returns_list()[2]) for w in weights_data)

        vol = getattr(self, 'portfolio_vol', None) or self.volatility()
        self.sr_value = (portfolio_return - market_data.rfr())/self.volatility()
        return self.sr_value

    def eff_frontier(self):
        # Returns the weights for portfolio to lie on efficient market frontier
        weights_data = self.weightings()
        n = len(weights_data)
        cov_matrix = self._cov_matrix()
        rf = market_data.rfr()

        mu = np.array([
            getattr(w[0], 'expected_return', None) or w[0].returns_list()[2]
            for w in weights_data
        ])

        def neg_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ cov_matrix @ w)
            return -(port_return - rf) / port_vol

        min_weight = 1 / (4*n)
        max_weight = (3/n)

        assert min_weight * n <= 1
        assert max_weight >= min_weight

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(min_weight, max_weight)] * n

        # Validate bounds are feasible — min_weight * n must not exceed 1
        if min_weight * n > 1:
            min_weight = 1 / (2 * n)
            bounds = [(min_weight, max_weight)] * n

        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )

        if not result.success:
            raise ValueError(f"Optimisation failed: {result.message}")

        optimal_weights = result.x
        optimal_return = optimal_weights @ mu
        optimal_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
        optimal_sr = (optimal_return - rf) / optimal_vol

        print(f"Tangency portfolio SR:     {optimal_sr:.4f}")
        print(f"Tangency portfolio return: {optimal_return:.4f}")
        print(f"Tangency portfolio vol:    {optimal_vol:.4f}")
        print(f"Current portfolio SR:      {self.sr():.4f}")

        return [(str(w[0]), round(optimal_weights[i], 4)) for i, w in enumerate(weights_data)]        

    def gmv_portfolio(self):
        # Returns weights for the global minimum variance portfolio
        weights_data = self.weightings()
        n = len(weights_data)
        cov_matrix = self._cov_matrix()
        rf = market_data.rfr()

        mu = np.array([
            getattr(w[0], 'expected_return', None) or w[0].returns_list()[2]
            for w in weights_data
        ])

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n

        result = minimize(
            lambda w: w @ cov_matrix @ w,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-12}
        )

        if not result.success:
            raise ValueError(f"GMV optimisation failed: {result.message}")

        gmv_weights = result.x
        gmv_return = gmv_weights @ mu
        gmv_vol = np.sqrt(gmv_weights @ cov_matrix @ gmv_weights)
        gmv_sr = (gmv_return - rf) / gmv_vol
        
        print(f"GMV portfolio SR: {gmv_sr:.4f}")
        print(f"GMV portfolio volatility: {gmv_vol:.4f}")
        print(f"GMV portfolio return: {gmv_return:.4f}")


        return [(str(w[0]), round(gmv_weights[i], 4)) for i, w in enumerate(weights_data)]

    def black_litterman(self, views=None, tau=0.05, bounded=True):
        # views: list of (ticker, expected_return, confidence) tuples
        # confidence in (0,1]; lower confidence = more uncertainty applied to that view
        weights_data = self.weightings()
        tickers = [str(w[0]) for w in weights_data]
        n = len(weights_data)
        cov_matrix = self._cov_matrix()
        rf = market_data.rfr()
        rm = market_data.mr()

        w_mkt = np.array([w[1] for w in weights_data])
        market_var = market_data.mreturns.var()*12
        delta = (rm - rf) / market_var
        pi = delta * cov_matrix @ w_mkt

        if not views:
            print("No views supplied — returning equilibrium (prior) returns only.")
            combined_returns = pi
        else:
            k = len(views)
            P = np.zeros((k, n))
            Q = np.zeros(k)
            confidences = np.zeros(k)
            for i, (ticker, view_return, confidence) in enumerate(views):
                P[i, tickers.index(ticker)] = 1
                Q[i] = view_return
                confidences[i] = confidence

            omega = np.diag([
                (tau * P[i] @ cov_matrix @ P[i].T) / max(confidences[i], 1e-6)
                for i in range(k)
            ])
            tau_cov_inv = np.linalg.inv(tau * cov_matrix)
            omega_inv = np.linalg.inv(omega)
            posterior_cov_inv = tau_cov_inv + P.T @ omega_inv @ P
            combined_returns = np.linalg.solve(
                posterior_cov_inv, tau_cov_inv @ pi + P.T @ omega_inv @ Q
            )

        # Quadratic utility maximisation — matches the reverse-optimisation FOC
        def neg_utility(w):
            return -(w @ combined_returns - (delta / 2) * (w @ cov_matrix @ w))

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(1/(4*n), 3/n)] * n if bounded else [(0, 1)] * n

        result = minimize(
            neg_utility, x0=w_mkt, method='SLSQP',
            bounds=bounds, constraints=constraints, options={'ftol': 1e-9}
        )

        if not result.success:
            raise ValueError(f"Black-Litterman optimisation failed: {result.message}")

        print(f"Implied risk aversion (delta): {delta:.4f}")
        for i, t in enumerate(tickers):
            print(f"  {t}: prior={pi[i]:.4f}, posterior={combined_returns[i]:.4f}")

        return [(tickers[i], round(result.x[i], 4)) for i in range(n)]

    def hrp(self):
        # Uses heirarchical risk parity to construct a portfolio where similar assets are clustered
        weights_data = self.weightings()
        tickers = [str(w[0]) for w in weights_data]

        returns_df = pd.concat(
            [w[0].returns.rename(str(w[0])) for w in weights_data],
            axis=1, sort=False
        ).dropna()

        corr = returns_df.corr()
        cov = returns_df.cov() * 12  # annualised, kept as DataFrame for label indexing

        dist = np.sqrt(0.5 * (1 - corr))
        condensed_dist = squareform(dist.values, checks=False)
        link = linkage(condensed_dist, method='average')

        def get_quasi_diag(link):
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i, j = df0.index, df0.values - num_items
                sort_ix[i] = link[j, 0]
                df1 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df1]).sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            return sort_ix.tolist()

        def get_cluster_var(cov, items):
            cov_slice = cov.loc[items, items]
            ivp = 1 / np.diag(cov_slice)
            ivp /= ivp.sum()
            w = ivp.reshape(-1, 1)
            return (w.T @ cov_slice.values @ w)[0, 0]

        def get_rec_bisection(cov, sort_ix):
            w = pd.Series(1.0, index=sort_ix)
            clusters = [sort_ix]
            while len(clusters) > 0:
                clusters = [
                    c[j:k] for c in clusters for j, k in
                    ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1
                ]
                for i in range(0, len(clusters), 2):
                    c0, c1 = clusters[i], clusters[i + 1]
                    var0 = get_cluster_var(cov, c0)
                    var1 = get_cluster_var(cov, c1)
                    alpha = 1 - var0 / (var0 + var1)
                    w[c0] *= alpha
                    w[c1] *= (1 - alpha)
            return w

        sorted_tickers = [tickers[i] for i in get_quasi_diag(link)]
        print("Cluster order:", sorted_tickers)
        hrp_weights = get_rec_bisection(cov, sorted_tickers)
        hrp_weights = (hrp_weights / hrp_weights.sum()).reindex(tickers)

        hrp_vol = np.sqrt(hrp_weights.values @ cov.values @ hrp_weights.values)
        print(f"HRP portfolio volatility: {hrp_vol:.4f}")

        return [(t, round(hrp_weights[t], 4)) for t in tickers]

    def kelly_criterion(self, fraction=0.5, long_only=True):
        weights_data = self.weightings()
        tickers = [str(w[0]) for w in weights_data]
        n = len(weights_data)
        cov_matrix = self._cov_matrix()
        rf = market_data.rfr()

        mu = np.array([
            getattr(w[0], 'expected_return', None) or w[0].returns_list()[2]
            for w in weights_data
        ])
        excess_returns = mu - rf

        if not long_only:
            kelly_weights = np.linalg.solve(cov_matrix, excess_returns) * fraction
            print(f"Raw Kelly weights sum to {kelly_weights.sum():.4f} (implied leverage if != 1)")
            return [(tickers[i], round(kelly_weights[i], 4)) for i in range(n)]

        # IMPORTANT: fraction must NOT appear anywhere below this line until after minimize()
        def neg_growth(w):
            port_return = w @ excess_returns
            port_var = w @ cov_matrix @ w
            return -(port_return - 0.5 * port_var)   # no 'fraction' here

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n

        result = minimize(
            neg_growth, x0=np.ones(n)/n, method='SLSQP',
            bounds=bounds, constraints=constraints, options={'ftol': 1e-9}
        )

        if not result.success:
            raise ValueError(f"Kelly optimisation failed: {result.message}")

        full_weights = result.x          # always 100% risky-asset allocation, fraction-independent
        final_weights = fraction * full_weights   # apply fraction AFTER solving

        cash_weight = 1 - final_weights.sum()
        print(f"{fraction*100:.0f}% Kelly: {final_weights.sum()*100:.1f}% in risky assets, "
              f"{cash_weight*100:.1f}% in cash/risk-free")

        return [(tickers[i], round(final_weights[i], 4)) for i in range(n)]

    def monte_carlo_simulation(self, n_portfolios=10000, seed=42):
        weights_data = self.weightings()
        tickers = [str(w[0]) for w in weights_data]
        n = len(weights_data)
        cov_matrix = self._cov_matrix()
        rf = market_data.rfr()

        mu = np.array([
            getattr(w[0], 'expected_return', None) or w[0].returns_list()[2]
            for w in weights_data
        ])

        rng = np.random.default_rng(seed)
        random_weights = rng.dirichlet(np.ones(n), size=n_portfolios)

        port_returns = random_weights @ mu
        port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))
        port_sharpes = (port_returns - rf) / port_vols

        results = pd.DataFrame({'return': port_returns, 'volatility': port_vols, 'sharpe': port_sharpes})

        best_sharpe_idx = results['sharpe'].idxmax()
        min_vol_idx = results['volatility'].idxmin()

        print(f"Simulated {n_portfolios} portfolios")
        print(f"Best Sharpe found:    {results.loc[best_sharpe_idx, 'sharpe']:.4f} "
              f"(return={results.loc[best_sharpe_idx, 'return']:.4f}, "
              f"vol={results.loc[best_sharpe_idx, 'volatility']:.4f})")
        print(f"Min volatility found: {results.loc[min_vol_idx, 'volatility']:.4f} "
              f"(return={results.loc[min_vol_idx, 'return']:.4f}, "
              f"sharpe={results.loc[min_vol_idx, 'sharpe']:.4f})")

        self.mc_results = results  # cached in case you want to plot it later

        return {
            'max_sharpe': [(tickers[i], round(random_weights[best_sharpe_idx, i], 4)) for i in range(n)],
            'min_volatility': [(tickers[i], round(random_weights[min_vol_idx, i], 4)) for i in range(n)],
            'all_results': results
        }
# Testing

AAPL = Asset('COST', market_data)
print(f'''Asset data:
{AAPL.data.head()}''')
print(f'''Current price of Asset:
{AAPL.current_price}''')
print(f'''Asset Returns:
{AAPL.returns}''')
print(f'Asset Beta: {AAPL.beta()}')
print(f'Asset Annual Volatility: {AAPL.volatility()}')
print(f'Asset Sharp Ratio: {AAPL.sr()}')
print(f'Asset Returns List: {AAPL.returns_list()}')
print(f'''Market Returns:
{market_data.mreturns}''')
print(f'Expected Market Return: {market_data.mr()}')
print(f'Trailing and forward P/E Ratios of Asset: {AAPL.pe()}')
print(f'Risk Free Rate: {market_data.rfr()}')

"""
Stock1 = Asset('NVDA',market_data)
print(Stock1.beta())
print(Stock1.volatility())
print(Stock1.returns_list())
Stock2 = Asset('AAPL',market_data)
print(Stock2.volatility())
print(Stock2.returns_list())
list1 = [('AAPL',1,290),('NVDA',2,200)]
portfolio1 = Portfolio(list1)
print(portfolio1.value())
print(portfolio1.weightings())
print(portfolio1.beta())
print(portfolio1.returns())
print(portfolio1.volatility())
print(portfolio1.sr())
print(portfolio1.weightings())
print(portfolio1.eff_frontier())
"""
"""
# 1. Pure mega-cap tech — high correlation, tests whether optimiser 
#    finds meaningful rebalancing when assets move together
tech_heavy = Portfolio([
    ('AAPL', 10, 165),
    ('MSFT', 5, 310),
    ('GOOGL', 8, 135),
    ('META', 4, 290)
])

# 2. Mixed sector — lower correlation between assets, should show 
#    the largest divergence between current and optimal weights
diversified = Portfolio([
    ('JPM', 6, 155),    # financials
    ('JNJ', 4, 165),    # healthcare
    ('NVDA', 3, 280),   # semiconductors
    ('XOM', 8, 110),    # energy
    ('AMZN', 5, 175)    # consumer/tech
])

# 3. High volatility — all volatile assets, tests whether optimiser 
#    finds a minimum variance weighting that meaningfully differs
high_vol = Portfolio([
    ('TSLA', 5, 220),
    ('NVDA', 4, 280),
    ('COIN', 6, 175),
    ('AMD', 8, 110)
])

# 4. Defensive — low beta assets, tests the floor on what 
#    the optimiser can achieve in terms of volatility reduction
defensive = Portfolio([
    ('KO', 10, 58),     # consumer staples
    ('PG', 8, 145),     # consumer staples
    ('VZ', 12, 40),     # telecom
    ('GLD', 5, 175)     # gold ETF
])

# 5. Deliberately unbalanced — one position dominates by value,
#    optimiser should aggressively rebalance toward smaller positions
unbalanced = Portfolio([
    ('NVDA', 50, 280),  # dominates portfolio
    ('AAPL', 2, 165),
    ('MSFT', 2, 310)
])
"""

portfolio1 = Portfolio([('NVDA',15,206.22),('IVV',10,659.01),('MSFT',5,418.68),('QQQM',11,255.68),('VGT',43,124.64),('LITE',1,870.00),('VOO',1,697.50)])


portfolio2 = Portfolio([('GOOGL',0.14491,362.91),('INTC',0.26395,113.49),('LITE',0.10196,898.30),('META',0.03819,614.36),('MSFT',0.31975,414.57),('MU',0.0287,815.95),('NVDA',0.42333,208.88),('PEP',0.18853,154.27),('QQQ',0.05459,715.64),('SMH',0.10667,595.19),('SOFI',2,16.12),('TSLA',0.11447,382.77)])

"""
# Testing
portfolios = {
    'tech_heavy': tech_heavy,
    'diversified': diversified,
    'high_vol': high_vol,
    'defensive': defensive,
    'unbalanced': unbalanced,
    'portfolio1': portfolio1,
    'portfolio2': portfolio2
}

for name, portfolio in portfolios.items():
    print(f"\n{'='*40}")
    print(f"Portfolio: {name}")
    print(f"Current weights:  {[(str(w[0]), round(w[1], 4)) for w in portfolio.weightings()]}")
    print(f"Current SR:       {portfolio.sr():.4f}")
    print(f"Current vol:      {portfolio.volatility():.4f}")
    print(f"Optimal weights:  {portfolio.eff_frontier()}")
    print(f"Portfolio value: {portfolio.value()}")
"""
"""
weights_data = portfolio2.weightings()
for w in weights_data:
    r = w[0].returns_list()
    vol = w[0].volatility()
    print(f"{w[0]}: CAPM_return={r[2]:.4f}, vol={vol:.4f}, SR={(r[2]-market_data.rfr())/vol:.4f}")
returns_df = pd.concat(
    [w[0].returns.rename(str(w[0])) for w in portfolio2.weightings()],
    axis=1, sort=False
).dropna()

corr_matrix = returns_df.corr()
print(corr_matrix.round(2))

# Specifically check what's pulling PEP, INTC, LITE high
print(corr_matrix['PEP'].sort_values())
print(corr_matrix['QQQ'].sort_values(ascending=False))

# Confirm the INTC hypothesis directly
print(f"INTC vs NVDA correlation: {corr_matrix.loc['INTC','NVDA']:.3f}")
print(f"INTC vs MU correlation:   {corr_matrix.loc['INTC','MU']:.3f}")
print(f"INTC standalone return:    {[w[0] for w in portfolio2.weightings() if str(w[0])=='INTC'][0].returns_list()[1]:.4f}")

returns_df = pd.concat(
    [w[0].returns.rename(str(w[0])) for w in portfolio2.weightings()],
    axis=1, sort=False
).dropna()

print(returns_df.shape)   # if this prints something like (2, 14), that's the bug
print(returns_df)
"""
"""
print("portfolio 2")
print(portfolio2.sr())
print(portfolio2.volatility())
print(portfolio2.eff_frontier())
print(portfolio2.gmv_portfolio())
print(portfolio2.black_litterman())
print(portfolio2.hrp())
print(portfolio2.kelly_criterion())
print(portfolio2.monte_carlo_simulation())

print("portfolio 1")
print(portfolio1.sr())
print(portfolio1.volatility())
print(portfolio1.eff_frontier())
print(portfolio1.gmv_portfolio())
print(portfolio1.black_litterman())
print(portfolio1.hrp())
print(portfolio1.kelly_criterion())
print(portfolio1.monte_carlo_simulation())

for w in portfolio1.weightings():
    r = w[0].returns_list()
    print(f"{w[0]}: beta={w[0].beta():.3f}, CAPM_return={r[2]:.4f}, vol={w[0].volatility():.4f}")

portfolio1_clean = Portfolio([
    ('VOO',  10, 480),   # S&P 500 core — remove IVV
    ('VGT',  43, 124),   # tech tilt — remove QQQM
    ('NVDA', 15, 206),   # individual semi
    ('MSFT',  5, 418),   # individual tech
    ('LITE',  1, 870),   # genuinely independent
])

print(portfolio1_clean.sr())
print(portfolio1_clean.volatility())
print(portfolio1_clean.eff_frontier())
print(portfolio1_clean.gmv_portfolio())
print(portfolio1_clean.black_litterman())
print(portfolio1_clean.hrp())
print(portfolio1_clean.kelly_criterion())
print(portfolio1_clean.monte_carlo_simulation())

for w in portfolio1_clean.weightings():
    print(f"{w[0]}: vol={w[0].volatility():.4f}")
cov = portfolio1_clean._cov_matrix()
eigenvalues = np.linalg.eigvalsh(cov)
print(eigenvalues.round(6))
print(f"Condition number: {np.linalg.cond(cov):.2e}")
print(f"Effective rank: {(eigenvalues > 0.001).sum()}")

returns_df = pd.concat(
    [w[0].returns.rename(str(w[0])) for w in portfolio1_clean.weightings()],
    axis=1, sort=False
).dropna()
print(returns_df.corr().round(3))

# Candidate additions — check correlations with VOO before deciding
candidates = ['GLD',  # gold — historically ~0.0 to 0.1 with equities
              'TLT',  # 20yr treasuries — historically negative with equities
              'VNQ',  # REITs — partial diversification (~0.6 with S&P)
              'BIL']  # short-term bills — near-zero correlation, low vol

for ticker in candidates:
    test = Asset(ticker, market_data)
    combined = pd.concat([test.returns, 
                         portfolio1_clean.weightings()[0][0].returns], 
                         axis=1, sort=False).dropna()
    combined.columns = [ticker, 'VOO']
    print(f"{ticker}: corr with VOO={combined.corr().loc[ticker,'VOO']:.3f}, "
          f"vol={test.volatility():.4f}")
"""
