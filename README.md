This repository is my own personal project tinkering around with yfinance and various financial mathematics concepts. 

Version 1 Main Features:
Market Class
- Tracks the US Market using the S&P500 as a proxy
- Returns key information such as risk free rate of return and expected market return

Stock Class
- Tracks an individual stock
- Returns information such as sharpe ratio, volatility, expected return, beta value

Portfolio Class
- Tracks multiple stocks in a user-generated portfolio
- Returns portfolio-wide information such as sharpe ratio, voltaility, expected return, beta value
- Includes optimiser functions that suggest weights depending on which stocks are included
- Optimisers include:
    - Efficient Frontier
    - Global Minimum Variance
    - Black-Litterman
    - Heirarchical Risk Parity
    - Kelly Criterion
    - Monte Carlo (Max return, Min Variance)

Version 2 is currently in development, with upcoming changes to:
- yfinance data calling will be done once at the start of the program running to increase efficiency
- Stock Class will gain an additional subfunction indicating if a stock if over or underpriced, and high or low quality based on factors such as P/E etc.
- Optimisers may be further improved
- Potential integration into excel or a web-based app for ease of use

Constructive Criticism is always welcome. 
