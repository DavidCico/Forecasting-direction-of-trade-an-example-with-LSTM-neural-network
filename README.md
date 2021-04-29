# Forecasting-trade-direction-an-example-with-LSTM-neural-network

<p align="justify">In this repository, a study on forecasting trade direction of a stock, from tick data is carried out using two different models:</p>

<ul>
      <li><div align="justify"><code>Linear regression</code> which is used as an example to show it's non application in such problem. As a regression model, it's prediction is based on a quantity, and then on thresholds rather than classifying the result with a label.</div></li>
      <li><div align="justify"><code>LSTM neural network</code> as an example in this study, which shows good results when applied with a large set of features.</li>
</ul>

<p align="justify">Based on the information provided by the order book, different important features such as <i>Volume Order Imbalance, Bid Ask spread, Mid-price basis, etc</i> are computed to capture the imbalance between buy and sell orders, that will drive the price to move up or down.</p>

<p align="justify">As seen in the full study report '<em>Predicting tick's direction of a stock.pdf</em>' the LSTM neural network has been trained and tested in several conditions, to find which features are the most important to be fed to the model, as well as the lookback period for predicting next days of prices.</p>

## Getting Started

<p align="justify">These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.</p>

### Prerequisites

<p align="justify">You need <strong>Python 3.x</strong> to run the following code.  You can have multiple Python versions (2.x and 3.x) installed on the same system without problems. Python needs to be first installed then <strong>SciPy</strong> and <strong>pymysql</strong> as there are dependencies on packages.</p>

In Ubuntu, Mint and Debian you can install Python 3 like this:

    sudo apt-get install python3 python3-pip

Alongside Python, the SciPy packages are also required. In Ubuntu and Debian, the SciPy ecosystem can be installed by:

    sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
    
The latest release of Scikit-Learn machine learning package, which can be installed with pip:
    
    pip install -U scikit-learn

Finally, the Tensorflow package for neural networks modelling, which can be installed with pip:
    
    pip install tensorflow

For other Linux flavors, OS X and Windows, packages are available at:

http://www.python.org/getit/  
https://www.scipy.org/install.html <br>
https://scikit-learn.org/stable/install.html


### File descriptions
<ul>
  
<li><div align="justify">'<em>BacktesterLoop.py</em>' in which the Backtest class hierarchy encapsulates the other classes, to carry out a nested while-loop event-driven system in order to handle the events placed on the Event Queue object.</div></li>
    
<li><div align="justify">'<em>DataHandler.py</em>' which defines a class that gives all subclasses an interface for providing market data to the remaining components within the system. Data can be obtained directly from the web, a database or be read from CSV files for instance.</div></li>

<li><div align="justify">'<em>Events.py</em>' with four types of events (market, signal, order and fill), which allow communication between the above components via an event queue, are implemented.</div></li>

<li><div align="justify">'<em>Execution.py</em>' to simulate the order handling mechanism and ultimately tie into a brokerage or other
means of market connectivity.</div</li>

<li><div align="justify">'<em>Main.py</em>' which is the main Python program, englobing all the different subroutines, and where the different parameters to initialize the backtesting simulations are specified.</div</li>

<li><div align="justify">'<em>Performance.py</em>' in which performance assessment criteria are implemented such as the Sharpe ratio and drawdowns.</div</li>
  
<li><div align="justify">'<em>PlotPerformance.py</em>' to plot figures based on the equity curve obtained after backtesting.</div</li>
  
<li><div align="justify">'<em>Portfolio.py</em>' that keeps track of the positions within a portfolio, and generates orders of a fixed quantity of stock based on signals.</div></li>

<li><div align="justify">'<em>RiskManagement.py</em>' which would be the class for implementing risk management measures, as its name suggests such as VaR calculation, Kelly criterion for position sizing, etc.</div></li>

<li><div align="justify">'<em>Strategy.py</em>' to generate a signal event from a particular strategy to communicate to the portfolio.</div></li>

<li><div align="justify">In the '<em>Strategies</em>' directory, different trading strategies are implemented to be used for backtesting:</div></li>

## Contributing

Please read [CONTRIBUTING.md](https://github.com/DavidCico/Forecasting-direction-of-trade-an-example-with-LSTM-neural-network/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **David Cicoria** - *Initial work* - [DavidCico](https://github.com/DavidCico)

See also the list of [contributors](https://github.com/DavidCico/Forecasting-direction-of-trade-an-example-with-LSTM-neural-network/graphs/contributors) who participated in this project.
