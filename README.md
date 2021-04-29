# Forecasting-trade-direction-an-example-with-LSTM-neural-network

<p align="justify">In this repository, a study on forecasting trade direction of a stock, from tick data is carried out using two different models:</p>

<ul>
      <li><div align="justify"><code>Linear regression</code> which is used as an example to show it's non application in such problem. As a regression model, </div></li>
    <li><div align="justify"></li>
</ul>

<p align="justify">Based on the information provided by the order book, different features such as  to capture the imbalance between buy and sell orders, which will drive the price to move up or down.</p>

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
