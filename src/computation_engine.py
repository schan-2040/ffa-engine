import sys
from datetime import datetime, timedelta

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas_datareader import data
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
from tqdm import tqdm

SNS_THEME = sns.axes_style("darkgrid")


class FFAEngine:
    # Collected data
    tickerData = None
    rawTickerData = None
    factorData = None
    term_structure = None

    # Window parameters
    covariance_lookback_period = None
    expected_returns_lookback_period = None
    max_lookback = None
    iteration_date = None

    # Date limits
    start_date = None
    end_date = None
    crisis_startdate = datetime(2007, 12, 31)
    crisis_enddate = datetime(2009, 6, 30)

    # Optimization parameters
    target_beta = None
    security_betas = None
    previous_weights = None
    optimized_weight_collection = []
    optimized_weight_DF = []
    raw_date_window = []
    date_window = []
    P = None
    q = None
    A = None
    b = None
    G = None
    h = None
    n = None

    figure_counter = None

    # Tearsheet
    performance_DF = None
    individual_performance_DF = None

    # Hyperparameter
    lambda_hyperparameter = 1e-5

    # Expected returns
    regression_DF = None
    term_structure_returns_array = None
    expected_returns_matrix = None
    expected_returns_covariance_matrix = None
    omega_covariance = None

    def __init__(self, tickerList, start_date, end_date, factorDirectory, covariance_lookback_period,
                 expected_returns_lookback_period, term_structure, target_beta):
        # Parse user input to collect data
        print("Collecting ticker data")
        tickerData = data.get_data_yahoo(tickerList, start_date, end_date)
        tickerData = tickerData['Adj Close']

        print("Collecting french farma factor data")
        factorData = pd.read_csv(factorDirectory, names=[
            'Date', 'Mkt-RF', 'SMB', 'HML', 'RF']).iloc[1:, :]

        # Scrub data to get ready for regression
        self.tickerData, self.factorData = self.dataScrubbing(
            tickerData, factorData)

        # Store findings into class attributes
        self.rawTickerData = tickerData
        self.target_beta = target_beta
        self.covariance_lookback_period = covariance_lookback_period
        self.expected_returns_lookback_period = expected_returns_lookback_period
        self.max_lookback = max(
            covariance_lookback_period, expected_returns_lookback_period)
        self.start_date = start_date
        self.end_date = end_date
        self.term_structure = term_structure
        self.figure_counter = 1

    def dataScrubbing(self, tickerData, factorData):
        """
        Function used for performing regression to obtain B_3, B_SMB, and B_HML for each day
        """

        # Set index to be the dates themselves for the ticker data
        tickerData = tickerData.sort_values(by='Date', ascending=True)
        tickerData.reset_index(inplace=True)
        tickerData.columns.name = None
        tickerData = tickerData.set_index('Date')

        # Convert data types to floating point
        factorData['Mkt-RF'] = factorData['Mkt-RF'].astype('float64')
        factorData['SMB'] = factorData['SMB'].astype('float64')
        factorData['HML'] = factorData['HML'].astype('float64')
        factorData['RF'] = factorData['RF'].astype('float64')

        # Convert improper date to pandas date format
        factorData['Date'] = pd.to_datetime(
            factorData['Date'], format='%Y%m%d')

        # Since FF data is in percents, divide by 100 to scale back to decimals
        factorData['Mkt-RF'] = factorData['Mkt-RF'] / 100
        factorData['SMB'] = factorData['SMB'] / 100
        factorData['HML'] = factorData['HML'] / 100
        factorData['RF'] = factorData['RF'] / 100

        # Set index to be the dates themselves for the factor data
        factorData.sort_values(by='Date', inplace=True, ascending=True)
        factorData.reset_index(inplace=True)

        # Remove top row to align dates between ticker data and factor data
        factorData = factorData.iloc[:, 1:]
        factorData = factorData.set_index('Date')

        print("Aligning dates")
        factorData = pd.merge(factorData, tickerData,
                              how='inner', on='Date').iloc[:, :4]

        print("Dates aligned: ", tickerData.index.equals(factorData.index))

        # Superimposing dates from tickerData to factorData to ensure equal row lenghts
        # Dependent variable
        tickerData = tickerData.pct_change().iloc[1:, :]

        # Independent variables
        factorData = factorData.iloc[1:, :]

        # Optimization requirement: Setting all weights equal to each other
        n = tickerData.columns.shape[0]
        weights = 1 / n
        self.previous_weights = np.repeat(weights, n)

        return tickerData, factorData

    def ffRegressor(self, asset_data):
        if (self.term_structure == 'pre-crisis'):
            asset_data = asset_data.loc[self.start_date: (
                self.crisis_startdate - timedelta(days=1))]
            start_date = self.start_date
            end_date = (self.crisis_startdate - timedelta(days=1))
        elif (self.term_structure == 'in-crisis'):
            asset_data = asset_data.loc[self.crisis_startdate: (
                self.crisis_enddate - timedelta(days=1))]
            start_date = self.crisis_startdate
            end_date = (self.crisis_enddate - timedelta(days=1))
        elif (self.term_structure == 'post-crisis'):
            asset_data = asset_data.loc[self.crisis_enddate: (
                self.end_date - timedelta(days=1))]
            start_date = self.crisis_enddate
            end_date = (self.end_date - timedelta(days=1))
        elif (self.term_structure == 'full-timeline'):
            asset_data = asset_data.loc[
                self.tickerData.index[0].to_pydatetime(): (self.tickerData.index[-1].to_pydatetime())]
            start_date = self.tickerData.index[0].to_pydatetime()
            end_date = self.tickerData.index[-1].to_pydatetime()
        else:
            print("Unknown term structure")
            exit(-1)

        risk_free_rate = np.asarray(
            self.factorData.loc[start_date: end_date, 'RF']).reshape(-1, 1)

        ff_factors = np.asarray(
            self.factorData.loc[start_date: end_date, self.factorData.columns != 'RF'])

        asset_data = np.asarray(asset_data).reshape(-1, 1)
        asset_data = asset_data - risk_free_rate

        model = LinearRegression()
        model.fit(X=ff_factors, y=asset_data)

        beta_vector = model.coef_
        alpha = model.intercept_

        model_data = np.concatenate((alpha, beta_vector), axis=None).ravel()
        model_data = pd.Series(
            model_data, index=['Alpha', 'B_3', 'B_SMB', 'B_HML'])

        return model_data

    def startFFRegression(self):
        regression_DF = self.tickerData.apply(self.ffRegressor, axis=0)
        self.regression_DF = regression_DF

    def generateReturns(self):
        start_date = max(self.start_date + timedelta(days=1),
                         self.iteration_date - timedelta(days=self.max_lookback))
        end_date = (self.iteration_date - timedelta(days=1))

        regression_Stats = self.regression_DF.transpose()
        beta_array = np.asarray(regression_Stats.iloc[:, 1:])
        alpha_array = np.asarray(
            regression_Stats.loc[:, 'Alpha']).reshape(-1, 1)

        risk_free_rate = np.asarray(
            self.factorData.loc[start_date: end_date, 'RF']).reshape(-1, 1)
        factor_matrix = np.asarray(
            self.factorData.loc[start_date: end_date, self.factorData.columns != 'RF'])

        return_array = []
        num_iterations = beta_array.shape[0]

        for i in range(num_iterations):
            alpha_val = alpha_array[i, :]
            beta_vector = beta_array[i, :].reshape(-1, 1)

            return_vector = risk_free_rate + \
                (factor_matrix @ beta_vector) + alpha_val
            return_array.append(return_vector.ravel())

        return_array = np.asarray(return_array).transpose()
        if (return_array.shape[0] <= self.expected_returns_lookback_period):
            expected_returns_matrix = return_array[-return_array.shape[0]:, :]
        else:
            expected_returns_matrix = return_array[-self.expected_returns_lookback_period:, :]

        self.term_structure_returns_array = return_array
        self.expected_returns_matrix = expected_returns_matrix

    def generateFactorCovariance(self):
        start_date = max(self.start_date + timedelta(days=1),
                         self.iteration_date - timedelta(days=self.max_lookback))
        end_date = (self.iteration_date - timedelta(days=1))

        factor_matrix = np.asarray(
            self.factorData.loc[start_date: end_date, self.factorData.columns != 'RF'])

        if (factor_matrix.shape[0] <= self.covariance_lookback_period):
            factor_matrix = factor_matrix[-factor_matrix.shape[0]:, :]
        else:
            factor_matrix = factor_matrix[-self.covariance_lookback_period:, :]

        factor_matrix = factor_matrix.transpose()

        factor_covariance = np.cov(factor_matrix)

        self.omega_covariance = factor_covariance

    def generateReturnsCovariance(self):
        start_date = max(self.start_date + timedelta(days=1),
                         self.iteration_date - timedelta(days=self.max_lookback))
        end_date = (self.iteration_date - timedelta(days=1))

        omega = self.omega_covariance
        beta_matrix = np.array(self.regression_DF.transpose().iloc[:, 1:])
        variance_array = np.array(
            self.tickerData.loc[start_date: end_date, :].var(axis=0)).ravel()
        variance_matrix = np.diag(variance_array)

        covariance_matrix = (beta_matrix @ omega @
                             beta_matrix.transpose()) + variance_matrix
        self.expected_returns_covariance_matrix = covariance_matrix

    def generateSecurityBetas(self, asset_data):
        if (self.term_structure == 'pre-crisis'):
            asset_data = asset_data.loc[self.start_date: (
                self.crisis_startdate - timedelta(days=1))]
            start_date = self.start_date
            end_date = (self.crisis_startdate - timedelta(days=1))
        elif (self.term_structure == 'in-crisis'):
            asset_data = asset_data.loc[self.crisis_startdate: (
                self.crisis_enddate - timedelta(days=1))]
            start_date = self.crisis_startdate
            end_date = (self.crisis_enddate - timedelta(days=1))
        elif (self.term_structure == 'post-crisis'):
            asset_data = asset_data.loc[self.crisis_enddate: (
                self.end_date - timedelta(days=1))]
            start_date = self.crisis_enddate
            end_date = (self.end_date - timedelta(days=1))
        elif (self.term_structure == 'full-timeline'):
            asset_data = asset_data.loc[
                self.tickerData.index[0].to_pydatetime(): (self.tickerData.index[-1].to_pydatetime())]
            start_date = self.tickerData.index[0].to_pydatetime()
            end_date = self.tickerData.index[-1].to_pydatetime()
        else:
            print("Unknown term structure")
            exit(-1)

        asset_data = np.asarray(asset_data)

        mkt_rf = np.asarray(
            self.factorData.loc[start_date: end_date, 'Mkt-RF'])

        numerator = np.cov(asset_data, mkt_rf).ravel()[1]
        denominator = mkt_rf.var()

        return numerator / denominator

    def generateOptimizationComponents(self):
        P = self.expected_returns_covariance_matrix
        q = np.array(pd.DataFrame(
            self.expected_returns_matrix).cumsum(axis=0).sum())
        n = self.expected_returns_covariance_matrix.shape[0]

        self.security_betas = self.tickerData.apply(
            self.generateSecurityBetas, axis=0)
        self.P = P
        self.q = q
        self.n = n

    def rebalancePortfolio(self):
        P = self.P
        q = self.q
        n = self.n

        security_betas = np.asarray(self.security_betas)
        target_beta = self.target_beta
        lambda_hyperparameter = self.lambda_hyperparameter

        w_p = self.previous_weights.reshape(-1, 1)

        w = cp.Variable((n, 1))

        rho_p = q.T @ w
        beta_p = security_betas @ w

        if (self.first_iteration == False):
            penalty = w - w_p
        else:
            penalty = w_p

        volatility = cp.quad_form(penalty, P)

        problem = cp.Problem(
            cp.Maximize(rho_p - (lambda_hyperparameter * volatility)),
            [
                cp.sum(w) == 1,
                -2 <= w,
                w <= 2,
                beta_p == target_beta
            ]
        )

        try:
            problem.solve(solver='CVXOPT')

            weights = [float('%0.10f' % v) for v in w.value]
            weights = np.asarray(weights).astype('float')

            return weights

        except cp.SolverError:
            print("Solver failed to converge, reusing previous iteration values")
            return self.previous_weights

    def startOptimization(self):
        weights = self.rebalancePortfolio()
        self.previous_weights = weights.ravel()

        return weights

    def iteratePortfolio(self):
        if (self.term_structure == 'pre-crisis'):
            start_date = self.start_date
            end_date = (self.crisis_startdate - timedelta(days=1))
        elif (self.term_structure == 'in-crisis'):
            start_date = self.crisis_startdate
            end_date = (self.crisis_enddate - timedelta(days=1))
        elif (self.term_structure == 'post-crisis'):
            start_date = self.crisis_enddate
            end_date = (self.end_date - timedelta(days=1))
        elif (self.term_structure == 'full-timeline'):
            start_date = self.tickerData.index[0].to_pydatetime()
            end_date = self.tickerData.index[-1].to_pydatetime()
        else:
            print("Unknown term structure")
            exit(-1)

        # date_window = self.tickerData.iloc[5:, :]
        date_window = self.tickerData.loc[start_date: end_date, :]
        date_window = date_window.iloc[5:, :]

        date_list = date_window.index
        counter = 0

        raw_date_window_list = []
        date_window_list = []

        self.first_iteration = True
        for date in tqdm(date_list, ncols=100, file=sys.stdout):
            self.iteration_date = date.to_pydatetime()
            self.generateReturns()
            self.generateFactorCovariance()
            self.generateReturnsCovariance()
            self.generateOptimizationComponents()
            weights = self.startOptimization()
            self.first_iteration = False
            if (counter == 5):
                self.optimized_weight_collection.append(weights)
                raw_date_window_list.append(
                    self.rawTickerData.loc[self.iteration_date, :])
                date_window_list.append(
                    self.tickerData.loc[self.iteration_date, :])
                counter = 0
            else:
                counter += 1

        validation_weights = self.optimized_weight_collection[-1]
        print("Checking optimization weights against constraints")

        print("Sum of weights: ", np.asarray(validation_weights).sum())
        print("∑𝛽𝑖𝑤𝑖=𝛽𝑇:\t", np.asarray(self.security_betas)
              @ np.asarray(validation_weights))

        self.raw_date_window = pd.DataFrame(raw_date_window_list)
        self.raw_date_window.columns.name = None
        self.raw_date_window.index.name = 'Date'

        self.date_window = pd.DataFrame(raw_date_window_list)
        self.date_window.columns.name = None
        self.date_window.index.name = 'Date'

        self.optimized_weight_collection = np.asarray(
            self.optimized_weight_collection)

        self.optimized_weight_DF = pd.DataFrame(
            np.asarray(self.optimized_weight_collection))
        self.optimized_weight_DF.index = pd.DataFrame(
            self.raw_date_window).index
        self.optimized_weight_DF.columns = self.tickerData.columns

    def invididualPerformance(self, row, weights):
        return row * weights.ravel()

    def createTearSheet(self):
        names = self.optimized_weight_DF.columns.values
        market_daily_returns = pd.DataFrame(
            self.tickerData.loc[:, 'SPY']).iloc[1:, :]

        final_weights = self.optimized_weight_collection[-1].reshape(-1, 1)

        performance_matrix = self.tickerData @ final_weights
        performance_DF = pd.DataFrame(performance_matrix)
        performance_DF.index = self.tickerData.index
        performance_DF.columns = ['Returns']

        individual_performance_matrix = np.apply_along_axis(
            self.invididualPerformance, 1, self.tickerData, weights=final_weights)
        individual_performance_DF = pd.DataFrame(individual_performance_matrix)
        individual_performance_DF.index = self.tickerData.index
        individual_performance_DF.columns = names

        # PnL
        performance_pnl = (((1 + performance_DF).cumprod() - 1).mean() * 250)

        # volatility
        performance_volatility = performance_DF.std()

        # mean return
        performance_mean = performance_DF.mean() * 250

        # kurtosis
        performance_kurtosis = performance_DF.kurtosis()

        # sharpe ratio
        performance_sharpe = performance_DF.mean() / performance_DF.std()

        # var
        daily_percentage_change = performance_DF.copy()
        daily_percentage_change.sort_values(
            by="Returns", inplace=True, ascending=True)

        VaR_95 = daily_percentage_change.quantile(0.05)

        # cvar
        CVaR_95 = daily_percentage_change[daily_percentage_change <= VaR_95].mean(
        )

        # Plotting
        with SNS_THEME:
            figure, axList = plt.subplots(ncols=2, figsize=(16, 8))
            plot1 = performance_DF.cumsum().plot(ax=axList[0])
            plot1.set_title("Cumulative Returns", fontsize=14)

            plot1a = market_daily_returns.cumsum().plot(ax=axList[0])
            axList[0].legend(['Portfolio', 'Market (S&P 500)'])

            plot1b = performance_DF.plot(kind='hist', ax=axList[1])
            plot1b.set_title("Distribution of Returns", fontsize=14)

            plt.savefig(
                './out/figure1_{}.png'.format(self.figure_counter), dpi=250)

        with SNS_THEME:
            figure, axList = plt.subplots(ncols=2, figsize=(15, 7))
            plot2 = individual_performance_DF.cumsum().plot(ax=axList[0])
            plot2.set_title("Individual Returns (Portfolio)", fontsize=14)

            plot2a = self.tickerData.cumsum().plot(ax=axList[1])
            plot2a.set_title("Invididual Returns (Baseline)", fontsize=14)

            plt.savefig(
                './out/figure2_{}.png'.format(self.figure_counter), dpi=250)

        with SNS_THEME:
            figure, ax = plt.subplots(figsize=(16, 7))
            plot3 = self.optimized_weight_DF.plot(ax=ax)
            plot3.set_title("Portfolio Weight Changes", fontsize=14)

            plt.savefig(
                './out/figure3_{}.png'.format(self.figure_counter), dpi=250)

        report_string = tabulate(
            [
                ['PnL', performance_pnl.values],
                ['Daily Mean', performance_mean.values],
                ['Volatility', performance_volatility.values],
                ['Kurtosis', performance_kurtosis.values],
                ['Sharpe Ratio', performance_sharpe.values],
                ['VaR 95%', VaR_95.values],
                ['CVaR 95%', CVaR_95.values]
            ],
            headers=['Performance Summary', 'S({} {}) 𝛽𝑇({})'.format(
                self.covariance_lookback_period, self.expected_returns_lookback_period, self.target_beta)]
        )

        # Print tearsheet
        print(report_string)

        with open('./out/report_{}.txt'.format(self.figure_counter), 'w', encoding="utf-8") as file:
            file.write(str(report_string))

        self.figure_counter += 1

        self.performance_DF = performance_DF
        self.individual_performance_DF = individual_performance_DF

    def start(self):
        self.optimized_weight_collection = []
        self.security_betas = None
        self.previous_weights = None
        self.optimized_weight_collection = []
        self.optimized_weight_DF = []
        self.raw_date_window = []
        self.date_window = []
        self.P = None
        self.q = None
        self.A = None
        self.b = None
        self.G = None
        self.h = None
        self.n = None
        self.performance_DF = None
        self.individual_performance_DF = None
        self.regression_DF = None
        self.term_structure_returns_array = None
        self.expected_returns_matrix = None
        self.expected_returns_covariance_matrix = None
        self.omega_covariance = None
        self.iteration_date = None
        self.first_iteration = True

        n = self.tickerData.columns.shape[0]

        weights = 1 / n
        self.previous_weights = np.repeat(weights, n)

        self.max_lookback = max(
            self.covariance_lookback_period, self.expected_returns_lookback_period)

        self.startFFRegression()
        self.iteratePortfolio()

        self.createTearSheet()
