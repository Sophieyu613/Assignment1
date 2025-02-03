import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from datetime import datetime


# Q4(a) YTM Curve

file_path = "/Users/apple/Desktop/f/bond_prices.xlsx"


def load_bond_data(file_path):

    df = pd.read_excel(file_path)
    df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
    df['Years to Maturity'] = (df['Maturity Date'] - pd.to_datetime("2025-01-06")).dt.days / 365
    df['Days since last coupon'] = df['Maturity Date'].apply(get_days_since_last_coupon)
    df['Dirty Price'] = df.apply(lambda row: compute_dirty_price(row['Price'], row['Days since last coupon'], row['Coupon']), axis=1)
    return df


def get_days_since_last_coupon(date):

    if date.month < 3:
        last_coupon_date = datetime(date.year - 1, 9, 1)
    elif date.month > 9:
        last_coupon_date = datetime(date.year, 9, 1)
    else:
        last_coupon_date = datetime(date.year, 3, 1)

    return (date - last_coupon_date).days


def compute_dirty_price(clean_price, days_since_last_coupon, coupon_rate):

    # Dirty Price = Clean Price + Accrued Interest
    # Accrued Interest = (days_since_last_coupon / 365) * coupon_rate
    # Assume face value is $100

    accrued_interest = (days_since_last_coupon / 365) * coupon_rate * 100
    return clean_price + accrued_interest


def compute_ytm(dirty_price, coupon_rate, years_to_maturity):
    
    # P = Σ (C / (1 + YTM/2)^(2t)) + F / (1 + YTM/2)^(2n)
    # Semi-annual payments: periods = years_to_maturity * 2
    # Use fsolve to calculate YTM

    if years_to_maturity < 0.5:
        return -np.log(dirty_price / (100 + coupon_rate * 100 / 2)) / years_to_maturity

    coupon_payment = 100 * coupon_rate / 2
    periods = int(years_to_maturity * 2)

    def bond_price_equation(ytm):
        total_pv = sum([coupon_payment / (1 + ytm / 2) ** (2 * t) for t in range(1, periods + 1)])
        total_pv += 100 / (1 + ytm / 2) ** (2 * years_to_maturity)
        return total_pv - dirty_price

    return fsolve(bond_price_equation, 0.05)[0]


def calculate_ytm_curve(df):

    df['YTM'] = df.apply(lambda row: compute_ytm(row['Dirty Price'], row['Coupon'], row['Years to Maturity']), axis=1)
    return df


def plot_ytm_curve(df):

    # X axis:Years to Maturity
    # Y axis：YTM
    # Each curve represents different trading days

    plt.figure(figsize=(10, 6))
    for date in df['Date'].unique():
        temp_df = df[df['Date'] == date]
        plt.plot(temp_df['Years to Maturity'], temp_df['YTM'], marker='o', label=f"{date}")
    plt.xlabel('Years to Maturity')
    plt.ylabel('Yield to Maturity (YTM)')
    plt.title('Yield Curve')
    plt.legend()
    plt.grid()
    plt.show()


df = load_bond_data("bonds-data.xlsx")

df = calculate_ytm_curve(df)
plot_ytm_curve(df)



#Q4(b) Spot Curve

def compute_spot_rates(df):
    spot_rates = {}
    for _, row in df.iterrows():
        t = row['Years until Maturity']
        price = row['Price']
        coupon = row['Coupon']
        
        if t not in spot_rates:
            spot_rates[t] = -np.log(price / (100 + coupon * 100 / 2)) / t
    return spot_rates

def plot_spot_curve(spot_rates):
    cmap = plt.get_cmap('tab10')
    for i, date in enumerate(spot_rates):
        years_until_maturity = list(spot_rates[date].keys())
        spot_values = list(spot_rates[date].values())
        plt.plot(years_until_maturity, spot_values, label=date, marker='o', linestyle='-', color=cmap(i / len(spot_rates)))
    
    plt.xlabel('Years until Maturity')
    plt.ylabel('Spot Rate')
    plt.title('Spot Curve')
    plt.legend()
    plt.show()



#Q4(c) Forward Curve

def compute_forward_rates(spot_rates):
    forward_rates = {}
    for date, rates in spot_rates.items():
        forward_rates_date = {}
        sorted_times = sorted(rates.keys())
        for i in range(1, len(sorted_times)):
            t1 = sorted_times[i - 1]
            t2 = sorted_times[i]
            r1 = rates[t1]
            r2 = rates[t2]
            forward_rate = ((1 + r2) ** t2 / (1 + r1) ** t1) ** (1 / (t2 - t1)) - 1
            forward_rates_date[f"{t1}-{t2}"] = forward_rate
        forward_rates[date] = forward_rates_date
    return forward_rates


def compute_covariance_matrix(time_series_data):
    returns = np.log(time_series_data / time_series_data.shift(1)).dropna()
    covariance_matrix = np.cov(returns.T)
    return covariance_matrix


def plot_forward_curve(forward_rates):
    cmap = plt.get_cmap('tab10')
    for i, date in enumerate(forward_rates):
        periods = list(forward_rates[date].keys())
        forward_values = list(forward_rates[date].values())
        plt.plot(periods, forward_values, label=date, marker='o', linestyle='-', color=cmap(i / len(forward_rates)))
    
    plt.xlabel('Period')
    plt.ylabel('Forward Rate')
    plt.title('Forward Curve')
    plt.legend()
    plt.show()



#Q5 Covariance Matrix
def calculate_log_returns(rates):
    log_returns = {}
    dates = list(rates.keys())
    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i + 1]
        log_returns[today] = {}
        for t in rates[today]:
            log_returns[today][t] = np.log(rates[tomorrow][t] / rates[today][t])
    return log_returns


def calculate_covariance_matrix(log_returns):
    data = pd.DataFrame(log_returns).T
    return data.cov()


# For example:
log_returns = calculate_log_returns(ytm_curve)
cov_matrix = calculate_covariance_matrix(log_returns)
print("Covariance Matrix:\n", cov_matrix)



#Q6 Eigenbalues & Eigenvectors

def calculate_eigen(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors


# For example:
eigenvalues, eigenvectors = calculate_eigen(cov_matrix)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)