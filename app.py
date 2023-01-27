import streamlit as st  # web development
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import time  # to simulate a real time data, time loop
import plotly.express as px  # interactive charts
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import plotly.graph_objects as go
import statistics as sta
import calendar
import datetime as dt
import cufflinks as cf


def get_prices(stocks, s1, e1):
    closing_df = yf.download(stocks, s1, e1)['Adj Close']
    closing_df["Date"] = closing_df.index.date
    closing_df.set_index("Date", drop=True, inplace=True)
    closing_df.rename(
        columns={'ADS.DE': 'ADS', 'AIR.DE': 'AIR', 'ALV.DE': 'ALV', 'BAS.DE': 'BAS', 'BAYN.DE': 'BAYN', 'BEI.DE': 'BEI',
                 'BMW.DE': 'BMW', 'BNR.DE': 'BNR', 'CON.DE': 'CON', '1COV.DE': 'COV', 'DTG.DE': 'DTG', 'DBK.DE': 'DBK',
                 'DB1.DE': 'DB1', 'DPW.DE': 'DPW', 'EOAN.DE': 'EOAN', 'FME.DE': 'FME', 'FRE.DE': 'FRE',
                 'HNR1.DE': 'HNR1', 'HEI.DE': 'HEI'}, inplace=True)
    return closing_df


# Get preliminary one year European Stock daily data
end = datetime.now()
start = datetime(end.year - 2, end.month, end.day)
stock = ['ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 'BMW.DE', 'BNR.DE', 'CON.DE', '1COV.DE',
         'DTG.DE', 'DBK.DE', 'DB1.DE', 'DPW.DE', 'EOAN.DE', 'FME.DE', 'FRE.DE', 'HNR1.DE', 'HEI.DE']

df = get_prices(stock, start, end)

st.set_page_config(
    page_title='Real-Time Apollo Option Dashboard',
    page_icon='✅',
    layout='wide'
)

# dashboard title

st.title("Real-Time Apollo Option Dashboard")

# top-level filters

column_headers = list(df.columns.values)
option_headers = ['Put', 'Call']
option_headers1 = [30, 60, 90, 120]

st.sidebar.write("European Stock Tools")
stock_filter = st.sidebar.selectbox("Select Ticker", column_headers)
option_strike = int(df[stock_filter].iloc[-1])
strike_opt = st.sidebar.text_input("Strike", option_strike, max_chars=None, key=None, type='default')
type_opt = st.sidebar.selectbox("Option Type", option_headers)
dte_opt = st.sidebar.selectbox("Days to Expiration", option_headers1)
# creating a single-element container.
placeholder = st.empty()

############################# Volatility Cones

log_returns = np.log(df[stock_filter] / df[stock_filter].shift(1)).dropna()

TRADING_DAYS = 20
volatility = log_returns.rolling(window=TRADING_DAYS).std() * np.sqrt(252)


def ThirdThurs(year, month):
    # Create a datetime.date for the last day of the given month
    daysInMonth = calendar.monthrange(year, month)[1]  # Returns (month, numberOfDaysInMonth)
    date = dt.date(year, month, daysInMonth)
    # Back up to the most recent Thursday
    offset = 4 - date.isoweekday()
    if offset >= 0: offset -= 7  # Back up one week if necessary
    date += dt.timedelta(offset)  # dt is now date of last Th in month

    # Throw an exception if dt is in the current month and occurred before today
    now = dt.date.today()  # Get current date (local time, not utc)
    if date.year == now.year and date.month == now.month and date < now:
        raise Exception('Missed third thursday')

    return date - dt.timedelta(7)


dates = [ThirdThurs(year, month) for year in range(2021, 2023) for month in range(1, 13) if
         ThirdThurs(year, month) < dt.datetime.now().date()]
columnsNames = ['1-mth', '3-mth', '6-mth', '9-mth', '12-mth']
tradingDays = [int(20 * n) for n in [1, 3, 6, 9, 12]]
DTE = [int(30 * n) for n in [1, 3, 6, 9, 12]]
data = np.array([np.arange(len(dates))] * len(columnsNames)).T

volatility = {}
for TRADING_DAYS, TimePeriod in zip(tradingDays, columnsNames):
    print(TRADING_DAYS, TimePeriod)
    volatility[TimePeriod] = log_returns.rolling(window=TRADING_DAYS).std() * np.sqrt(252)

df1 = pd.DataFrame(data, columns=columnsNames, index=dates)
df1.index.name = 'period'


def historical_vol(x):
    df2 = x.copy()
    for date, val in x.iteritems():
        try:
            df2.loc[date] = round(volatility[x.name].loc[date] * 100, 2)
        except:
            df2.loc[date] = np.nan
    return df2


df1 = df1.apply(lambda x: historical_vol(x), axis=0)

df2 = pd.DataFrame(data='', columns=['max', 'mean', 'min'], index=DTE)
df2.index.name = 'DTE'

df2['max'] = pd.Series(df1[columnsNames].max().values, index=DTE)
df2['mean'] = pd.Series(df1[columnsNames].mean().values, index=DTE)
df2['min'] = pd.Series(df1[columnsNames].min().values, index=DTE)

##################### Option Simulation !!!!
# create list of Log Returns
lista = list()
for i in range(df[stock_filter].shape[0] - 1):
    lista.append(np.log(df[stock_filter].iloc[i + 1] / df[stock_filter].iloc[i]))
v = sta.stdev(lista)

# Simulation Parameters
r = 0.06
k = option_strike
T = dte_opt / 252
n = 10
g = 100
o = 50

# spreads of stock price
dt = T / n
lista = [0]
while len(lista) < n + 1:
    lista.append(lista[-1] + dt)
S0 = df[stock_filter].iloc[-1]
lista1 = [S0]
v1 = v
while len(lista1) < o:
    lista1.append(lista1[-1] + v1)
    if lista1[0] - v1 >= 0:
        lista1.append(lista1[0] - v1)
    lista1.sort()
while len(lista1) != o:
    lista1.pop()

# dataframe filter
df['Returns'] = df[stock_filter].pct_change() * 100
df['Log Returns'] = np.log(df[stock_filter]) - np.log(df[stock_filter].shift(1))
df['20 day Historical Volatility'] = 100 * df['Log Returns'].rolling(window=20).std() * np.sqrt(20)
df.dropna(inplace=True)
# near real-time / live feed simulation

for seconds in range(200):
    # while True:
    with placeholder.container():
        # create three columns
        # kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs 
        # kpi1.metric(label="Age ⏳", value=1, delta=1)
        # kpi2.metric(label="Married Count 💍", value=2, delta=2)
        # kpi3.metric(label="A/C Balance ＄", value=3, delta=3)

        # create two columns for charts 

        fig_col1, fig_col2, fig_col3 = st.columns(3)
        with fig_col1:
            #st.markdown("Daily Price Of " + stock_filter)
            #fig = px.line(df, y=stock_filter, title="Prices", height=400, width=600)
            fig = df[stock_filter].iplot(kind='line',
                                       asFigure=True,
                                       colors='Black',
                                       layout=Layout(title=stock_filter+" daily chart", yaxis=dict(title="Prices"),
                                                     autosize=True, height=350, width=400))
            #fig = px.line(df, y=stock_filter, title="Prices", autosize=True)
            st.plotly_chart(fig)
            # st.line_chart(df[stock_filter])
            # st.write(fig)
        with fig_col2:
            fig2 = df['Returns'].iplot(kind='box',
                                       asFigure=True,
                                       colors='Green',
                                       layout=Layout(title="1-Day Stock Returns", yaxis=dict(title="Volatility (%)"), autosize=True, height=350, width=400))
                                       #layout = Layout(title="Stocks Returns", autosize=True))
            st.plotly_chart(fig2)
        with fig_col3:
            # st.markdown("Distribution Of Stock Return")
            # st.line_chart(df[stock_filter].pct_change())
            fig3 = df['20 day Historical Volatility'].iplot(kind='box', asFigure=True,
                                                            layout=Layout(
                                                                title="20-Day HV Distrib.- "
                                                                      "q1(-25%) q3(+25%)", yaxis=dict(title="Volatility (%)"),
                                                                autosize=True, height=350, width=400))
            #"q1(-25%) q3(+25%)", autosize = True))
            st.plotly_chart(fig3)

        fig_col22, fig_col21, fig_col23 = st.columns(3)
        with fig_col22:
            # Monte Carlo
            fig22 = go.Figure()
            fig21 = go.Figure()
            lc1 = list()
            lp2 = list()
            for i4 in lista1:
                lista4 = list()
                lista5 = list()
                for i5 in list(range(g)):
                    lis = [i4]
                    m = tuple([n])
                    while n != 0:
                        i4 = i4 * np.exp((r - v ** 2 / 2) * dt + v * np.sqrt(dt) * np.random.normal(0, 1))
                        lis.append(i4)
                        n = n - 1
                    lista4.append(max(i4 - k, 0) * np.exp(-r * T))
                    lista5.append(max(k - i4, 0) * np.exp(-r * T))
                    i4 = lis[0]
                    n = int(m[0])
                    if i4 == S0:
                        fig22 = fig22.add_trace(go.Scatter(x=lista, y=lis))

                c1 = max(0, sta.mean(lista4))
                p2 = max(0, sta.mean(lista5))
                lc1.append(c1)
                lp2.append(p2)
                # if i4 == S0:
                #   fig21 = fig21.add_trace(go.Scatter(x=i4, y=c1))
                #   fig21 = fig21.add_trace(go.Scatter(x=i4, y=p2))
            fig21 = fig21.add_trace(go.Scatter(x=lista1, y=lc1, name='Call'))
            fig21 = fig21.add_trace(go.Scatter(x=lista1, y=lp2, name='Put'))
            fig22.update_layout(title="Stock price pattern simulated by Monte Carlo method", autosize=False,
                                yaxis_title="Stock price", xaxis_title="Fractional Time N/T(252)",
                                width=600, height=400, margin=dict(l=40, r=40, b=40, t=40))
            st.plotly_chart(fig22)
        with fig_col21:
            # st.markdown("Distribution Of Stock Return")
            # st.line_chart(df[stock_filter].pct_change())
            fig21.update_layout(title="Stock Option pricing by Monte Carlo", autosize=False,
                                yaxis_title="Simulated Price", xaxis_title="Stock price Range",
                                width=600, height=400, margin=dict(l=40, r=40, b=40, t=40))
            st.plotly_chart(fig21)
        with fig_col23:
            fig23 = go.Figure()
            plt.xticks(np.linspace(0, 360, 13))
            plt.ylim(0, 100)
            plt.xlim(0, 370)
            fig23.update_layout(title="Historical Volatility Cone vs. Implied volatility", autosize=False,
                                yaxis_title="Volatility (%)", xaxis_title="Days to Expiry (DTE)",
                                width=600, height=400, margin=dict(l=40, r=40, b=40, t=40))
            fig23 = fig23.add_trace(go.Scatter(x=df2.index, y=df2['max'], name='Max'))
            fig23 = fig23.add_trace(go.Scatter(x=df2.index, y=df2['mean'], name='Mean'))
            fig23 = fig23.add_trace(go.Scatter(x=df2.index, y=df1.iloc[-1, :], name='Current'))
            fig23 = fig23.add_trace(go.Scatter(x=df2.index, y=df2['min'], name='MIn'))
            st.plotly_chart(fig23)

        # st.markdown("### Detailed Data View")
        # st.dataframe(df)
        time.sleep(10)
    # placeholder.empty()
