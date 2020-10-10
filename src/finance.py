import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

from .payoffs import payment_func
from .payoffs import simulate_wallet


def calculate_NPV(T, r, cashflow_list):
    NPV = 0
    for t in range(1, T+1):
        NPV += cashflow_list[t-1] / (1+r)**t
    return NPV


def finance_plots(N, actual_payoffs, derivative_payoffs):

    actual_payoffs = actual_payoffs.copy()
    derivative_payoffs = derivative_payoffs.copy()

    for i in range(len(actual_payoffs)):
        if i == 0:
            actual_payoffs.iloc[i] = actual_payoffs.iloc[i] - 100
            derivative_payoffs.iloc[i] = derivative_payoffs.iloc[i] - 100
        else:
            for row in range(i):
                actual_payoffs.iloc[i] = actual_payoffs.iloc[i] - actual_payoffs.iloc[row]
                derivative_payoffs.iloc[i] = derivative_payoffs.iloc[i] - derivative_payoffs.iloc[row]

    scales = alt.selection_interval(bind='scales')
    plots = []
    f_payoffs = derivative_payoffs - actual_payoffs
    for i in range(N):
        payoffs_df = f_payoffs.iloc[:,i].reset_index()
        payoffs_df['index'] = np.arange(1, len(payoffs_df) + 1)
        title = "Short Forward Payoff to Player " + str(i) + " per Period"
        payoff_fig = alt.Chart(payoffs_df.melt('index')).mark_bar().encode(
            x=alt.X('index:O', axis=alt.Axis(title='Time Period')),
            y=alt.Y('value', axis=alt.Axis(title='Payoff')),
            color=alt.condition(
                alt.datum.value > 0,
                alt.value("#6AB187"),
                alt.value('#D32D41')
            )
        ).add_selection(
        scales
        ).properties(title=title)
        plots.append(payoff_fig)
    return plots


def cashflow_loader(T, wallets_df):
    cashflow_list = []
    for t in range(0, T):
        cashflow_list.append(sum(wallets_df.iloc[t,:]))
    return cashflow_list