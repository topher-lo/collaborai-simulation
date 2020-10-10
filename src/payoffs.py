import streamlit as st

import pandas as pd
import numpy as np
import altair as alt


def create_init_wallet(init_margin, N, seed=42):
    return np.array([init_margin for i in range(N)])


def payment_func(N, weights, mrml_array, index):
    """
    Input:
        weights: Array of weights for some round t
        mrml_array: Array of marginal return to ML for some round t
        index - Index of participant 
    Returns:
        List of (positive) payments from player at mrml_index 
        to other players.
    """

    mrml_i = mrml_array[index]
    weight_i = weights[index]
    payments_from_i = []
    for j in range(N):
        payment_from_i_to_j = mrml_array[j] * weights[j] - mrml_i * weight_i
        if payment_from_i_to_j > 0:
            payments_from_i.append(payment_from_i_to_j)
        else:
            payments_from_i.append(0)
    return payments_from_i


def make_payments(N, weights, mrml_array, wallet):
    """
    Input:
        wallet - 
            Array represent wallets for N players 
            at beginning of some round t        
        weights -
            Array of weights for some round t
    Returns: 
        N x N numpy matrix of payment from ith (row) player to 
        jth (column) player at some round t
    """

    payoffs = np.empty([N, N])
    for i in range(N):
        payments_from_i = payment_func(N, weights, mrml_array, i)
        for j in range(N):
            payoffs[i,j] = payments_from_i[j]
    return payoffs


def send_payment(payment, i, j, wallet):
    """
    Input:
        payment - 
            Payment made from player i to player j
        wallet - 
            Array represent wallets for N players 
            at beginning of round t
    Effect:
        Subtracts payment from ith index in wallet
        Adds payment to jth index in wallet
    Returns:
        (Amount left in player i's wallet, Amount left in player j's wallet)
    """

    wallet[i] -= payment
    wallet[j] += payment
    return (wallet[i], wallet[j])


def split_surplus(N, score, mrml_array):

    total_surplus = 0
    for i in range(N):
        total_surplus += mrml_array[i] * score
    return total_surplus/N


def simulate_wallet(T, N, ml_results, init_margin, mean_mrml, sd_mrml, fixed_mrml=False, mrml=None, fixed_weights=False, weight=None, SEED=42):

    wallet_over_time = np.zeros([T, N])
    margin_account = np.zeros([T, N])
    payments_over_time = np.zeros([T, N])
    init_wallet = create_init_wallet(init_margin, N)
    for i in range(N):
        wallet_over_time[0][i] = init_wallet[i] - init_margin
        margin_account[0][i] = init_wallet[i]
    if not(fixed_weights):
        weights_over_time = ml_results.iloc[:,N+1:].to_numpy()
    else:
        weights_over_time = np.empty([T, N])
        weights_over_time.fill(weight)
    if not(fixed_mrml):
        np.random.seed(SEED)
        mrml_array = np.random.normal(mean_mrml, sd_mrml, N)
    else:
        mrml_array = np.array([mrml for i in range(N)])
    for t in range(0,T):
        score = ml_results.iloc[t,N]
        weights = weights_over_time[t]
        # Compute payments matrix
        payments = make_payments(N, weights, mrml_array, init_wallet)
        for i in range(N):
            # Split surplus
            wallet_over_time[t][i] = split_surplus(N, score, mrml_array)
            # Append to payments over time matrix
            payments_over_time[t][i] = sum(payments[i])
        for i in range(N):
            for j in range(N):
                i_wallet, j_wallet = send_payment(payments[i][j], i, j, wallet_over_time[t])
                wallet_over_time[t][i] = i_wallet
                wallet_over_time[t][j] = j_wallet
                if t == 0:
                    i_margin, j_margin = send_payment(payments[i][j], i, j, [init_margin]*N)
                else:
                    i_margin, j_margin = send_payment(payments[i][j], i, j, margin_account[t-1])
                margin_account[t][i] = i_margin
                margin_account[t][j] = j_margin
    return payments_over_time, wallet_over_time, margin_account, mrml_array


def payoff_plots(payments_df, wallets_df, margin_df):

    scales = alt.selection_interval(bind='scales')
    payments_df = payments_df
    payments_df.reset_index(inplace=True)
    payments_fig = alt.Chart(payments_df.melt('index')).mark_area().encode(
        x=alt.X('index:O', axis=alt.Axis(title='Time Period')),
        y=alt.Y('value', axis=alt.Axis(title='Total Payment')),
        color=alt.Color(
            'variable:N', 
            legend=alt.Legend(title="Participant")
        )
    ).add_selection(
        scales
    ).properties(title='Payments Made per Participant over Time')

    margin_df = margin_df
    margin_df.reset_index(inplace=True)
    margin_fig = alt.Chart(margin_df.melt('index')).mark_line().encode(
        x=alt.X('index:O', axis=alt.Axis(title='Time Period')),
        y=alt.Y('value', axis=alt.Axis(title='Monetary Amount')),
        color=alt.Color(
            'variable:N', 
            legend=alt.Legend(title="Participant")
        )
    ).add_selection(
        scales
    ).properties(title='Margin Account per Participant over Time')

    wallets_df = wallets_df.cumsum()
    wallets_df.reset_index(inplace=True)
    wallets_fig = alt.Chart(wallets_df.melt('index')).mark_line().encode(
        x=alt.X('index:O', axis=alt.Axis(title='Time Period')),
        y=alt.Y('value', axis=alt.Axis(title='Monetary Amount')),
        color=alt.Color(
            'variable:N', 
            legend=alt.Legend(title="Participant")
        )
    ).add_selection(
        scales
    ).properties(title='Cumulative Cashflow per Participant over Time')

    return payments_fig, margin_fig, wallets_fig


def payoffs_loader(T, N, ml_results, init_margin, mean_mrml, sd_mrml, kwargs={}):
    payments_matrix, wallets_matrix, margin_matrix, mrml_df = simulate_wallet(T, N, ml_results, init_margin, mean_mrml, sd_mrml, **kwargs)
    return pd.DataFrame(payments_matrix), pd.DataFrame(wallets_matrix), pd.DataFrame(margin_matrix), pd.DataFrame(mrml_df)