import numpy as np
import pandas as pd

import altair as alt
import streamlit as st

from src.simulation import ml_loader
from src.payoffs import payoffs_loader
from src.payoffs import payoff_plots
from src.finance import calculate_NPV
from src.finance import cashflow_loader
from src.finance import finance_plots


DATA_URL = "data/insurance.csv"

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data


def main():

    st.markdown(
        """
        # Summary
        ## Prediction Problem:
        Likelihood that an insurance client at Allstate Corporation defaults on their payment.
        ## Data Source:
        Predictive models trained on Allstate insurance claims data found from Kaggle.
        ## Additional Information:
        The formulas and intuition behind the payoff and financial dashboards 
        are outlined in our business canvas and technical paper found in our deliverables. 
        """
    )

    ### Sidebar
    st.sidebar.title("collabor.ai Simulation")
    st.sidebar.markdown(
        """
        ## ML Task:
        Insurance claims default probability.
        """
    )
    st.sidebar.markdown('## Number of Players:')
    N_players = st.sidebar.slider('How many participants?', 2, 5, 3)
    st.sidebar.markdown('## Time Periods:')
    T_periods = st.sidebar.slider('For long do participants collaborate?', 2, 10, 5)
    st.sidebar.markdown('## Initial Margin:')
    init_margin = st.sidebar.number_input(
        'How much do participants initially deposit in their wallet?', 
        min_value=None,
        max_value=None,
        value=50,
    )
    st.sidebar.markdown('## Marginal Return to ML:')
    mean_mrml = st.sidebar.number_input(
        'What is the average marginal return to ML across participants?',
        min_value=None,
        max_value=None,
        value = 10,
        )
    sd_mrml = st.sidebar.number_input(
        'What is the standard deviation of marginal returns to ML across participants?', 
        min_value=None,
        max_value=None,
        value = 0,
    )

    ### Data and Summary Stats

    st.markdown(
        """
        ## Summary Statistics: 
        """
    )

    data = load_data()
    st.write(data.describe())

    ### ML Results

    st.title("Dashboards")
    results = ml_loader("default", data, N_players, T_periods)

    st.write(results.iloc[:,:N_players])
    st.write(results.iloc[:,N_players:])

    ### ML Plots

    st.markdown(
        """
        ## Machine Learning Evaluation   
        ### Individual ML Algorithm:
        * Output Layer - Single densely connected NN layer with batch normalisation
        * Activation Function - Sigmoid
        * Optimiser - ADAM
        * Loss Function - Binary cross entropy     
        ### Ensemble ML Algorithm:
        CatBoost - Categorical boost classifier
        """
    )

    scales = alt.selection_interval(bind='scales')

    scores_df = results.iloc[:,:N_players+1].reset_index()
    scores_fig = alt.Chart(scores_df.melt('index')).mark_line().encode(
        x=alt.X('index:O', axis=alt.Axis(title='Time Period')),
        y=alt.Y('value', axis=alt.Axis(title='Score')),
        color=alt.Color(
            'variable:N', 
            legend=alt.Legend(title="Produced by:")
        )
    ).add_selection(
        scales
    ).properties(title='Prediction Scores per Participant over Time')

    st.altair_chart(scores_fig, use_container_width=True)

    weights_df = results.iloc[:,N_players+1:].reset_index()
    weights_fig = alt.Chart(weights_df.melt('index')).mark_line().encode(
        x=alt.X('index:O', axis=alt.Axis(title='Time Period')),
        y=alt.Y('value', axis=alt.Axis(title='Weight')),
        color=alt.Color(
            'variable:N', 
            legend=alt.Legend(title="Produced by:")
        )
    ).add_selection(
        scales
    ).properties(title='Ensemble Weight per Participant over Time')

    st.altair_chart(weights_fig, use_container_width=True)

    ### Payoff Plots

    payments_df, wallets_df, margin_df, mrml_df = payoffs_loader(T_periods, N_players, results, init_margin, mean_mrml, sd_mrml)
    st.markdown(
        """
        ## Payoffs Evaluation
        """
    )
    mrml_df = mrml_df.reset_index()
    mrml_fig = alt.Chart(mrml_df.melt('index')).mark_bar().encode(
        x=alt.X('index:N', axis=alt.Axis(title='Player')),
        y=alt.Y('value', axis=alt.Axis(title='Marginal Return to ML')),
    ).add_selection(
        scales
    ).properties(title='Marginal Return to ML over Time')
    st.altair_chart(mrml_fig, use_container_width=True)
    st.markdown(
        """
        ### Payment Function
        From player *i* to player *j*:
        """
    )
    st.latex(
        r"""
        b_{ij}(\boldsymbol{y_t}) = \theta_j\phi_j x_t - \theta_i\phi_i x_t
        """
    )
    st.markdown(
        """
        #### Note:

        The payment function represents the *difference* in the level of contribution between player *j* and player *i*.
        Formally, this function captures the (positive / negative) externality player *i* imposes on player *j*.
        """
    )
    for plot in payoff_plots(payments_df, wallets_df, margin_df):
        st.altair_chart(plot, use_container_width=True)

    ### Financial Plots

    st.markdown("## Futures Contract Evaluation")
    st.markdown('### Discount Rate:')
    discount_rate = st.number_input('How much do participants discount their future payoffs from collaborative machine learning?', 
        min_value=0.0,
        max_value=1.0,
        value = 0.2,
    )
    cashflow = cashflow_loader(T_periods, payments_df)
    NPV = calculate_NPV(T_periods, discount_rate, cashflow)
    st.markdown(
        """
        ### NPV of Network:
        """
    )
    st.write("$",round(NPV,2))
    st.latex(
        r"""
        NPV = \sum^{\infty}_{t=0}\frac{\sum^N_{i=1} E[\theta_{it} x_t ]}{(1+r)^t}
        """
    )
    st.markdown('### Hedging R&D Risk: Strike Ensemble Weight')
    strike_weight = st.number_input(
        'What is the strike ensemble weight offered by the active participant to the investor?', 
        min_value=0.0,
        max_value=1.0,
        value = 1/N_players,
    )
    st.markdown('### Hedging Market Risk: Strike Marginal Return to ML')
    strike_mrml = st.number_input(
        'What is the strike marginal rate of ML offered by the active participant to the investor?', 
        min_value=None,
        max_value=None,
        value = 4,
    )
    fweights_payments_df, fweights_wallets_df, margin, mrml_df = payoffs_loader(T_periods, N_players, results, init_margin, mean_mrml, sd_mrml, kwargs={"fixed_weights":True, "weight":strike_weight})
    fmrml_payments_df, fmrml_wallets_df, margin, mrml_df = payoffs_loader(T_periods, N_players, results, init_margin, mean_mrml, sd_mrml, kwargs={"fixed_mrml":True, "mrml":strike_mrml})

    st.markdown(
        """
        ## Futures (Ensemble Weight) Payoff Graphs
        ### Futures Payoff Function
        For player *i* at round *t*:
        """
    )
    st.latex(
        r"""
        payoff_{it} = payment_{it}(\widehat{\phi_{i}} | \phi_{-it}, \theta_t) - payment_{it}(\phi_{t}, \theta_{t})
        """
    )
    st.markdown(
        """
        #### Note:

        This futures payoff function compares the payoff at the strike weight
        with the ***actual*** payoff.
        """
    )
    for plot in finance_plots(N_players, payments_df, fweights_payments_df):
        st.altair_chart(plot, use_container_width=True)

    st.markdown(
        """
        ## Futures (Marginal Returns to ML) Payoff Graphs
        ### Futures Payoff Function
        For player *i* at round *t*:
        """
    )
    st.latex(
        r"""
        payoff_{it} = payment_{it}(\widehat{\theta_{i}} | \theta_{-it}, \phi_t) - payment_{it}(\phi_{t}, \theta_{t})
        """
    )
    st.markdown(
        """
        #### Note:

        This futures payoff function compares the payoff at the strike marginal return to ML 
        with the ***actual*** payoff.
        """
    )
    for plot in finance_plots(N_players, payments_df, fmrml_payments_df):
        st.altair_chart(plot, use_container_width=True)


if __name__ == "__main__":
    main()
