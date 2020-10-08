import numpy as np
import pandas as pd

import altair as alt
import streamlit as st

from src.simulation import ml_loader


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
    st.title("Dashboards")

    ### Sidebar
    st.sidebar.title("collabor.ai's PyPoP Simulation")
    st.sidebar.markdown(
        """
        ## PoP Protocol Puzzle:
        Insurance claims default probability.
        """
    )
    st.sidebar.markdown('## Number of Players:')
    N_players = st.sidebar.slider('How many participants?', 2, 5, 3)
    st.sidebar.markdown('## Time Periods:')
    T_periods = st.sidebar.slider('For long do participants collaborate?', 2, 10, 3)
    st.sidebar.markdown('## Initial Margin:')
    init_margin = st.sidebar.number_input(
        'How much do participants initially deposit in their wallet?', 
        min_value=None,
        max_value=None,
        value=100,
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
        value = 5,
    )

    ### ML Results

    data = load_data()
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


if __name__ == "__main__":
    main()