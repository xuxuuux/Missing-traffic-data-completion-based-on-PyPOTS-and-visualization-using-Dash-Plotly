'''

This file mainly generates an interactive interface to display
and compare the complete time series data, the time series data
after random cropping, and the time series data completed by the
SAITS model.

Main tools: dash and plotly
'''


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

MISSING_RATE = 0.1


'''
to process the data in the same way as trainmodel.py.
'''
arr = np.load("datas/pems.npy")
imputed =np.load("imputed.npy")
print("original data shape:", arr.shape)

data = arr[np.newaxis, :, :]

rng = np.random.default_rng(42)
mask = rng.random(data.shape) < MISSING_RATE
incomplete = data.copy()
incomplete[mask] = np.nan
print("incomplete data shape:", incomplete.shape)


time_steps = data.shape[1]
num_features = data.shape[2]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Traffic Data Imputation Visualization"),
    html.Label("Select Feature:"),
    dcc.Dropdown(
        id='feature_dropdown',
        options=[{'label': f'Feature {i}', 'value': i} for i in range(num_features)],
        value=0
    ),
    dcc.Graph(id='imputation_graph')
])

@app.callback(
    Output('imputation_graph', 'figure'),
    Input('feature_dropdown', 'value')
)
def update_graph(feature_index):
    fig = go.Figure()


    fig.add_trace(go.Scatter(
        y=data[0, :, feature_index],
        mode='lines',
        name='Original',
        line=dict(color='#2E86AB', width=3)
    ))
    fig.add_trace(go.Scatter(
        y=incomplete[0, :, feature_index],
        mode='markers+lines',
        name='With Missing',
        line=dict(color='#F24236', width=2, dash='dot'),
        marker=dict(color='#F24236', size=6, symbol='circle-open')
    ))
    fig.add_trace(go.Scatter(
        y=imputed[0, :, feature_index],
        mode='lines',
        name='Imputed',
        line=dict(color='#26C485', width=3)  # 鲜艳的绿色
    ))

    fig.update_layout(
        title=f"Traffic Data for Feature {feature_index}",
        xaxis_title="Time Step",
        yaxis_title="Value",
        template="plotly_white",
        plot_bgcolor='rgba(248,249,250,1)',  # color for bg
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type="linear",
            gridcolor='rgba(230,230,230,0.5)'
        ),
        yaxis=dict(
            gridcolor='rgba(230,230,230,0.5)'  # color
        )
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)