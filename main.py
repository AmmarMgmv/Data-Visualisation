import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

app = dash.Dash(__name__)

data = pd.read_csv('Student Depression Dataset.csv')

data['Have you ever had suicidal thoughts ?'] = data['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
data['Family History of Mental Illness'] = data['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
data['Depression Status'] = data['Depression'].map({1: 'Depressed', 0: 'Not Depressed'})
sleep_duration_map = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4,
    'Others': 4.5
}
data['Sleep Duration Numeric'] = data['Sleep Duration'].map(sleep_duration_map)

# Add jitter
def preprocess_data(df):
    df = df.copy()
    df['Sleep Duration Jittered'] = df['Sleep Duration Numeric'] + np.random.uniform(-0.45, 0.45, len(df))
    df['Jittered Status'] = df['Depression Status'].apply(lambda x: 1 if x == 'Depressed' else 0) + np.random.uniform(-0.45, 0.45, len(df))
    return df

# Data samples
data_samples = {
    '25%': preprocess_data(data.sample(frac=0.25, random_state=42)),
    '50%': preprocess_data(data.sample(frac=0.5, random_state=42)),
    '75%': preprocess_data(data.sample(frac=0.75, random_state=42)),
    '100%': preprocess_data(data)
}

color_map = {'Healthy': 'green', 'Moderate': 'orange', 'Unhealthy': 'red', 'Others': 'purple'}

# Scatter plot
def create_scatter_plot(data_subset, gender):
    fig = go.Figure()
    filtered_data = data_subset

    if gender != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == gender]

    for habit in filtered_data['Dietary Habits'].unique():
        gender_data = filtered_data[filtered_data['Dietary Habits'] == habit]
        fig.add_trace(go.Scattergl(
            x=gender_data['Sleep Duration Jittered'],
            y=gender_data['Jittered Status'],
            mode='markers',
            marker=dict(
                size=gender_data['Work/Study Hours'] * 1.25,
                color=color_map[habit],
                sizemode='diameter',
                opacity=0.6,
                sizemin=6
            ),
            customdata=gender_data[['Gender', 'Sleep Duration', 'Depression Status', 'Dietary Habits', 'Work/Study Hours']],
            hovertemplate=(
                "<b>Gender:</b> %{customdata[0]}<br>"
                "<b>Sleep Duration:</b> %{customdata[1]}<br>"
                "<b>Depression Status:</b> %{customdata[2]}<br>"
                "<b>Dietary Habits:</b> %{customdata[3]}<br>"
                "<b>Work/Study Hours:</b> %{customdata[4]}<extra></extra>"
            ),
            name=f"{gender} - {habit}" if gender != "All" else habit
        ))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 4.5],
            ticktext=['<5 hours', '5-6 hours', '7-8 hours', '>8 hours', 'Others'],
            title='Sleep Duration'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Not Depressed', 'Depressed'],
            title='Depression Status'
        ),
        title={
            'text': 'How Sleep, Diet, and Work Hours Affect Student Depression',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top', 
            'y': 0.975
        },
        legend_title='Dietary Habits',
        margin=dict(l=40, r=40, t=40, b=40),
        height=600
    )
    return fig

#
def create_bi_bar_chart(age_range, filtered):
    males = filtered[filtered['Gender'] == 'Male']
    females = filtered[filtered['Gender'] == 'Female']

    metrics = ['Academic Pressure', 'Study Satisfaction', 'Work/Study Hours', 'Depressed (%)', 'Suicidal Thoughts (%)']

    male_vals = []
    female_vals = []
    hover_data_male = []
    hover_data_female = []

    # Process metrics
    for col in metrics:
        if col == 'Depressed (%)':
            male_avg = males['Depression'].mean() * 100 if not males.empty else 0
            female_avg = females['Depression'].mean() * 100 if not females.empty else 0
            male_display = f"{male_avg:.2f}%"
            female_display = f"{female_avg:.2f}%"
        elif col == 'Suicidal Thoughts (%)':
            male_avg = males['Have you ever had suicidal thoughts ?'].mean() * 100 if not males.empty else 0
            female_avg = females['Have you ever had suicidal thoughts ?'].mean() * 100 if not females.empty else 0
            male_display = f"{male_avg:.2f}%"
            female_display = f"{female_avg:.2f}%"
        elif col == 'Work/Study Hours':
            male_avg = males[col].mean() / 24 * 100 if not males.empty else 0
            female_avg = females[col].mean() / 24 * 100 if not females.empty else 0
            male_display = f"{males[col].mean():.2f} / 24"
            female_display = f"{females[col].mean():.2f} / 24"
        else:
            male_avg = males[col].mean() / 5 * 100 if not males.empty else 0 
            female_avg = females[col].mean() / 5 * 100 if not females.empty else 0
            male_display = f"{males[col].mean():.2f} / 5"
            female_display = f"{females[col].mean():.2f} / 5"

        male_vals.append(male_avg)
        female_vals.append(-female_avg) 
        hover_data_male.append([col, "Male", male_display])
        hover_data_female.append([col, "Female", female_display])

    y_positions = list(range(len(metrics)))

    # Create chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=y_positions,
        x=male_vals,
        name='Male',
        orientation='h',
        marker_color='blue',
        customdata=hover_data_male,
        hovertemplate=(
            "<b>Metric:</b> %{customdata[0]}<br>"
            "<b>Gender:</b> %{customdata[1]}<br>"
            "<b>Value:</b> %{customdata[2]}<extra></extra>"
        ),
        showlegend=False
    ))

    fig.add_trace(go.Bar(
        y=y_positions,
        x=female_vals,
        name='Female',
        orientation='h',
        marker_color='pink',
        customdata=hover_data_female,
        hovertemplate=(
            "<b>Metric:</b> %{customdata[0]}<br>"
            "<b>Gender:</b> %{customdata[1]}<br>"
            "<b>Value:</b> %{customdata[2]}<extra></extra>"
        ),
        showlegend=False
    ))

    for i, metric in enumerate(metrics):
        fig.add_annotation(
            x=-120,
            y=i,
            text=metric,
            showarrow=False,
            font=dict(size=12, color='black'),
            xanchor='left',
            yanchor='middle'
        )

    fig.update_layout(
        title={
            "text": f"Comparison of Metrics for Ages {age_range[0]}-{age_range[1]}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis=dict(
            range=[-120, 120],
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=['', 'Female', '', 'Male', '']
        ),
        yaxis=dict(
            tickvals=y_positions,
            ticktext=[''] * len(metrics),
            showline=False
        ),
        yaxis_title="",
        xaxis_title="Percentage Scale",
        barmode="relative",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# Heatmap
def create_heatmap(data):
    # Create CGPA bucket
    data['CGPA Bucket'] = pd.cut(data['CGPA'], bins=[5, 6, 7, 8, 9, 10], labels=["5-6", "6-7", "7-8", "8-9", "9-10"])

    # Group by Degree and CGPA Bucket, get depression percentage
    grouped_data = data.groupby(['Degree', 'CGPA Bucket']).agg(
        Percent_Depressed=('Depression', lambda x: (x.sum() / x.count()) * 100),
        Depressed_Count=('Depression', 'sum'),
        Total_Count=('Depression', 'count')
    ).reset_index()

    # Pivot table
    percent_pivot = grouped_data.pivot(index='CGPA Bucket', columns='Degree', values='Percent_Depressed')
    hover_pivot = grouped_data.pivot(index='CGPA Bucket', columns='Degree', values='Depressed_Count').fillna(0).astype(int)
    total_pivot = grouped_data.pivot(index='CGPA Bucket', columns='Degree', values='Total_Count').fillna(0).astype(int)

    hover_text = hover_pivot.astype(str) + " out of " + total_pivot.astype(str) + " students depressed"
    percent_pivot = percent_pivot.sort_index(ascending=True).iloc[::-1]

    # Create chart
    fig = px.imshow(
        percent_pivot,
        labels=dict(x='Degree', y='CGPA Bucket', color='Depression (%)'),
        color_continuous_scale='Reds',
        title='Percentage of Students Depressed by Degree and CGPA',
        text_auto=False
    )

    fig.update_traces(
        hovertemplate="<b>Degree:</b> %{x}<br>" +
                      "<b>CGPA Bucket:</b> %{y}<br>" +
                      "<b>Depression (%):</b> %{z:.2f}<br>" +
                      "<b>Details:</b> %{customdata}<extra></extra>",
        customdata=hover_text.values 
    )

    fig.update_layout(
        title={
            "text": "Percentage of Students Depressed by Degree and CGPA",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        height=400,
        xaxis_title="Degree",
        yaxis_title="CGPA Bucket",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# Styling
app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "flex-start",
        "padding": "10px",
    },
    children=[
        html.H1(
            "Student Depression Analysis",
            style={
                "textAlign": "center",
                "marginTop": "0",
                "width": "100%",
                "fontWeight": "bold",
                "fontSize": "30px",
                "color": "#2C3E50",
                "borderBottom": "2px solid #2980B9",
                "paddingBottom": "10px",
                "marginBottom": "10px"
            }
        ),
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "space-between",
                "width": "100%",
                "marginTop": "20px"
            },
            children=[
                # Scatter Plot
                html.Div(
                    style={
                        "width": "48%",
                        "padding": "5px",
                        "borderRadius": "8px",
                        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                    },
                    children=[
                        dcc.Graph(
                            id="depression-graph",
                            style={"width": "100%", "height": "600px"}
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "center",
                                "marginTop": "5px",
                                "width": "100%",
                                 "gap": "20%"
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.Label(
                                            "Select Gender:",
                                            style={
                                                "display": "block",
                                                "marginBottom": "5px",
                                                "fontSize": "14px",
                                                "color": "#333",
                                                "fontWeight": "600"
                                            }
                                        ),
                                        dcc.Dropdown(
                                            id="gender-dropdown",
                                            options=[
                                                {"label": "All", "value": "All"},
                                                {"label": "Male", "value": "Male"},
                                                {"label": "Female", "value": "Female"}
                                            ],
                                            placeholder="Select Gender",
                                            value="All",
                                            clearable=False,
                                            style={"width": "100px"}
                                        )
                                    ],
                                    style={"textAlign": "center"}
                                ),
                                html.Div(
                                    children=[
                                        html.Label(
                                            "Select Percentage:",
                                            style={
                                                "display": "block",
                                                "marginBottom": "5px",
                                                "fontSize": "14px",
                                                "color": "#333",
                                                "fontWeight": "600"
                                            }
                                        ),
                                        dcc.Dropdown(
                                            id="percentage-dropdown",
                                            options=[
                                                {"label": "25%", "value": "25%"},
                                                {"label": "50%", "value": "50%"},
                                                {"label": "75%", "value": "75%"},
                                                {"label": "100%", "value": "100%"}
                                            ],
                                            placeholder="Select Percentage",
                                            value="100%",
                                            clearable=False,
                                            style={"width": "100px"}
                                        )
                                    ],
                                    style={"textAlign": "center"}
                                )
                            ] 
                        )
                    ] 
                ),
                html.Div(
                    style={
                        "width": "48%",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "20px"
                    },
                    children=[
                        # Bidirectional chart
                        html.Div(
                            style={
                                "padding": "10px",
                                "borderRadius": "8px",
                                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
                            },
                            children=[
                                dcc.Graph(
                                    id="bi_bar_chart",
                                    style={"width": "100%", "height": "300px"}
                                ),
                                dcc.Store(id="play-state", data={"playing": True}),
                                html.Div(
                                    style={"display": "flex", "justifyContent": "end", "paddingRight": "20px"},
                                    children=[
                                        html.Button(
                                            "Pause", id="toggle-button", n_clicks=0,
                                            style={
                                                "padding": "8px 16px",
                                                "fontSize": "14px",
                                                "color": "white",
                                                "background": "linear-gradient(to right, #2980B9, #2E86C1)",
                                                "border": "none",
                                                "borderRadius": "5px",
                                                "cursor": "pointer",
                                                "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                                                "transition": "background 0.3s ease, box-shadow 0.3s ease"
                                            }
                                        )
                                    ]
                                ),
                                dcc.Interval(
                                    id="interval-component",
                                    interval=2000,
                                    n_intervals=0,
                                    disabled=False
                                )
                            ]
                        ),
                        # Heatmap
                        html.Div(
                            style={
                                "padding": "10px",
                                "borderRadius": "8px",
                                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
                            },
                            children=[
                                dcc.Graph(
                                    id="heatmap-graph",
                                    figure=create_heatmap(data),
                                    style={"width": "100%", "height": "400px"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Update graph based on dropdown selections
@app.callback(
    Output("depression-graph", "figure"),
    Input("gender-dropdown", "value"),
    Input("percentage-dropdown", "value")
)
def update_graph(selected_gender, selected_percentage):
    if not selected_percentage or selected_percentage not in data_samples:
        selected_percentage = "100%"
    if not selected_gender:
        selected_gender = "All"
    filtered_data = data_samples[selected_percentage]
    return create_scatter_plot(filtered_data, selected_gender)

# Bi bar chart animation
@app.callback(
    Output("bi_bar_chart", "figure"),
    Input("interval-component", "n_intervals")
)
def update_split_chart(n_intervals):
    age_ranges = [(15, 20), (21, 25), (26, 30), (31, 35), (36, 40), (41, 45), (46, 50)]
    current_range = age_ranges[n_intervals % len(age_ranges)]
    filtered_data = data[(data['Age'] >= current_range[0]) & (data['Age'] <= current_range[1])]
    return create_bi_bar_chart(current_range, filtered_data)

# Play/Pause button
@app.callback(
    [Output("interval-component", "disabled"),
     Output("toggle-button", "children"),
     Output("play-state", "data")],
    Input("toggle-button", "n_clicks"),
    State("play-state", "data")
)
def toggle_play_pause(n_clicks, play_state):
    if n_clicks > 0:
        playing = not play_state["playing"]
    else:
        playing = True

    button_label = "Pause" if playing else "Play"
    return not playing, button_label, {"playing": playing}

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
