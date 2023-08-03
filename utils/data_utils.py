import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def animate_play_court(data, play_id, save_as_video=False, video_filename="animation.html"):
    play_data = data[data['play'] == play_id]

    # Create a list to hold our frames
    frames = []

    # Get unique timestamps
    timestamps = play_data['timestamp'].unique()

    for timestamp in timestamps:
        snapshot = play_data[play_data['timestamp'] == timestamp]

        frame = go.Frame(data=[
            go.Scatter(
                x=snapshot[snapshot['player_id'] != -1]['x'],  # Exclude the ball
                y=snapshot[snapshot['player_id'] != -1]['y'],
                mode='markers+text',
                text=snapshot['player_name'],  # Player ID
                marker=dict(size=8, color=snapshot[snapshot['player_id'] != -1]['team_id']),  # Color by team ID
                name='Players',
                textposition="bottom center"
            ),
            go.Scatter(
                x=snapshot[snapshot['player_id'] == -1]['x'],  # Include only the ball
                y=snapshot[snapshot['player_id'] == -1]['y'],
                mode='markers',
                marker=dict(size=8, symbol='diamond', color='black'),
                name='Ball'
            )
        ])

        frames.append(frame)

    # Create the base figure layout
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"Player and Ball Positions for Play {play_id}",),
        specs=[[{'type': 'scatter'}]]
    )

    fig.update_xaxes(range=[0, 100], showgrid=False, zeroline=False, title="X Coordinate", row=1, col=1)  # Approx dimensions of a basketball court
    fig.update_yaxes(range=[0, 50], showgrid=False, zeroline=False, title="Y Coordinate", row=1, col=1)  # Approx dimensions of a basketball court

    fig.add_trace(
        go.Scatter(
            x=[0, 0, 100, 100, 0],
            y=[0, 50, 50, 0, 0],
            mode="lines",
            line=dict(width=2, color="Black"),
            showlegend=False
        ),
        row=1, col=1
    )

    fig.frames = frames

    # Adjust frame duration and transition duration to make the animation faster and smoother
    fig.layout.update(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 10, "redraw": True},
                                       "fromcurrent": True, 
                                       "transition": {"duration": 10}}])])],
        autosize=True
    )

    if save_as_video:
        fig.write_html(video_filename)
        
    return fig
df = pd.read_csv("./data/0021500391.csv")
animate_play_court(df, 20, save_as_video=True, video_filename="animation.html")