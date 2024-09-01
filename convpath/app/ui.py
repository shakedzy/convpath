import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
from ..color_logger import get_logger
from ..settings import Settings
from ..utils import path_to_resource
from ..package_types import Conversation


class AppUI:
    OPACITY = 0.7
    LINE_WIDTH = 1

    def __init__(self, conversations: list[Conversation]) -> None:
        self.logger = get_logger()
        self.settings = Settings()
        self.conversations = conversations

    def _x_axis(self) -> list[int]:
        return list(range(1, len(self.conversations)+1))
    
    def _plot_similarities_averages(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        color = 'navy'
        plot = go.Scatter(
            x=self._x_axis(),
            y=[c.avg_similarity for c in self.conversations],
            mode='markers',
            marker_symbol='circle',
            marker_color=color,
            marker={ 'size': 8, },
        )
        figure.add_trace(plot, row=row, col=col)

        avg = sum([c.avg_similarity for c in self.conversations]) / len(self.conversations)
        figure.add_hline(y=avg, line_color=color, opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                      # type: ignore

        std = np.std([c.avg_similarity for c in self.conversations])
        if avg + std <= 1:
            figure.add_hline(y=avg + std, line_color=color, line_dash="dash", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)          # type: ignore
        if avg - std >= 0:
            figure.add_hline(y=avg - std, line_color=color, line_dash="dash", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)          # type: ignore
        if avg + (2*std) <= 1:
            figure.add_hline(y=avg + (2*std), line_color=color, line_dash="dot", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)       # type: ignore
        if avg - (2*std) >= 0:
            figure.add_hline(y=avg - (2*std), line_color=color, line_dash="dot", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)       # type: ignore
        
        return figure
    
    def _plot_error_bars(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        green_color = 'green'
        red_color = 'red'
        points_color = 'navy'

        green_error_bars = [c.max_similarity - c.avg_similarity for c in self.conversations]
        red_error_bars = [c.avg_similarity - c.min_similarity for c in self.conversations]
        avg = sum([c.avg_similarity for c in self.conversations]) / len(self.conversations)
        y = [avg] * len(self.conversations)
        zeros = [0] * len(self.conversations)

        green_plot = go.Scatter(
            x=self._x_axis(),
            y=y,
            mode='markers',
            marker={'size': 0},
            marker_color=green_color,
            error_y={
                'type': 'data',
                'symmetric': False,
                'array': green_error_bars,
                'arrayminus': zeros,
            }
        )
        figure.add_trace(green_plot, row=row, col=col)

        red_plot = go.Scatter(
            x=self._x_axis(),
            y=y,
            mode='markers',
            marker={'size': 0},
            marker_color=red_color,
            error_y={
                'type': 'data',
                'symmetric': False,
                'array': zeros,
                'arrayminus': red_error_bars,
            }
        )
        figure.add_trace(red_plot, row=row, col=col)

        points_plot = go.Scatter(
            x=self._x_axis(),
            y=y,
            mode='markers',
            marker={'size': 5},
            marker_color=points_color,
        )
        figure.add_trace(points_plot, row=row, col=col)

        avg_green_error = sum(green_error_bars) / len(green_error_bars)
        avg_red_error = sum(red_error_bars) / len(red_error_bars)
        green_errors_std = np.std(green_error_bars)
        red_errors_std = np.std(red_error_bars)
        figure.add_hline(y=avg, line_color=points_color, opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                                            # type: ignore

        figure.add_hline(y=avg + avg_green_error, line_color=green_color, opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                           # type: ignore
        figure.add_hline(y=avg + avg_green_error + green_errors_std, line_color=green_color, line_dash="dash", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)      # type: ignore
        figure.add_hline(y=avg + avg_green_error + (2*green_errors_std), line_color=green_color, line_dash="dot", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)   # type: ignore
        
        figure.add_hline(y=avg - avg_red_error, line_color=red_color, opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                               # type: ignore
        figure.add_hline(y=avg - avg_red_error - red_errors_std, line_color=red_color, line_dash="dash", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)            # type: ignore
        figure.add_hline(y=avg - avg_red_error - (2*red_errors_std), line_color=red_color, line_dash="dot", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)         # type: ignore

        return figure
    
    def _plot_first_last_similarities(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        color = 'magenta'
        abs_first_last = [abs(c.first_last_similarity_difference) for c in self.conversations]
        plot = go.Scatter(
            x=self._x_axis(),
            y=abs_first_last,
            mode='markers',
            marker_symbol='x',
            marker_color=color,
            marker={ 'size': 8, },
        )
        figure.add_trace(plot, row=row, col=col)

        avg = sum(abs_first_last) / len(self.conversations)
        figure.add_hline(y=avg, line_color=color, opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                  # type: ignore

        std = np.std(abs_first_last)
        if avg + std <= 1:
            figure.add_hline(y=avg + std, line_color=color, line_dash="dash", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)      # type: ignore
        if avg - std >= 0:
            figure.add_hline(y=avg - std, line_color=color, line_dash="dash", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)      # type: ignore
        if avg + (2*std) <= 1:
            figure.add_hline(y=avg + (2*std), line_color=color, line_dash="dot", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)   # type: ignore
        if avg - (2*std) >= 0:
            figure.add_hline(y=avg - (2*std), line_color=color, line_dash="dot", opacity=self.OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)   # type: ignore
        
        return figure


    def launch(self, host: str | None = None, port: int | None = None) -> None:
        """
        Launch the App server.

        Args:
            host: Hostname to listen on (default: "127.0.0.1").
            port: Port to listen on (default: 8050)
        """
        dash_app = Dash(
            assets_folder=path_to_resource("assets/"),
            external_stylesheets=["https://fonts.googleapis.com/icon?family=Material+Icons"]
        )
        rows = 3
        figure = make_subplots(rows=rows, cols=1,
                               shared_xaxes=True,
                               row_heights=[1./rows] * rows,
                               subplot_titles=[
                                   "Conversation Similarity",
                                   "Top Min/Max Similarity Across Conversations",
                                   "Distance of First and Last Rounds"
                                ])

        figure = self._plot_similarities_averages(figure, 1, 1)
        figure = self._plot_error_bars(figure, 2, 1)
        figure = self._plot_first_last_similarities(figure, 3, 1)

        figure.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='#fbfbfb')
        figure.update_traces(ids=[c.title for c in self.conversations], hovertemplate='%{id}<extra></extra>')
        figure.update_xaxes(visible=False)

        dash_app.layout = [
            html.Div(className="card", style={'text-align': 'center'}, children=[
                html.Img(src=dash_app.get_asset_url('logo.png'), style={'max-height': '100px'})
            ]),
            html.Div(className="graph-container",
                     children=[dcc.Graph(figure=figure, style={'height': '70vh'}),]),
        ]
        host_: str = host or self.settings.host
        port_: int = port or self.settings.port
        dash_app.run(host=host_, port=str(port_))
