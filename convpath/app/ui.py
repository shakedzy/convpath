import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output, State
from dash.exceptions import PreventUpdate
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
    
    def _marker_sizes(self, highlighted_id: str | None, default_size: int = 8) -> dict[str, list]:
        sizes = [16 if c.title == highlighted_id else default_size for c in self.conversations]
        return {'size': sizes}
    
    def _plot_similarities_averages(self, figure: go.Figure, row: int, col: int, highlighted_id: str | None = None) -> go.Figure:
        color = 'navy'
        plot = go.Scatter(
            x=self._x_axis(),
            y=[c.avg_similarity for c in self.conversations],
            mode='markers',
            marker_symbol='square',
            marker_color=color,
            marker=self._marker_sizes(highlighted_id),
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
    
    def _plot_error_bars(self, figure: go.Figure, row: int, col: int, highlighted_id: str | None = None) -> go.Figure:
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
            marker=self._marker_sizes(highlighted_id, default_size=5),
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
    
    def _plot_first_last_similarities(self, figure: go.Figure, row: int, col: int, highlighted_id: str | None = None) -> go.Figure:
        color = 'magenta'
        abs_first_last = [abs(c.first_last_similarity_difference) for c in self.conversations]
        plot = go.Scatter(
            x=self._x_axis(),
            y=abs_first_last,
            mode='markers',
            marker_symbol='x',
            marker_color=color,
            marker=self._marker_sizes(highlighted_id),
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
    
    def _register_callbacks(self) -> None:
        @callback(
            [Output('main-figure', 'figure'), Output('id-selector', 'value'), Output('last-id-displayed', 'data'), Output('clear-button-clicks', 'data')], 
            [Input('main-figure', 'hoverData'), Input('id-selector', 'value'), Input('clear-selector-button', 'n_clicks')],
            [State('last-id-displayed', 'data'), State('clear-button-clicks', 'data')]
        )
        def update_graph(hover_data, selector_value, button_clicks, last_id, prev_button_clicks) -> tuple[go.Figure, str | None, str | None, int]:
            if button_clicks is None:
                button_clicks = 0

            if selector_value != last_id:
                hover_id = selector_value
            elif button_clicks > prev_button_clicks:
                hover_id = None
            elif hover_data is not None:
                hover_id = hover_data['points'][0]['id']  # ID of the hovered point
            else:
                raise PreventUpdate()

            return (self._create_figure(hover_id), hover_id, hover_id, button_clicks)

    def _create_figure(self, highlighted_id: str | None = None) -> go.Figure:
        rows = 3
        figure = make_subplots(rows=rows, cols=1,
                               shared_xaxes=True,
                               row_heights=[1./rows] * rows,
                               subplot_titles=[
                                   "Conversation Similarity",
                                   "Top Min/Max Similarity Across Conversations",
                                   "Distance of First and Last Rounds"
                                ])

        figure = self._plot_similarities_averages(figure, row=1, col=1, highlighted_id=highlighted_id)
        figure = self._plot_error_bars(figure, row=2, col=1, highlighted_id=highlighted_id)
        figure = self._plot_first_last_similarities(figure, row=3, col=1, highlighted_id=highlighted_id)

        figure.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='#fbfbfb')
        figure.update_traces(ids=[c.title for c in self.conversations], hovertemplate='%{id}<extra></extra>')
        figure.update_xaxes(visible=False)

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
            external_stylesheets=[
                "https://fonts.googleapis.com/icon?family=Material+Icons",
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
            ]
        )
        self._register_callbacks()
        
        dash_app.layout = [
            dcc.Store(id='last-id-displayed', data=None),
            dcc.Store(id='clear-button-clicks', data=0),
            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="left-column", children=[
                        html.Div(className="card", style={'text-align': 'center'}, children=[
                            html.Img(src=dash_app.get_asset_url('logo.png'), style={'max-height': '100px'})
                        ])
                    ]),
                    html.Div(className="right-column", children=[
                        html.Div(className="card", children=[
                            html.Label('Focus on conversation:'),
                            html.Div(style={'margin-top': '10px'}, children=[
                                html.Button(id='clear-selector-button', className="btn", children=[
                                    html.I(className="fas fa-times") 
                                ], style={'margin-right': '10px', 'vertical-align': 'top'}),
                                dcc.Dropdown(
                                    id='id-selector',
                                    options=[c.title for c in self.conversations],
                                    value=None,
                                    style={'width': '70%', 'display': 'inline-block'}
                                ),                                
                            ])

                        ], style={'width': '100%', 'align-items': 'stretch', 'align-content': 'center'})
                    ], style={'align-items': 'stretch', 'justify-content': 'flex-start'})
                ]),
                html.Div(className="graph-container", children=[dcc.Graph(id='main-figure', figure=self._create_figure(), style={'height': '70vh'}),]),
            ])
        ]
        host_: str = host or self.settings.host
        port_: int = port or self.settings.port
        dash_app.run(host=host_, port=str(port_))
