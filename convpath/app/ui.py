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
    LINE_OPACITY = 0.7
    LINE_WIDTH = 1

    FIGURES_CONFIG = {
        'similarities_averages': {'size': 8, 'color': 'navy', 'symbol': 'square'},
        'error_bars': {'size': 8, 'color': 'navy', 'symbol': 'square'},
        'first_last_similarities': {'size': 8, 'color':'navy', 'symbol': 'x'},
        'tsne': {'size': 8, 'color':'#aaaaaa', 'highlight_color': 'brown'},
    }

    def __init__(self, conversations: list[Conversation]) -> None:
        self.logger = get_logger()
        self.settings = Settings()
        self.conversations = conversations
        self._static_figure = self._create_static_figure()

    @property
    def static_figure(self):
        return go.Figure(self._static_figure)

    def _x_axis(self) -> list[int]:
        return list(range(1, len(self.conversations)+1))
    
    def _plot_static_similarities_averages(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        color = self.FIGURES_CONFIG['similarities_averages']['color']
        plot = go.Scatter(
            x=self._x_axis(),
            y=[c.avg_similarity for c in self.conversations],
            mode='markers',
            marker_symbol=self.FIGURES_CONFIG['similarities_averages']['symbol'],
            marker_color=color,
            marker_size=self.FIGURES_CONFIG['similarities_averages']['size'],
            ids=[c.title for c in self.conversations]
        )
        figure.add_trace(plot, row=row, col=col)

        avg = sum([c.avg_similarity for c in self.conversations]) / len(self.conversations)
        figure.add_hline(y=avg, line_color=color, opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                      # type: ignore

        std = np.std([c.avg_similarity for c in self.conversations])
        if avg + std <= 1:
            figure.add_hline(y=avg + std, line_color=color, line_dash="dash", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)          # type: ignore
        if avg - std >= 0:
            figure.add_hline(y=avg - std, line_color=color, line_dash="dash", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)          # type: ignore
        if avg + (2*std) <= 1:
            figure.add_hline(y=avg + (2*std), line_color=color, line_dash="dot", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)       # type: ignore
        if avg - (2*std) >= 0:
            figure.add_hline(y=avg - (2*std), line_color=color, line_dash="dot", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)       # type: ignore
        
        figure.update_traces(hovertemplate='%{id}<extra></extra>', row=row, col=col)
        figure.update_xaxes(visible=False, row=row, col=col)

        return figure
    
    def _update_similarities_averages_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int) -> go.Figure: 
        i, v = [(i, c.avg_similarity) for i, c in enumerate(self.conversations) if c.title == highlighted_id][0]
        plot = go.Scatter(
            x=[self._x_axis()[i]],
            y=[v],
            mode='markers',
            marker_symbol=self.FIGURES_CONFIG['similarities_averages']['symbol'],
            marker_color=self.FIGURES_CONFIG['similarities_averages']['color'],
            marker_size=2 * self.FIGURES_CONFIG['similarities_averages']['size'],
            ids=[highlighted_id]
        )
        figure.add_trace(plot, row=row, col=col)
        return figure

    def _plot_static_error_bars(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        green_color = 'green'
        red_color = 'red'
        points_color = self.FIGURES_CONFIG['error_bars']['color']

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
            marker_size=self.FIGURES_CONFIG['error_bars']['size'],
            marker_color=points_color,
            marker_symbol=self.FIGURES_CONFIG['error_bars']['symbol'],
            ids=[c.title for c in self.conversations]
        )
        figure.add_trace(points_plot, row=row, col=col)

        avg_green_error = sum(green_error_bars) / len(green_error_bars)
        avg_red_error = sum(red_error_bars) / len(red_error_bars)
        green_errors_std = np.std(green_error_bars)
        red_errors_std = np.std(red_error_bars)
        figure.add_hline(y=avg, line_color=points_color, opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                                            # type: ignore

        figure.add_hline(y=avg + avg_green_error, line_color=green_color, opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                           # type: ignore
        figure.add_hline(y=avg + avg_green_error + green_errors_std, line_color=green_color, line_dash="dash", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)      # type: ignore
        figure.add_hline(y=avg + avg_green_error + (2*green_errors_std), line_color=green_color, line_dash="dot", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)   # type: ignore
        
        figure.add_hline(y=avg - avg_red_error, line_color=red_color, opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                               # type: ignore
        figure.add_hline(y=avg - avg_red_error - red_errors_std, line_color=red_color, line_dash="dash", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)            # type: ignore
        figure.add_hline(y=avg - avg_red_error - (2*red_errors_std), line_color=red_color, line_dash="dot", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)         # type: ignore

        figure.update_traces(hovertemplate='%{id}<extra></extra>', row=row, col=col)
        figure.update_xaxes(visible=False, row=row, col=col)

        return figure
    
    def _update_error_bars_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int) -> go.Figure: 
        i = [i for i, c in enumerate(self.conversations) if c.title == highlighted_id][0]
        plot = go.Scatter(
            x=[self._x_axis()[i]],
            y=[sum([c.avg_similarity for c in self.conversations]) / len(self.conversations)],
            mode='markers',
            marker_symbol=self.FIGURES_CONFIG['error_bars']['symbol'],
            marker_color=self.FIGURES_CONFIG['error_bars']['color'],
            marker_size=2 * self.FIGURES_CONFIG['error_bars']['size'],
            ids=[highlighted_id]
        )
        figure.add_trace(plot, row=row, col=col)
        return figure
    
    def _plot_static_first_last_similarities(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        color = self.FIGURES_CONFIG['first_last_similarities']['color']
        abs_first_last = [abs(c.first_last_similarity_difference) for c in self.conversations]
        plot = go.Scatter(
            x=self._x_axis(),
            y=abs_first_last,
            mode='markers',
            marker_symbol=self.FIGURES_CONFIG['first_last_similarities']['symbol'],
            marker_color=color,
            marker_size=self.FIGURES_CONFIG['first_last_similarities']['size'],
            ids=[c.title for c in self.conversations]
        )
        figure.add_trace(plot, row=row, col=col)

        avg = sum(abs_first_last) / len(self.conversations)
        figure.add_hline(y=avg, line_color=color, opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)                                  # type: ignore

        std = np.std(abs_first_last)
        if avg + std <= 1:
            figure.add_hline(y=avg + std, line_color=color, line_dash="dash", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)      # type: ignore
        if avg - std >= 0:
            figure.add_hline(y=avg - std, line_color=color, line_dash="dash", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)      # type: ignore
        if avg + (2*std) <= 1:
            figure.add_hline(y=avg + (2*std), line_color=color, line_dash="dot", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)   # type: ignore
        if avg - (2*std) >= 0:
            figure.add_hline(y=avg - (2*std), line_color=color, line_dash="dot", opacity=self.LINE_OPACITY, line_width=self.LINE_WIDTH, row=row, col=col)   # type: ignore

        figure.update_traces(hovertemplate='%{id}<extra></extra>', row=row, col=col)
        figure.update_xaxes(visible=False, row=row, col=col)
        
        return figure
    
    def _update_first_last_similarities_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int) -> go.Figure: 
        i, v = [(i, c.first_last_similarity_difference) for i, c in enumerate(self.conversations) if c.title == highlighted_id][0]
        plot = go.Scatter(
            x=[self._x_axis()[i]],
            y=[v],
            mode='markers',
            marker_symbol=self.FIGURES_CONFIG['first_last_similarities']['symbol'],
            marker_color=self.FIGURES_CONFIG['first_last_similarities']['color'],
            marker_size=2 * self.FIGURES_CONFIG['first_last_similarities']['size'],
            ids=[highlighted_id]
        )
        figure.add_trace(plot, row=row, col=col)
        return figure
    
    def _plot_static_tsne(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        titles_and_tsne_embeddings = [(c.title, [r.tsne_embedding for r in c.rounds]) for c in self.conversations]

        for title, conversation_embeddings in titles_and_tsne_embeddings:
            x = [e[0] for e in conversation_embeddings]
            y = [e[1] for e in conversation_embeddings]
            symbols = ["square"] + ["circle"]*len(conversation_embeddings[1:-1]) + ["x"]
            for x_, y_, s_ in zip(x, y, symbols):
                figure.add_trace(
                    go.Scatter(
                        x=[x_], y=[y_], 
                        mode="markers", 
                        marker=dict(
                            color=self.FIGURES_CONFIG['tsne']['color'],
                            size=self.FIGURES_CONFIG['tsne']['size'],
                            symbol=s_
                        ),
                        ids=[title]
                    ),
                    row=row, col=col
                )
                
        figure.update_traces(hovertemplate='%{id}<extra></extra>', row=row, col=col)
        return figure
    
    def _update_tsne_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int) -> go.Figure: 
        highlighted_embeddings = [[r.tsne_embedding for r in c.rounds] for c in self.conversations if c.title == highlighted_id][0]
        x = [e[0] for e in highlighted_embeddings]
        y = [e[1] for e in highlighted_embeddings]
        symbols = ["square"] + ["circle"]*len(highlighted_embeddings[1:-1]) + ["x"]
        for x_, y_, s_ in zip(x, y, symbols):
            figure.add_trace(
                go.Scatter(
                    x=[x_], y=[y_], 
                    mode="markers", 
                    marker=dict(
                        color=self.FIGURES_CONFIG['tsne']['highlight_color'],
                        size=self.FIGURES_CONFIG['tsne']['size'],
                        symbol=s_
                    ),
                    ids=[highlighted_id]
                ),
                row=row, col=col
            )
        figure.add_trace(
                go.Scatter(
                    x=x, y=y, 
                    mode="lines", 
                    marker=dict(
                        color=self.FIGURES_CONFIG['tsne']['highlight_color'],
                    ),
                    ids=[highlighted_id]
                ),
                row=row, col=col
            )
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
            
            if hover_id is None:
                figure = self.static_figure
            else:
                figure = self.static_figure
                figure = self._update_similarities_averages_with_highlighted_id(figure, hover_id, row=1, col=1)
                figure = self._update_error_bars_with_highlighted_id(figure, hover_id, row=2, col=1)
                figure = self._update_first_last_similarities_with_highlighted_id(figure, hover_id, row=3, col=1)
                figure = self._update_tsne_with_highlighted_id(figure, hover_id, row=1, col=2)
            return (figure, hover_id, hover_id, button_clicks)

    def _create_static_figure(self) -> go.Figure:
        figure = make_subplots(
            rows=3, cols=2,
            specs=[[{"type": "scatter"}, {"type": "scatter", "rowspan": 3}], 
                   [{"type": "scatter"}, None],
                   [{"type": "scatter"}, None]],
            subplot_titles=["Conversation Similarity", "Paths T-SNE Projections",
                            "Top Min/Max Similarity Across Conversations",
                            "Distance of First and Last Rounds"],
            column_widths=[.5, .5],   
            row_heights=[1./3] * 3, 
            horizontal_spacing=0.05,  
            vertical_spacing=0.05 
        )
        figure = self._plot_static_similarities_averages(figure, row=1, col=1)
        figure = self._plot_static_error_bars(figure, row=2, col=1)
        figure = self._plot_static_first_last_similarities(figure, row=3, col=1)
        figure = self._plot_static_tsne(figure, row=1, col=2)
        figure.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='#fbfbfb')
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
                html.Div(className="graph-container", children=[dcc.Graph(id='main-figure', figure=self.static_figure, style={'height': '80vh'}),]),
            ])
        ]
        host_: str = host or self.settings.host
        port_: int = port or self.settings.port
        dash_app.run(host=host_, port=str(port_))
