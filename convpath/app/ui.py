import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output, State
from dash.exceptions import PreventUpdate
from ..color_logger import get_logger
from ..settings import Settings
from ..utils import path_to_resource
from ..package_types import Conversation, Embedding


class AppUI:
    LINE_OPACITY = 0.7
    LINE_WIDTH = 1

    NEIGHBORS_COLOR = 'orange'
    HIGHLIGHT_COLOR = 'magenta'
    FIGURES_CONFIG = {
        'similarities_averages': {'size': 8, 'color': 'navy', 'symbol': 'square'},
        'error_bars': {'size': 8, 'color': 'navy', 'symbol': 'square'},
        'first_last_similarities': {'size': 8, 'color':'navy', 'symbol': 'x'},
        'tsne': {'size': 8, 'color':'#aaaaaa'},
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
    
    def __plot_left_figures_specific_point(self, 
                                           figure: go.Figure,
                                           *, 
                                           title: str, 
                                           value: float, 
                                           index: int, 
                                           color: str,
                                           size: int, 
                                           symbol: str, 
                                           row: int, 
                                           col: int,
                                           pos_err: float = 0.,
                                           neg_err: float = 0.,
                                           ) -> go.Figure:
        plot = go.Scatter(
            x=[self._x_axis()[index]],
            y=[value],
            mode='markers',
            marker_symbol=symbol,
            marker_color=color,
            marker_size=size,
            ids=[title],
            error_y={
                'type': 'data',
                'symmetric': False,
                'array': [pos_err],
                'arrayminus': [neg_err],
            }
        )
        figure.add_trace(plot, row=row, col=col)
        return figure

    def _update_similarities_averages_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int, neighbors: list[str]) -> go.Figure: 
        neighbors_i_v = [(i, c.avg_similarity) for i, c in enumerate(self.conversations) if c.title in neighbors]
        i, v = [(i, c.avg_similarity) for i, c in enumerate(self.conversations) if c.title == highlighted_id][0]
        size = self.FIGURES_CONFIG['similarities_averages']['size']
        symbol = self.FIGURES_CONFIG['similarities_averages']['symbol']
        
        for index, value in neighbors_i_v:
            figure = self.__plot_left_figures_specific_point(figure, title=self.conversations[index].title, value=value, index=index, row=row, col=col, size=size, symbol=symbol, color=self.NEIGHBORS_COLOR)        
        figure = self.__plot_left_figures_specific_point(figure, title=highlighted_id, value=v, index=i, row=row, col=col, symbol=symbol, size=2*size, color=self.HIGHLIGHT_COLOR)

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
    
    def _update_error_bars_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int, neighbors: list[str]) -> go.Figure: 
        i, pos_error, neg_error = [(i, c.max_similarity - c.avg_similarity, c.avg_similarity - c.min_similarity) for i, c in enumerate(self.conversations) if c.title == highlighted_id][0]
        y = sum([c.avg_similarity for c in self.conversations]) / len(self.conversations)
        neighbors_values = [(i, c.max_similarity - c.avg_similarity, c.avg_similarity - c.min_similarity) for i, c in enumerate(self.conversations) if c.title in neighbors]
        size = self.FIGURES_CONFIG['error_bars']['size']
        symbol = self.FIGURES_CONFIG['error_bars']['symbol']
        
        for n, pos_err, neg_err in neighbors_values:
            figure = self.__plot_left_figures_specific_point(figure, title=self.conversations[n].title, value=y, index=n, row=row, col=col, size=size, symbol=symbol, color=self.NEIGHBORS_COLOR, pos_err=pos_err, neg_err=neg_err)        
        figure = self.__plot_left_figures_specific_point(figure, title=highlighted_id, value=y, index=i, row=row, col=col, symbol=symbol, size=2*size, color=self.HIGHLIGHT_COLOR, pos_err=pos_error, neg_err=neg_error)

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
    
    def _update_first_last_similarities_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int, neighbors: list[str]) -> go.Figure: 
        i, v = [(i, c.first_last_similarity_difference) for i, c in enumerate(self.conversations) if c.title == highlighted_id][0]
        neighbors_i_v = [(i, c.first_last_similarity_difference) for i, c in enumerate(self.conversations) if c.title in neighbors]
        size = self.FIGURES_CONFIG['first_last_similarities']['size']
        symbol = self.FIGURES_CONFIG['first_last_similarities']['symbol']
        
        for index, value in neighbors_i_v:
            figure = self.__plot_left_figures_specific_point(figure, title=self.conversations[index].title, value=value, index=index, row=row, col=col, size=size, symbol=symbol, color=self.NEIGHBORS_COLOR)        
        figure = self.__plot_left_figures_specific_point(figure, title=highlighted_id, value=v, index=i, row=row, col=col, symbol=symbol, size=2*size, color=self.HIGHLIGHT_COLOR)

        return figure
    
    def _plot_static_tsne(self, figure: go.Figure, row: int, col: int) -> go.Figure:
        titles_and_tsne_embeddings = [(c.title, [r.tsne_embedding for r in c.steps]) for c in self.conversations]

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
    
    def _update_tsne_with_highlighted_id(self, figure: go.Figure, highlighted_id: str, row: int, col: int, neighbors: list[str]) -> go.Figure: 
        highlighted_conversation = [c for c in self.conversations if c.title == highlighted_id][0]
        highlighted_embeddings = [s.tsne_embedding for s in highlighted_conversation.steps]
        closest_embeddings = [(c.title, [s.tsne_embedding for s in c.steps]) for c in self.conversations if c.title in neighbors]
        
        def plot_path(title: str, embeddings: list[Embedding], color: str):
            nonlocal figure
            x = [e[0] for e in embeddings]
            y = [e[1] for e in embeddings]
            symbols = ["square"] + ["circle"]*len(embeddings[1:-1]) + ["x"]
            for x_, y_, s_ in zip(x, y, symbols):
                figure.add_trace(
                    go.Scatter(
                        x=[x_], y=[y_], 
                        mode="markers", 
                        marker=dict(
                            color=color,
                            size=self.FIGURES_CONFIG['tsne']['size'],
                            symbol=s_
                        ),
                        ids=[title]
                    ),
                    row=row, col=col
                )
            figure.add_trace(
                    go.Scatter(
                        x=x, y=y, 
                        mode="lines", 
                        marker=dict(
                            color=color,
                        ),
                        ids=[title]
                    ),
                    row=row, col=col
                )
        
        for title, embeddings in closest_embeddings:
            plot_path(title=title, embeddings=embeddings, color=self.NEIGHBORS_COLOR)
        plot_path(title=highlighted_conversation.title, embeddings=highlighted_embeddings, color=self.HIGHLIGHT_COLOR)
            
        return figure
    
    def _get_neighbors_ids(self, highlighted_id: str, k: int) -> list[str]:
        highlighted_conversation = [c for c in self.conversations if c.title == highlighted_id][0]
        neighbors = [c for c in self.conversations
                     if c.title in [
                            t[0] for t in highlighted_conversation.closest_conversations_titles_and_distances[0:k]
                        ]
                    ]
        return [n.title for n in neighbors]
    
    def _get_conversation_as_text(self, conversation: Conversation) -> str:
        return '\n'.join(s.as_text(prefix='>> ') for s in conversation.steps)

    def _register_callbacks(self) -> None:
        @callback(
            [
                Output('main-figure', 'figure'), 
                Output('id-selector', 'value'), 
                Output('last-id-displayed', 'data'), 
                Output('clear-button-clicks', 'data'),
                Output('selected-title', 'value'),
                Output('neighbor-titles', 'options'),
                Output('selected-conversation', 'value'),
                Output('last-neighbor-id-shown', 'value'),
                Output('neighbor-conversation', 'value'),
            ], 
            [
                Input('main-figure', 'figure'),
                Input('main-figure', 'clickData'), 
                Input('id-selector', 'value'), 
                Input('clear-selector-button', 'n_clicks'), 
                Input('neighbor-slider', 'value'),
                Input('neighbor-titles', 'value')
            ],
            [
                State('last-id-displayed', 'data'), 
                State('clear-button-clicks', 'data'),
                State('last-neighbor-id-shown', 'value')
            ]
        )
        def update_graph(figure: go.Figure,
                         click_data: dict, 
                         selector_value: str, 
                         button_clicks: int | None, 
                         k: int, 
                         selected_neighbor: str | None,
                         # States
                         last_id: str, 
                         prev_button_clicks: int,
                         last_selected_neighbor: str | None,
                         ) -> tuple[go.Figure, 
                                    str | None, 
                                    str | None, 
                                    int,
                                    str | None,
                                    list[str],
                                    str | None,
                                    str | None,
                                    str | None,
                                    ]:
            if button_clicks is None:
                button_clicks = 0

            if selector_value != last_id:
                hover_id = selector_value
            elif button_clicks > prev_button_clicks:
                hover_id = None
            elif click_data is not None:
                hover_id = click_data['points'][0]['id']
            else:
                raise PreventUpdate()
            
            print(selected_neighbor, last_selected_neighbor)
            if selected_neighbor and selected_neighbor != last_selected_neighbor:  # dropdown of neighbor conversations was changed, not the selected ID
                last_selected_neighbor = selected_neighbor
            elif hover_id is None:
                neighbors = []
                last_selected_neighbor = None
                figure = self.static_figure
            else:
                neighbors = self._get_neighbors_ids(hover_id, k)
                last_selected_neighbor = None
                figure = self.static_figure
                figure = self._update_similarities_averages_with_highlighted_id(figure, hover_id, row=1, col=1, neighbors=neighbors)
                figure = self._update_error_bars_with_highlighted_id(figure, hover_id, row=2, col=1, neighbors=neighbors)
                figure = self._update_first_last_similarities_with_highlighted_id(figure, hover_id, row=3, col=1, neighbors=neighbors)
                figure = self._update_tsne_with_highlighted_id(figure, hover_id, row=1, col=2, neighbors=neighbors)
                figure.update_traces(hovertemplate='%{id}<extra></extra>')

            return (figure, hover_id, hover_id, button_clicks, hover_id, neighbors, 
                    [self._get_conversation_as_text(c) for c in self.conversations if c.title == hover_id][0] if hover_id else None,
                    last_selected_neighbor, 
                    [self._get_conversation_as_text(c) for c in self.conversations if c.title == selected_neighbor][0] if selected_neighbor else None)

    def _create_static_figure(self) -> go.Figure:
        figure = make_subplots(
            rows=3, cols=2,
            specs=[[{"type": "scatter"}, {"type": "scatter", "rowspan": 3}], 
                   [{"type": "scatter"}, None],
                   [{"type": "scatter"}, None]],
            subplot_titles=["Conversation Similarity", "Paths T-SNE Projections",
                            "Top Min/Max Similarity Across Conversations",
                            "Distance of First and Last Rounds"],
            column_widths=[.6, .4],   
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

        Parameters
        ----------
            host : str | None 
                Hostname to listen on (default: "127.0.0.1").
            port : int | None 
                Port to listen on (default: 8050)
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
            dcc.Store(id='last-neighbor-id-shown', data=None),

            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="left-column", children=[
                        html.Div(className="card", style={'text-align': 'center'}, children=[
                            html.Img(src=dash_app.get_asset_url('logo.png'), style={'height': '100px', 'padding': '25px'})
                        ])
                    ]),
                    html.Div(className="right-column", children=[
                        html.Div(className="card ", children=[
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
                            ]),
                            html.Hr(),
                            html.Label('Number of neighbors to display:'),
                                    html.Div(style={'margin-top': '10px', 'width': '60%'}, children=[
                                        dcc.Slider(
                                            min=0, max=10, step=1, value=3, id='neighbor-slider',
                                        )                                    
                                    ]),
                        ], style={'width': '100%', 'align-items': 'stretch', 'align-content': 'center'})
                    ], style={'align-items': 'stretch', 'justify-content': 'flex-start'})
                ]),
                html.Div(className="graph-container", children=[dcc.Graph(id='main-figure', figure=self.static_figure, style={'height': '80vh'}),]),
                html.Div(className="card", children=[
                    html.Div(className="row", children=[
                        html.Div(style={'width': '50%'}, children=[
                            html.Label("Selected Conversation"),
                            dcc.Textarea(id='selected-title', style={'width': '100%'}, disabled=True),
                        ]),
                        html.Div(style={'width': '50%'}, children=[
                            html.Label("Neighbors"),
                            dcc.Dropdown(id='neighbor-titles', style={'width': '100%'}), 
                        ]),
                    ]),
                    html.Hr(),
                    html.Div(style={'display': 'flex', 'align-items':'stretch'}, children=[
                        dcc.Textarea(id='selected-conversation', style={'width': '50%', 'flex': '1', 'height': '500px'}, disabled=True),
                        dcc.Textarea(id='neighbor-conversation', style={'width': '50%', 'flex': '1', 'height': '500px'}, disabled=True)                        
                    ])

                ]),
            ])
        ]
        host_: str = host or self.settings.host
        port_: int = port or self.settings.port
        dash_app.run(host=host_, port=str(port_))
