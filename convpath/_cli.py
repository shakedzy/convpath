import pandas as pd
from argparse import ArgumentParser
from .app import App


def run():
    parser = ArgumentParser()
    parser.add_argument('-l', '--load', dest='load', help="Path to message CSV file to load", type=str, required=False)
    parser.add_argument('-m', '--model', dest='model', help="Embedding model to use", type=str, required=False)
    parser.add_argument('--api-key', dest='api_key', help='API key to use for embedding creation', type=str, required=False, default=None)
    parser.add_argument('--base-url', dest='api_key', help='Base URL to use for embedding creation', type=str, required=False, default=None)
    parser.add_argument('-max-tokens', dest='max_tokens', help="Max tokens of the embedding model", required=False, type=int, default=None)
    parser.add_argument('-lp', '--load-prepared', dest='prepared', help="Path to preprocessed messages JSON file to load", type=str, required=False)
    parser.add_argument('-s', '--save', dest='save', help="Path to file to save preprocessed data", type=str, required=False)
    parser.add_argument('--host', dest='host', help="Host of the app", type=str, required=False, default='127.0.0.1')
    parser.add_argument('--port', dest='port', help="Port of the app", type=int, required=False, default=8050)
    args = parser.parse_args()

    if not args.load and not args.prepared:
        raise ValueError('No input file specified')
    
    app = App()
    if args.load:
        if not args.model:
            raise ValueError('No model specified')
        df = pd.read_csv(args.load, header=0)
        app.load_and_process(
            conversations=df['messages'].to_list(), 
            titles=df['title'].to_list(),
            embedding_model=args.model,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            base_url=args.base_url
        )
    elif args.prepared:
        app.load_prepared(args.prepared)
    
    if args.save:
        app.save(args.save)
    
    app.launch(host=args.host, port=args.port)
