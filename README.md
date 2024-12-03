<p align="center">
  <img src="https://github.com/shakedzy/convpath/blob/main/convpath/__resources__/assets/logo.png?raw=true">
</p>

[![License](https://img.shields.io/github/license/shakedzy/convpath?style=for-the-badge
)](https://github.com/shakedzy/convpath/blob/master/LICENSE)

**ConvPath** is a tool to compare LLM chats with one another. It allows you to find the most similar conversations between two or more conversations, and 
visualize how conversation change along time. It can assist in finding irregularities or off-topic messages in a conversation, by comparing the embeddings
of messages in the conversation.

## Installation
Simply run:
```
pip install git+https://github.com/shakedzy/convpath.git
```

## Usage
```
usage: convpath [-h] [-l LOAD] [-m MODEL] [--api-key API_KEY] [--base-url API_KEY] [-max-tokens MAX_TOKENS] [-lp PREPARED] [-s SAVE] [--host HOST] [--port PORT]

options:
  -h, --help            show this help message and exit
  -l LOAD, --load LOAD  Path to message CSV file to load
  -m MODEL, --model MODEL
                        Embedding model to use
  --api-key API_KEY     API key to use for embedding creation
  --base-url API_KEY    Base URL to use for embedding creation
  -max-tokens MAX_TOKENS
                        Max tokens of the embedding model
  -lp PREPARED, --load-prepared PREPARED
                        Path to preprocessed messages JSON file to load
  -s SAVE, --save SAVE  Path to file to save preprocessed data
  --host HOST           Host of the app
  --port PORT           Port of the app
```

ConvPath has two main different run modes: `--load` and `--load-prepared`.
When using `--load`, ConvPath will load messages from a CSV file, and create an embedding for each message in the file. 
When using `--load-prepared`, ConvPath will load preprocessed messages from a JSON file, and will use the embeddings that were previously created.

When loading a new messages file, ConvPath expects a CSV file with two columns: `messages` and `title`. Each row is expected to be a standalone chat conversation,
meaning - a list of messages. Messages can be either strings or JSON objects with the keys `role` (only `user` or `assistant`) and `content`.

When processing a new conversation, a an embedding model must be specified.

Once processed and loaded, ConvPath will create a visualization of the conversations,
visible by default on [127.0.0.1:8050](http://localhost:8050).