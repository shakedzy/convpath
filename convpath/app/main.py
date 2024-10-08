from typing import Any
from .loader import DataLoader
from .ui import AppUI
from ..package_types import Conversation


class App:
    def __init__(self):
        """
        Initialize an App.
        """
        self.conversations: list[Conversation]
        self.loader = DataLoader()

    def load_and_process(self, 
                         conversations: list[list[dict[str, Any] | str]], 
                         titles: list[str] = [],
                         *,
                         embedding_model: str,
                         max_tokens: int | None = None,
                         base_url: str | None = None,
                         api_key: str | None = None
                         ) -> None:
        """
        Loads given conversations and preprocess them.

        Parameters
        ----------
            conversations : list
                A list of lists of dicts or strings. 
                If dicts, each dict is a message in the conversation with an LLM.
                The keys must be 'role' and 'content' and the values must be strings.
                The 'role' key must be one of 'user' or 'assistant' (case-insensitive).
                If strings, each string is assumed to be a standalone message.
                Each conversations must be made either from dicts or strings, not a mix of both.
            titles : list[str]
                An optional list of strings. Each string is the title for each conversation in conversations.
            embedding_model : str
                The name of the embedding model to use.
            max_tokens : int | None
                The maximum number of tokens accepted by the embedding model. This assists the loader in reducing calls to the embedding model by combining several 
                texts in a single call.
                If None, the loader will use pre-configured settings based on the model name. If it cannot find the model name in the pre-configured settings, 
                it will not stack texts t reduce calls to the model, but will call the model for each text individually.
            base_url : str | None
                The base URL to use with OpenAI SDK. If None, OpenAI's default is used.
            api_key : str | None
                The API key to use with OpenAI SDK. If None, OpenAI's default is used.
        """
        self.conversations = self.loader.load_and_process_conversations(conversations, titles, embedding_model=embedding_model, max_tokens=max_tokens, base_url=base_url, api_key=api_key)
        
    def launch(self, *, host: str | None = None, port: int | None = None) -> None:
        """
        Launch the App server.

        Parameters
        ----------
            host: str | None
                Hostname to listen on (default: "127.0.0.1").
            port: int | None
                Port to listen on (default: 8050)
        """
        ui = AppUI(self.conversations)
        ui.launch(host, port)

    def save(self, filename: str) -> None:
        """
        Saves all loaded and preprocessed data to a file.

        Parameters
        ----------
            conversations : list[Conversation]
                The conversations to save.
            filename : str
                The filename to save the data under.
        """
        self.loader.save(self.conversations, filename)

    def load_prepared(self, filename: str) -> None:
        """
        Loads preprocessed conversations from a file.

        Parameters
        ----------
            filename : str
                The filename to load the data from.
        """
        self.conversations = self.loader.load_prepared(filename)