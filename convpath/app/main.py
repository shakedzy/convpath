from typing import Any
from .loader import DataLoader
from .ui import AppUI
from ..package_types import Conversation


class App:
    def __init__(self, 
                 *,
                 embedding_model: str,
                 max_tokens: int = 8190,
                 base_url: str | None = None,
                 api_key: str | None = None):
        """
        Initialize an App.

        Args:
            embedding_model (str): The name of the embedding model to use.
            max_tokens (int): The maximum number of tokens accepted by the embedding model.
            base_url (str | None): The base URL to use with OpenAI SDK. If None, OpenAI's default is used.
            api_key (str | None): The API key to use with OpenAI SDK. If None, OpenAI's default is used.
        """
        self.conversations: list[Conversation]
        self.loader = DataLoader(embedding_model=embedding_model, max_tokens=max_tokens, base_url=base_url, api_key=api_key)

    def load_and_process(self, 
                         conversations: list[list[dict[str, Any] | str]], 
                         titles: list[str] = []
                         ) -> None:
        """
        Loads given conversations and preprocess them.

        Args:
            conversations (list): A list of lists of dicts or strings. 
                            If dicts, each dict is a message in the conversation with an LLM.
                            The keys must be 'role' and 'content' and the values must be strings.
                            The 'role' key must be one of 'user' or 'assistant' (case-insensitive).
                            If strings, each string is assumed to be a standalone message.
                            Each conversations must be made either from dicts or strings, not a mix of both.
            titles (list):  An optional list of strings. Each string is the title for each conversation in conversations.
        """
        self.conversations = self.loader.load_and_process_conversations(conversations, titles)
        
    def launch(self, *, host: str | None = None, port: int | None = None) -> None:
        """
        Launch the App server.

        Args:
            host: Hostname to listen on (default: "127.0.0.1").
            port: Port to listen on (default: 8050)
        """
        ui = AppUI(self.conversations)
        ui.launch(host, port)