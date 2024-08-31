import tiktoken
from openai import OpenAI
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track
from .color_logger import get_logger
from .settings import Settings
from .package_types import Conversation, LLMMessage, Round, Embedding
from .constants import USER, ASSISTANT


class App:
    def __init__(self, 
                 conversations: list[list[dict[str, Any]]], 
                 titles: list[str] = [],
                 *,
                 embedding_model: str,
                 max_tokens: int = 8190,
                 base_url: str | None = None,
                 api_key: str | None = None):
        """
        Initialize a ConvPath App.

        Args:
            conversations (list): A list of lists of dicts. Each dict is a message in the conversation with an LLM.
                                  The keys must be 'role' and 'content' and the values must be strings.
                                  The 'role' key must be one of 'user' or 'assistant' (case-insensitive).
            titles (list):  An optional list of strings. Each string is the title for each conversation in conversations.
            embedding_model (str): The name of the embedding model to use.
            max_tokens (int): The maximum number of tokens accepted by the embedding model.
            base_url (str | None): The base URL to use with OpenAI SDK. If None, OpenAI's default is used.
            api_key (str | None): The API key to use with OpenAI SDK. If None, OpenAI's default is used.
        """
        self.logger = get_logger()
        self.settings = Settings()
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.conversations = self._prepare_conversations(conversations, titles)

    def _prepare_conversations(self, conversations: list[list[dict[str, Any]]], titles: list[str] = []) -> list[Conversation]:
        self.logger.info('Preparing conversations...', color='green')
        if not titles:
            titles = [''] * len(conversations)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._load_single_conversation, c, t) for c, t in zip(conversations, titles)]
            loaded = [c for f in track(futures, description="├ Loading conversations") if (c := f.result())]
        self.logger.info(f'├ Loaded {len(loaded)}/{len(conversations)} conversations.')
        loaded = self._create_embeddings(loaded)
        self.logger.info('Done.', color='green')
        return loaded

    def _load_single_conversation(self, conversation: list[dict[str, Any]], title: str | None) -> Conversation | None:
        messages = [LLMMessage(**m) for m in conversation]
        messages = [m for m in messages if m.role in [USER, ASSISTANT]]
        if not (all(m.role == USER for m in messages[::2]) and \
                all(m.role == ASSISTANT for m in messages[1::2])):
            raise ValueError('Conversation must be alternating between `user` and `assistant`, (`user` goes first)')
        rounds = [self._create_round_from_messages(user_message=messages[i], assistant_message=messages[i+1]) 
                  for i in range(0, len(messages), 2)]
        rounds = [r for r in rounds if r]
        if len(rounds) < self.settings.min_rounds:
            return None
        else:
            if not title:
                title = str(rounds[0].id)
            return Conversation(title=title, rounds=rounds)
        
    def _get_embeddings_from_openai(self, texts: list[str]) -> list[Embedding]:
        texts = [text.replace('\n', ' ') for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.embedding_model).data
        return [emb.embedding for emb in response]
        
    def _create_embeddings(self, conversations: list[Conversation]) -> list[Conversation]:
        encoder = tiktoken.encoding_for_model(self.embedding_model)
        total_tokens = 0
        texts_cache: list[tuple[int, int, int, str]] = []   # (conversation index, round index, number of tokens, text)

        def create_embeddings_and_clear_texts_cache():
            nonlocal texts_cache, total_tokens
            embeddings = self._get_embeddings_from_openai([t[-1] for t in texts_cache])
            for t, emb in zip(texts_cache, embeddings):
                conversations[t[0]].rounds[t[1]].embedding = emb
                total_tokens += t[2]
            texts_cache = []

        for i, c in enumerate(track(conversations, description='├ Creating embeddings')):
            for j, round in enumerate(c.rounds):
                round_text = round.as_text()
                tokens = encoder.encode(round_text)
                if len(tokens) > self.max_tokens:
                    trimmed_tokens = tokens[:self.max_tokens]
                    trimmed_round_text = encoder.decode(trimmed_tokens)
                    embedding: Embedding = self._get_embeddings_from_openai([trimmed_round_text])[0]
                    round.embedding = embedding
                    total_tokens += len(trimmed_tokens)
                else:
                    tokens_so_far = sum(t[2] for t in texts_cache)
                    if tokens_so_far + len(tokens) > self.max_tokens:
                        create_embeddings_and_clear_texts_cache()
                    texts_cache.append((i, j, len(tokens), round_text))
        if texts_cache:
            create_embeddings_and_clear_texts_cache()
        self.logger.info(f'├ Embedded {total_tokens} tokens')
        return conversations

    def _create_round_from_messages(self, user_message: LLMMessage, assistant_message: LLMMessage) -> Round | None:
        if len(f'{user_message.content} {assistant_message.content}'.split(' ')) < self.settings.min_words_per_round:
            return None
        return Round(user_message=user_message, assistant_message=assistant_message)
    