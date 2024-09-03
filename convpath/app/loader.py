import heapq
import tiktoken
import numpy as np
from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from rich.progress import track
from rich.console import Console
from typing import Any, cast
from ..mathematics import cosine_similarity, dynamic_time_warping_distance
from ..color_logger import get_logger
from ..settings import Settings
from ..utils import flatten, deflatten
from ..package_types import Conversation, LLMMessage, Step, Embedding
from ..constants import USER, ASSISTANT


class DataLoader:
    SINGLE = 'SINGLE'

    def __init__(self, 
                 *,
                 embedding_model: str,
                 max_tokens: int | None,
                 base_url: str | None,
                 api_key: str | None):
        """
        Initialize a Data Loader.

        Args:
            embedding_model (str): The name of the embedding model to use.
            max_tokens (int | None): The maximum number of tokens accepted by the embedding model. This assists the loader in reducing calls to the embedding model by combining several 
                                     texts in a single call.
                                     If None, the loader will use pre-configured settings based on the model name. If it cannot find the model name in the pre-configured settings, 
                                     it will not stack texts t reduce calls to the model, but will call the model for each text individually.
            base_url (str | None): The base URL to use with OpenAI SDK. If None, OpenAI's default is used.
            api_key (str | None): The API key to use with OpenAI SDK. If None, OpenAI's default is used.
        """
        self.logger = get_logger()
        self.settings = Settings()
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens or self.find_max_tokens_per_model(embedding_model)

    @staticmethod
    def find_max_tokens_per_model(model: str) -> int | None:
        """
        Returns the maximum number of tokens an embedding model can receive in a single call.
        If the model is unknown, returns None.

        Args:
            model (str): The name of the embedding model to use.

        Returns:
            (int | None): The maximum number of tokens, if known. Otherwise, returns None.
        """
        match model:
            case 'text-embedding-ada-002' | 'text-embedding-3-small' | 'text-embedding-3-large':
                return 8191
            case _:
                return None

    def load_and_process_conversations(self, 
                                       conversations: list[list[dict[str, Any] | str]], 
                                       titles: list[str] = []
                                       ) -> list[Conversation]:
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

        Returns:
            (list[Conversation]): A list of Conversation objects.
        """
        if not titles:
            titles = [''] * len(conversations)
        elif len(titles) != len(conversations):
            raise ValueError('The number of titles must match the number of conversations.')
        elif len(titles) != len(set(titles)):
            raise ValueError('Titles must be unique.')
        
        console = Console()
        console.log("[bold green]Preparing conversations...")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._load_single_conversation, c, t) for c, t in zip(conversations, titles)]
            loaded = [c for f in track(futures, description="Loading conversations") if (c := f.result())]
        console.print(f'[green]Loaded {len(loaded)}/{len(conversations)} conversations' + 
                      f' ([red]{len(conversations)-len(loaded)} filtered)' if len(loaded)!= len(conversations) else '')
        
        loaded, total_tokens = self._create_embeddings(loaded)
        console.print(f"[yellow]Embedded {total_tokens} tokens")

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._compute_single_conversation_similarities, c) for c in loaded]
            loaded = [f.result() for f in track(futures, description="Computing similarities")]

        loaded = self._find_k_closest_paths(loaded)

        with console.status('[green]Fitting T-SNE...'):
            loaded = self._create_tsne_embeddings(loaded)
        console.print("[green]Created TSNE embeddings")

        console.log('[bold green]Done.')
        return loaded

    def _load_single_conversation(self, conversation: list[dict[str, Any] | str], title: str | None) -> Conversation | None:
        if isinstance(conversation[0], str):
            messages = [LLMMessage(role=self.SINGLE, content=m) for m in cast(list[str], conversation)]
            steps = [self._create_step_from_messages(user_message=m) for m in messages]
        else:
            messages = [LLMMessage(**m) for m in cast(list[dict[str, Any]], conversation)]
            messages = [m for m in messages if m.role in [USER, ASSISTANT]]
            if not (all(m.role == USER for m in messages[::2]) and \
                    all(m.role == ASSISTANT for m in messages[1::2])):
                raise ValueError('Conversation must be alternating between `user` and `assistant`, (`user` goes first)')
            steps = [self._create_step_from_messages(user_message=messages[i], assistant_message=messages[i+1]) 
                    for i in range(0, len(messages), 2)]
        
        steps = [s for s in steps if s]
        if len(steps) < self.settings.min_steps:
            return None
        else:
            if not title:
                title = str(steps[0].id)
            return Conversation(title=title, steps=steps)
        
    def _get_embeddings_from_openai(self, texts: list[str]) -> list[Embedding]:
        texts = [text.replace('\n', ' ') for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.embedding_model).data
        return [emb.embedding for emb in response]
        
    def _create_embeddings(self, conversations: list[Conversation]) -> tuple[list[Conversation], int]:
        encoder = tiktoken.encoding_for_model(self.embedding_model)
        total_tokens = 0
        texts_cache: list[tuple[int, int, int, str]] = []   # (conversation index, step index, number of tokens, text)

        def create_embeddings_and_clear_texts_cache():
            nonlocal texts_cache, total_tokens
            embeddings = self._get_embeddings_from_openai([t[-1] for t in texts_cache])
            for t, emb in zip(texts_cache, embeddings):
                conversations[t[0]].steps[t[1]].embedding = emb
                total_tokens += t[2]
            texts_cache = []

        for i, c in enumerate(track(conversations, description='Creating embeddings')):
            for j, step in enumerate(c.steps):
                step_text = step.as_text()
                tokens = encoder.encode(step_text)
                if self.max_tokens is None:
                    embedding: Embedding = self._get_embeddings_from_openai([step_text])[0]
                    step.embedding = embedding
                    total_tokens += len(tokens)
                elif len(tokens) > self.max_tokens:
                    trimmed_tokens = tokens[:self.max_tokens]
                    trimmed_step_text = encoder.decode(trimmed_tokens)
                    embedding: Embedding = self._get_embeddings_from_openai([trimmed_step_text])[0]
                    step.embedding = embedding
                    step.trimmed = True
                    total_tokens += len(trimmed_tokens)
                else:
                    tokens_so_far = sum(t[2] for t in texts_cache)
                    if tokens_so_far + len(tokens) > self.max_tokens:
                        create_embeddings_and_clear_texts_cache()
                    texts_cache.append((i, j, len(tokens), step_text))
        if texts_cache:
            create_embeddings_and_clear_texts_cache()
        return conversations, total_tokens

    def _create_step_from_messages(self, user_message: LLMMessage, assistant_message: LLMMessage | None = None) -> Step | None:
        all_words = f'{user_message.content} {assistant_message.content}' if assistant_message else user_message.content
        if len(all_words.split(' ')) < self.settings.min_words_per_step:
            return None
        return Step(user_message=user_message, assistant_message=assistant_message)
    
    def _compute_single_conversation_similarities(self, conversation: Conversation) -> Conversation:
        embeddings = [s.embedding for s in conversation.steps]
        similarities: list[float] = [cosine_similarity(embeddings[i], embeddings[i+1]) for i in range(len(embeddings) - 1)]
        conversation.similarities = similarities
        conversation.min_similarity = min(similarities)
        conversation.max_similarity = max(similarities)
        conversation.avg_similarity = sum(similarities) / len(similarities)
        conversation.first_last_similarity_difference = cosine_similarity(embeddings[0], embeddings[-1])
        return conversation

    def _create_tsne_embeddings(self, conversations: list[Conversation]) -> list[Conversation]:
        embeddings: list[list[Embedding]] = [[r.embedding for r in c.steps] for c in conversations]
        array = np.array(flatten(embeddings))
        scaled_array = StandardScaler().fit_transform(array)
        tsne_embeddings: list[Embedding] = TSNE(n_components=2).fit_transform(scaled_array).tolist()
        deflatten_tsne_embeddings = deflatten(tsne_embeddings, embeddings)
        for i in range(len(conversations)):
            for j in range(len(conversations[i].steps)):
                conversations[i].steps[j].tsne_embedding = deflatten_tsne_embeddings[i][j]
        return conversations
    
    def _compute_conversation_dtw_distances(self, conversations: list[Conversation], index: int) -> tuple[int, list[tuple[int, float]]]:
        conversation = conversations[index]
        distances = [(i, dynamic_time_warping_distance([s.embedding for s in conversation.steps], [s.embedding for s in other_conversation.steps])) 
                     for i, other_conversation in enumerate(conversations) if i != index]
        return index, heapq.nsmallest(len(conversations)-1, distances, key=lambda x: x[1])

    def _find_k_closest_paths(self, conversations: list[Conversation]) -> list[Conversation]:        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._compute_conversation_dtw_distances, conversations, i) for i in range(len(conversations))]
            results = [f.result() for f in track(futures, description='Computing distances')]

        for i, lst in results:
            titles_and_distances = [(conversations[j].title, dist) for j, dist in lst]
            conversations[i].closest_conversations_titles_and_distances = titles_and_distances
        
        return conversations
    