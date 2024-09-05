from pydantic import BaseModel, field_validator, ValidationError
from typing import Literal, Any, Sequence
from .constants import USER, ASSISTANT


Color = Literal[
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan"
]


Embedding = list[float]


class LLMMessage(BaseModel):
    role: str
    content: str

    @field_validator("content", mode='before')
    def validate_content(cls, v):
        if not isinstance(v, str) or len(v.strip()) == 0:
            raise ValidationError('Content must be a non-empty string.')
        else:
            return v.strip()
        
    @field_validator("role", mode="before")
    def validate_role(cls, v):
        if not isinstance(v, str) or len(v.strip()) == 0:
            raise ValidationError('Role must be a non-empty string.')
        else:
            return v.strip().upper()


class Step():
    def __init__(self, 
                 user_message: LLMMessage | dict[str, Any], 
                 assistant_message: LLMMessage | dict[str, Any] | None = None,
                 embedding: Embedding = [],
                 tsne_embedding: Embedding = [],
                 trimmed: bool = False
                 ) -> None:
        self.user_message = user_message if isinstance(user_message, LLMMessage) else LLMMessage(**user_message)
        self.assistant_message = assistant_message if (assistant_message is None or isinstance(assistant_message, LLMMessage)) else LLMMessage(**assistant_message)
        self.embedding = embedding
        self.tsne_embedding = tsne_embedding
        self.trimmed = trimmed
        self.id = self._create_id()        

    @property
    def is_paired(self) -> bool:
        return self.assistant_message is not None
    
    def _create_id(self) -> int:
        return hash(self.as_text())

    def as_text(self) -> str:
        if self.assistant_message:
            return f'{USER}: {self.user_message.content} {ASSISTANT}: {self.assistant_message.content}'
        else:
            return self.user_message.content
        
    def to_dict(self) -> dict[str, Any]:
        return {
            'user_message': self.user_message.model_dump(mode='json'), 
            'assistant_message': self.assistant_message.model_dump(mode='json') if self.assistant_message else None,
            'trimmed': self.trimmed,
            'embedding': self.embedding, 
            'tsne_embedding': self.tsne_embedding
        }


class Conversation():
    def __init__(self, 
                 steps: Sequence[Step | dict[str, Any]], 
                 title: str,
                 similarities: list[float] = [],
                 min_similarity: float = 0,
                 max_similarity: float = 0,
                 avg_similarity: float = 0,
                 first_last_similarity_difference: float = 0,
                 closest_conversations_titles_and_distances: Sequence[tuple[str, float] | list] = []
                 ) -> None:
        self.title = title
        self.steps = [step if isinstance(step, Step) else Step(**step) for step in steps]
        self.similarities: list[float] = similarities
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.avg_similarity = avg_similarity
        self.first_last_similarity_difference = first_last_similarity_difference

        if closest_conversations_titles_and_distances:
            if isinstance(closest_conversations_titles_and_distances[0], list):
                closest_conversations_titles_and_distances = [(str(t[0]), float(t[1])) for t in closest_conversations_titles_and_distances]
        self.closest_conversations_titles_and_distances = closest_conversations_titles_and_distances

    def to_dict(self):
        return {
            'title': self.title,
            'steps': [step.to_dict() for step in self.steps],
            'similarities': self.similarities,
            'min_similarity': self.min_similarity,
            'max_similarity': self.max_similarity,
            'avg_similarity': self.avg_similarity,
            'first_last_similarity_difference': self.first_last_similarity_difference,
            'closest_conversations_titles_and_distances': self.closest_conversations_titles_and_distances
        }
