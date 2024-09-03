from pydantic import BaseModel, field_validator, ValidationError
from typing import Literal
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


class Round:
    embedding: Embedding
    tsne_embedding: Embedding
    trimmed: bool = False

    def __init__(self, user_message: LLMMessage, assistant_message: LLMMessage | None = None) -> None:
        self.user_message = user_message
        self.assistant_message = assistant_message
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


class Conversation:
    similarities: list[float]
    min_similarity: float
    max_similarity: float
    avg_similarity: float
    first_last_similarity_difference: float

    def __init__(self, rounds: list[Round], title: str) -> None:
        self.title = title
        self.rounds = rounds
