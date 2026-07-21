from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional

class Source(BaseModel):
    tool: str
    url: str
    line_range: Optional[str] = None
    original_text: str

class decisions(BaseModel):
    name  : str
    title: str
    summary: str
    source: Source
    decision: str
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class rules(BaseModel):
    name: str
    summary: str
    source: Source
    notes: str
    scope: str
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class warnings(BaseModel):
    name : str
    area: str
    message : str
    severity : str
    source: Source
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class StructuredQuery(BaseModel):
    """LLM-generated query over the structured data store (see FileStorage).

    Uses fixed, named fields rather than an open dict because OpenAI's
    strict structured-output schema cannot represent arbitrary dict[str, str].
    """
    data_type: Literal["rules", "warnings", "decisions"]
    name: Optional[str] = None       # exact item name, any data_type
    scope: Optional[str] = None      # rules
    area: Optional[str] = None       # warnings
    severity: Optional[str] = None   # warnings
    since: Optional[str] = None      # ISO-8601 timestamp, any data_type
