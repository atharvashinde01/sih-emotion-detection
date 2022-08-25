from typing import Text
from pydantic import BaseModel

class Base(BaseModel):
    Emotion: str
    Text: str
    Clean_Text: str