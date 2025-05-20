from typing import List, Union

from pydantic import BaseModel, RootModel


class ApiResponse(BaseModel):
    success: bool
    data: Union[int, str, dict, list, None]
    code: int | None = None
    message: str | None = None

    class Config:
        extra = "allow"


class Verse(BaseModel):
    book: str
    chapter: int
    versesText: str
    versesNumRange: str


class VerseListStream(BaseModel):
    verses: List[Verse]


class VerseList(RootModel[List[Verse]]):
    pass


class UserFeedback(Verse):
    feedback: str
