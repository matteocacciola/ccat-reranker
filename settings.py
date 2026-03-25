from cat import plugin
from pydantic import BaseModel


class MySettings(BaseModel):
    litm: bool = False
    sbert: bool = False
    ranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@plugin
def settings_schema():
    return MySettings.model_json_schema()
