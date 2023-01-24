from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from input_preprocess import nl_preprocess, context_preprocess
from run import *


class Item(BaseModel):
    context: str
    nl: str

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    item_dict.update({"nl": nl_preprocess(item.nl)})
    item_dict.update({"context": context_preprocess(item.context)})
    result = [item_dict]
    # prediction()
    return result