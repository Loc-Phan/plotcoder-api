from loguru import logger

from fastapi import APIRouter
from pydantic import BaseModel
import arguments
import torch
import config as conf
from run import inference
import requests
from os import path

api_router = APIRouter()
conf.cuda = not conf.cpu and torch.cuda.is_available()

def download_weight(conf):
    github_link = 'https://github.com/Loc-Phan/plotcoder-api/releases/download/v1.0.0/'
    if not path.exists(conf.load_model):
        logger.info('Download ckpt file')
        model_link = github_link + 'ckpt-00001500'
        r = requests.get(model_link, allow_redirects=True)
        open(conf.load_model, 'wb').write(r.content)
    if not path.exists(conf.code_vocab):
        logger.info('Download code vocab file')
        code_vocab_link = github_link + 'code_vocab.json'
        r = requests.get(code_vocab_link, allow_redirects=True)
        open(conf.code_vocab, 'wb').write(r.content)
    if not path.exists(conf.word_vocab):
        logger.info('Download word vocab file')
        word_vocab_link = github_link + 'code_vocab.json'
        r = requests.get(word_vocab_link, allow_redirects=True)
        open(conf.word_vocab, 'wb').write(r.content)
    pass

download_weight(conf)

@api_router.get('/welcome/')
def welcome():
    result_dict = dict()
    result_dict['Message'] = 'Welcome to plotcoder-api System'
    return result_dict

class Item(BaseModel):
    natural_language: str
    local_code_context: str
    dataframe_schema: str = None

@api_router.post('/infer/plotcoder/')
def plotcode(
    item: Item
):
    # Run model here
    # message = model(natural_language, local_code_content, dataframe_schema)
    message = inference(
            conf, item.natural_language, item.local_code_context
    )
    response = dict()
    response['message'] = message
    return response
