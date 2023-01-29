from loguru import logger

from fastapi import APIRouter
from pydantic import BaseModel
import arguments
import torch
from run import inference

api_router = APIRouter()

arg_parser = arguments.get_arg_parser('juice')
args = arg_parser.parse_args()
args.cuda = not args.cpu and torch.cuda.is_available()

# load model here
MODEL = 1


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
    message = inference(args,item.natural_language,item.local_code_context)
    response = dict()
    response['message'] = message
    return response
