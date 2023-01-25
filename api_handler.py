from loguru import logger

from fastapi import APIRouter, Form
from starlette.requests import Request


api_router = APIRouter()


# load model here
MODEL = 1


@api_router.get('/welcome/')
def welcome():
    result_dict = dict()
    result_dict['Message'] = 'Welcome to plotcoder-api System'
    return result_dict


@api_router.post('/infer/plotcoder/')
def plotcode(
    request: Request,
    natural_language: str = Form(...),
    local_code_content: str = Form(...),
    dataframe_schema: str = Form(...),
):
    # Run model here
    # message = model(natural_language, local_code_content, dataframe_schema)
    message = 'Result Template'
    response = dict()
    response['message'] = message
    return response
