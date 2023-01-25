from typing import Optional
from fastapi import FastAPI

from fastapi import APIRouter, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import time
from utils import log as fulog

from api_handler import api_router

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
# 
# 
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

def app_setting():
    # log setting
    time_stamp = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())
    log_file_path = './logs/app_{}.log'.format(time_stamp)
    logger = fulog.setup_logger(
        log_file_path=log_file_path,
        level='INFO',
        rotation='500 MB',
        retention='10 days',
    )


    logger.info('Init the engine')

    app = FastAPI(
        title='plotcoder-api',
        version='1.0.0',
        description='UI for chatbot',
        debug=False,
    )

    logger.info('Started the engine')

    # setting logger
    app.logger = logger

    # add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    # setting router
    app.include_router(api_router, prefix='/api')
    return app


with logger.catch():
    app = app_setting()
