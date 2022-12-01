from typing import Union
from __init__ import run_ai
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/ai/{filename}")
def read_item(filename: str, q: Union[str, None] = None):
    print("GET: Fetch data")
    # json_compatible_item_data = jsonable_encoder(run_ai(filename))
    # # return JSONResponse(content=json_compatible_item_data)
    return run_ai(filename)