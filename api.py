import pandas as pd
from fastapi import FastAPI
from knn import make_knn
from pydantic import BaseModel

app = FastAPI()

class Body(BaseModel):
    pass_class: int
    sex: int
    age: int
    sibsNspouses: int
    parentsNchildren: int
    fare: float

def dead_or_alive(body: Body):
    knn = make_knn()
    test_person = pd.DataFrame.from_dict([body])
    # print(test_person.shape)
    # print(test_person.dtypes)
    result = knn.predict(test_person)[0]
    result = result.item()
    return result


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/send")
async def send(body: Body):
    return dead_or_alive(body.dict())