from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from enum import Enum
import re
from pydantic import BaseModel
import cv2
from fastapi.responses import FileResponse

class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

class DomainEnum(Enum):
    gmail = "gmail"
    hotmail = "hotmail"

class MailItem(BaseModel):
    email: str
    domain_match: DomainEnum

app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}

@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"

@app.post("/text_model/")
def contains_email_domain(data: MailItem):
    """Simple function to check if an email is valid."""
    if data.domain_match is DomainEnum.gmail:
        regex = r"\b[A-Za-z0-9._%+-]+@gmail+\.[A-Z|a-z]{2,}\b"
    if data.domain_match is DomainEnum.hotmail:
        regex = r"\b[A-Za-z0-9._%+-]+@hotmail+\.[A-Z|a-z]{2,}\b"
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data.email) is not None,
    }
    return response


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()
    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (128, 128))
    cv2.imwrite("image.jpg", res)
    FileResponse('image.jpg')
    
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

