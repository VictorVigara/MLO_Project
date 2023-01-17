from fastapi import FastAPI, Query
from http import HTTPStatus
from enum import Enum
import requests
import regex as re
from pydantic import BaseModel
from typing import Dict, Any
from fastapi import UploadFile, File
from typing import Optional
import cv2
from fastapi.responses import FileResponse


app = FastAPI()

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



class ItemEnum(Enum):
   alexnet = "alexnet"
   resnet = "resnet"
   lenet = "lenet"

@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
   return {"item_id": item_id}

@app.get("/query_items")
def read_item(item_id: int, skip: int = Query(0, ge=0)):
    return {"item_id": item_id, "skip": skip}


#Invoke-WebRequest -Uri "http://localhost:8000/login?username=victor&password=vigara" -Method POST
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

class Data(BaseModel):
    email: str
    domain_match: str


# http://localhost:8000/text_model/?data=myemail@example.com
@app.post("/text_model/")
def contains_email(data: Data):
    email = data["email"]
    domain_match = data["domain_match"]
    regex = r'\b[A-Za-z0-9._%+-]+@' + domain_match + '\.[A-Z|a-z]{2,}\b'
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, email) is not None
    }
    return response

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()
        
    import cv2
    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))
    cv2.imwrite("image_resize.jpg",res)
    return FileResponse('image_resize.jpg')




