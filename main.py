from fastapi import FastAPI, HTTPException,File,UploadFile, Form, Request
from typing import List
import shutil
import zipfile
import re
import mysql.connector
import sys
import os
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from pathlib import Path
import bcrypt
import requests 
from train3 import *

app=FastAPI()

origins = [
    "http://localhost:3000",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str

def get_db():
    conn = mysql.connector.connect(
        host=os.environ.get("DB_HOST"),
        port=3306,
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        database=os.environ.get("DB_NAME")
    )
    return conn

def create_table_users(conn):
    cursor = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        email VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL
    )
    """
    cursor.execute(create_table_query)

#Register
@app.post("/register")
def register_user(user: UserCreate):
    
    conn=get_db()
    create_table_users(conn)
    cursor=conn.cursor()

    query="SELECT email from users where email = %s"
    values=(user.email,)
    cursor.execute(query, values)
    new_user= cursor.fetchone()

    if new_user is not None:
        raise HTTPException(status_code=400, detail="Email already registered.")

    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    query="INSERT INTO users (username, email, password)  VALUES (%s, %s, %s)"
    values=(user.username, user.email, hashed_password)
    cursor.execute(query, values)
    conn.commit()

    return {"message": "Succesfully registered!"}

#Login
@app.post("/login")
def authenticate_user(email: EmailStr =Form(), password: str= Form()):
    conn= get_db()
    cursor= conn.cursor()
    query="SELECT * FROM users WHERE email = %s"
    values=(email,)
    cursor.execute(query, values)
    user= cursor.fetchone()

    if user is None:
        raise HTTPException(status_code=401, detail="Email not registered.")
    if not bcrypt.checkpw(password.encode("utf-8"), user[3].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid password.")

    return {"message": "Login successful!"}

base_path = r"/app/low-code"

#Data Collection node
@app.post("/upload_zip")
async def upload_zip(zip_file: UploadFile=File(...)):
    
    # Save the uploaded file to the desired path
    file_path = os.path.join(base_path, "src", zip_file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)
    print("saved the uploaded zip.")
    
    extract_path = os.path.join(base_path,"src", os.path.splitext(zip_file.filename)[0])
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(file_path)
    

    # Remove the previous "data" folder if it exists
    extracted_folder = os.path.join(base_path, "src/data")
    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)
    

    # Create a copy of the current extracted folder as "data"
    shutil.copytree(extract_path, extracted_folder)
    
    os.chdir(extracted_folder)
    list=['./train','./test','./val']
    for new_path in list:
        os.mkdir(new_path)
        os.chdir(new_path)
        os.mkdir("./images")
        os.mkdir("./labels")
        os.chdir(extracted_folder)   

    return {'filename': zip_file.content_type}

#Data Preprocessing node
@app.post("/preprocess")
async def preprocessing(ratio: float= File(...), aug_num: int= Form(...)):
    #split function call
    data_path= os.path.join(base_path, "src/data/images") 
    split_data(data_path,ratio)
    os.chdir(base_path)

    aug_path=os.path.join(base_path,"src/aug_data")
    if os.path.exists(aug_path):
        shutil.rmtree(aug_path)
    os.mkdir(aug_path)

    os.chdir(aug_path)
    list=['./train','./test','./val']
    for new_path in list:
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            os.chdir(new_path)
            os.mkdir("./images")
            os.mkdir("./labels")
            os.chdir(aug_path)   

    #aug function call
    augment_data(aug_num)
    return {"message": "Data preprocessing successful."}

@app.get("/images")
async def list_images():
    folder_path = os.path.join(base_path, "src/data/train/images")
    extensions = (".png", ".jpg", ".jpeg")

    # Get the list of all files in the local folder
    files = os.listdir(folder_path)

    # Filter the list of files to include only images with the specified extensions
    images=[file for file in files if file.lower().endswith(extensions)]

    return {"images": images}

@app.delete("/deleteimage/{image_name}")
def delete_image(image_name: str):
    image_path = os.path.join(base_path,"src/data/train/images",image_name)
    label_path =  os.path.join(base_path,"src/data/train/labels",(image_name.split("."))[0]+".json")
    print(image_path)
    try:
        os.remove(image_path)
        os.remove(label_path)
        return {"message": "Image deleted successfully."}
    except OSError:
        return {"message": "Failed to delete the image."}

@app.post("/train")
async def train_model_endpoint(epochs: int = Form(...)):
    epochs = int(epochs)
    train_epoch(epochs)
    print("Number of Epochs:", epochs)
    return {"message": f"Model trained successfully with {epochs} epochs."}

#path for trained model
@app.get("/getModelPath")
def getPath():
    path=os.path.join(os.path.dirname(base_path),"src/facetracker.h5")
    if os.path.exists(path):
        return {"path":path}
    else:
        raise HTTPException(status_code=400,detail="Model file not found.")

@app.get("/getGraphPlot")
def getPlot(plotType:str):
    plot_filename = f"{plotType}.png"
    plot_filename=os.path.join(plot_filename)
    src_path = os.path.join(base_path, 'assets', plot_filename)
    # return{dest_path}
    if os.path.exists(src_path):
        return {"graph": plotType}
    else:
        raise HTTPException(status_code=400, detail="Plot file not found.")
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port="8000")