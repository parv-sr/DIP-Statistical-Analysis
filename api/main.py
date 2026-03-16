from fastapi import FastAPI
from src.pipeline import main

app = FastAPI()

@app.get("/")
def startup():
    return "Hello world!"

@app.get("/run")
def run_model():
    print("Calling pipeline API...")
    txt = main()
    return txt