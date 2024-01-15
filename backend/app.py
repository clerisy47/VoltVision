from fastapi import  FastAPI

app = FastAPI()

@app.get('/poop')
def fun():
    return 'poop'
