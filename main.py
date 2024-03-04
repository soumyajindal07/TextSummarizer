from fastapi import FastAPI
from transformers import pipeline,AutoModelForSeq2SeqLM
import os
from os import path

app = FastAPI()

@app.get("/isBARTModelAvailable")
def isBARTModelAvailable():
    bartLargeFolder = path.relpath("models/BART-large")
    if(os.path.isdir(bartLargeFolder)):
        return True
    else:
        return False
    
@app.get("/isPegasusModelAvailable")
def isPegasusModelAvailable():
    pegasusSmallFolder = path.relpath("models/PEGASUS-small")
    if(os.path.isdir(pegasusSmallFolder)):
        return True
    else:
        return False

@app.post("/BART-large")
def textSummarizeUsingBART(input:str):
    bartPipeline = pipeline(task='summarization', model="models/BART-large", min_length = 25, max_length = 50)    
    result = bartPipeline(input)
    try:
        return result[0]['summary_text']
    except:
        return "Server Error: Please try again later"

@app.post("/PEGASUS-small")
def textSummarizeUsingPEGASUS(input:str):
    pegasusPipeline = pipeline(task='summarization', model="models/PEGASUS-small",min_length= 25, max_length = 50)    
    result = pegasusPipeline(input)
    try:
        return result[0]['summary_text'].replace("<n>","")
    except:
        return "Server Error: Please try again later"

@app.post("/FACEBOOK-bartbase")
def textSummarizeUsingBARTBase(input:str):
    bartBasePipeline = pipeline(task='summarization', model="Falconsai/text_summarization")    
    result = bartBasePipeline(input,max_length=100, min_length=30, do_sample=False)
    try:
        return result[0]['summary_text']
    except:
        return "Server Error: Please try again later"
