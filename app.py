from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

Model_Name='meta-llama/Llama-2-7b-chat-hf'

tokenizer=AutoTokenizer.from_pretrained(Model_Name)
model=AutoModelForCausalLM.from_pretrained(Model_Name,torch_dtype=torch.float16, device_map='auto')

app=FastAPI()

class chatrequest(BaseModel):
    message: 'str'

@app.post("/chat")
async def chat_endpoint(req:chatrequest):
    input_text=f"[INST] {req.messages}[/INST]"
    inputs=tokenizer(input_text,return_tensors='pt').to(model.device)


    with torch.nograd():
        output=model.generate(**inputs,max_new_tokens=200,do_sample=True,top_p=0.95,temperature=0.7)

    response_text=tokenizer.decode(output[0],skip_special_token=True)
    return{'response' : response_text}
 
    