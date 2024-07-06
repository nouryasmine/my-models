import base64
from io import BytesIO
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from matplotlib.pyplot import draw
from pydantic import BaseModel
from model.model import predict_dipolemoment, predict_gap, predict_homo, generate_molecules, predict_lumo, predict_zpve, smilesToMol, visualize_molecules

#from app.model.model import model
from fastapi.middleware.cors import CORSMiddleware
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
#uvicorn app.main:app 
app= FastAPI()
origins =[
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "https://ai-mol.vercel.app"

]

app.add_middleware(
    CORSMiddleware,
     allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Molecule(BaseModel):
    molecule:str

class PredictionOut(BaseModel):
    gap:float
    homo:float
    lumo:float
    zpve:float
    dipolemoment:float

class MoleculeRequest(BaseModel):
    num_molecules: int = 10

class MoleculeList(BaseModel):
    moleculeslist: List[str]
#C1=CC=CC=C1

@app.get("/")
def home():
    return {"health_check":"ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: Molecule):
    dipolemoment=predict_dipolemoment(payload.molecule)
    lumo = predict_lumo(payload.molecule)
    homo = predict_homo(payload.molecule)
    gap = lumo-homo
    zpve = predict_zpve(payload.molecule)
    
    return {"dipolemoment":dipolemoment, "lumo":lumo, "homo":homo, "gap": gap, "zpve":zpve}

@app.post("/generate_molecules")
def generate_molecules_endpoint(request: MoleculeRequest):
    molecules = generate_molecules()
    return {"molecules": molecules}

@app.post("/visualize_molecules")
async def visualize_mols(request: MoleculeList):
    try:
        logging.info(f"Received request: {request.moleculeslist}")
        images = visualize_molecules(request.moleculeslist)
        logging.info("Generated image successfully")
        return Response(content=images, media_type="image/png")
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/visualize3d")
async def visualize3d(request:MoleculeList):
    try:
        processed_molecules = smilesToMol(request.moleculeslist)
        
        # Generate 3D visualization for each molecule
        pdb_strings = []
        for mol in processed_molecules:
            pdb_str = Chem.MolToPDBBlock(mol)
            pdb_strings.append(pdb_str)
            
        return {"pdb_strings": pdb_strings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))