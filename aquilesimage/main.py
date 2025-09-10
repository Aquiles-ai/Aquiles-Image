"""
The goal is to create image generation, editing, and variance endpoints compatible with the OpenAI client.

APIs:

POST /images/variations (create_variation)
POST /images/edits (edit)
POST /images/generations (generate)
"""

from fastapi import FastAPI, UploadFile, File
from aquilesimage.models import CreateImageRequest, ImagesResponse, CreateImageEditRequest, CreateImageVariationRequest

app = FastAPI(title="Aquiles-Image")

@app.post("/images/generations", response_model=ImagesResponse, tags=["Generation"])
async def create_image(input_r: CreateImageRequest):
    pass

@app.post("/images/edits", response_model=ImagesResponse, tags=["Edit"])  
async def create_image_edit(input_r: CreateImageEditRequest, image: UploadFile = File(...), mask:  UploadFile | None = File(default=None)):
    pass

@app.post("/images/variations", response_model=ImagesResponse, tags=["Variations"])
async def create_image_variation(input_r: CreateImageVariationRequest, image: UploadFile = File(...)):
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5500)