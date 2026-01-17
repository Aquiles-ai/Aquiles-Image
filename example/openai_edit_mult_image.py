import base64
from openai import OpenAI

client = OpenAI(base_url="https://f4k3r22--aquiles-image-server-serve.modal.run", api_key="dummy-api-key")

prompt = """
Generate a photorealistic image of a gift basket on a white background 
labeled 'Relax & Unwind' with a ribbon and handwriting-like font, 
containing all the items in the reference pictures.
"""

result = client.images.edit(
    model="black-forest-labs/FLUX.2-klein-4B",
    image=[
        open("body-lotion.png", "rb"),
        open("bath-bomb.png", "rb"),
        open("incense-kit.png", "rb"),
        open("soap.png", "rb"),
    ],
    response_format="b64_json",
    prompt=prompt
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("output.png", "wb") as f:
    f.write(image_bytes)