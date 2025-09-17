from openai import OpenAI
import base64

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

print("Editing an image...")
result = client.images.edit(
    model="black-forest-labs/FLUX.1-Kontext-dev", image=open("vercel.jpeg", "rb"),
    prompt="Hey, remove the triangle next to the word 'Vercel' and change the word 'Vercel' to 'Aquiles-ai'",
    response_format="b64_json")

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

print("Saving the image to output.png...")
# Save the image to a file
with open("output.png", "wb") as f:
    f.write(image_bytes)