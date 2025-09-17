from openai import OpenAI
import base64

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

result = client.images.edit(
    model="black-forest-labs/FLUX.1-Kontext-dev",
    image=open("porsche911.png", "rb"),
    prompt="Add a Hello Kitty sticker to the hood of the car, covering the entire hood.",
    response_format="b64_json"
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("gift-basket.png", "wb") as f:
    f.write(image_bytes)