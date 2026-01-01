from openai import OpenAI
import base64

client = OpenAI(base_url="https://f4k3r22--aquiles-image-server-serve.modal.run", api_key="dummy-api-key")

prompt = "A realistic photo of a small European kitchen in the morning, natural light entering through a window, slightly cluttered countertop, everyday objects, realistic shadows, imperfect alignment, shot at eye level, wide-angle 24mm lens"

result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt=prompt,
    size="1024x1024",
    response_format="b64_json"
)

print(f"Downloading image\n")

image_bytes = base64.b64decode(result.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)

print(f"Image downloaded successfully\n")