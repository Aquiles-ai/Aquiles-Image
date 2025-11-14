from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="dummy")

prompt = """
A high-speed cinematic race through a vast sci-fi landscape of twin suns and drifting asteroids. Sleek spacecrafts weave and twist between glowing debris fields and towering metallic canyons, their engines flaring with neon blue light. The camera tracks dynamically from cockpit to wide aerial shots, capturing motion blur, heat distortion, and lens flares as racers skim just meters above the surface. Sparks and vapor trails arc behind them as they bank through tight turns at breakneck speed. The soundscape roars with thrusters, wind, and faint radio chatter. In the distance, a colossal city hovers above the horizon, lights shimmering through cosmic dust. The scene is epic, cinematic, and kinetic—an adrenaline-fueled ballet of light, speed, and gravity-defying motion.
"""

video = client.videos.create(
    model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    prompt=prompt,
    seconds='4',
    size="1280x720"
)

print(f"Video response: {video}")