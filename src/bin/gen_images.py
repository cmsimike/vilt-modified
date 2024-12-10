import io
import base64
import requests
import json
import os
from PIL import Image, PngImagePlugin
import uuid

rootdir = "data/multi-label"


def generate_image(prompt, output_path):

    url = "http://10.5.65.48:7860/"  # change to your SD url

    # Note: You can no longer pass file path in payload as in sd-webui-controlnet.
    # StateDict Keys: {'unet': 1680, 'vae': 248, 'text_encoder': 197, 'text_encoder_2': 518, 'ignore': 0}
    payload = {
        "model": "juggernautXL_juggXIByRundiffusion.safetensors",
        "override_settings": {
            "sd_model_checkpoint": "juggernautXL_juggXIByRundiffusion.safetensors"
        },
        "prompt": prompt,
        "negative_prompt": "bad anatomy, unappetizing, sloppy, unprofessional, comics, cropped, cross-eyed, worst quality, low quality, painting, 3D render, drawing, crayon, sketch, graphite, impressionist, cartoon, anime, noisy, blurry, soft, deformed, ugly, lowres, low details, JPEG artifacts, airbrushed, semi-realistic, CGI, render, Blender, digital art, manga, amateur, mutilated, distorted",
        "sampler_name": "DPM++ 2M",
        "scheduler": "Karras",
        "batch_size": 1,
        "steps": 35,
        "cfg_scale": 7,
        "width": 1024,
        "height": 1024,
        "seed": -1,
    }

    response = requests.post(url=f"{url}/sdapi/v1/txt2img", json=payload)

    r = response.json()

    result = r["images"][0]
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))

    pnginfo = PngImagePlugin.PngInfo()
    if "parameters" in image.info and image.info["parameters"] is not None:
        pnginfo.add_text("parameters", image.info.get("parameters"))

    image.save(os.path.join(output_path, str(uuid.uuid4()) + ".png"), pnginfo=pnginfo)


for entry in os.listdir(rootdir):
    # Create full path
    full_path = os.path.join(rootdir, entry)
    stable_diffusion_prompt_file = os.path.join(
        full_path, "stable_diffusion_prompt.txt"
    )
    if not os.path.exists(stable_diffusion_prompt_file):
        print(f"File does not exist {stable_diffusion_prompt}")
        exit(1)
    with open(stable_diffusion_prompt_file, "r", encoding="utf-8") as f:
        stable_diffusion_prompt = f.read()

    image_output_dir = os.path.join(full_path, "output")
    os.makedirs(image_output_dir, exist_ok=True)

    generate_image(stable_diffusion_prompt, image_output_dir)
