import io
import base64
import requests
import json
import os
from PIL import Image, PngImagePlugin
import uuid


def test():

    url = "http://10.5.65.48:7860/"  # change to your SD url

    # Note: You can no longer pass file path in payload as in sd-webui-controlnet.
    # StateDict Keys: {'unet': 1680, 'vae': 248, 'text_encoder': 197, 'text_encoder_2': 518, 'ignore': 0}
    payload = {
        "model": "juggernautXL_juggXIByRundiffusion.safetensors",
        "override_settings": {
            "sd_model_checkpoint": "juggernautXL_juggXIByRundiffusion.safetensors"
        },
        "prompt": "Food photography of a front-view rustic presentation of hamburger and French fries in a wooden basket, the fries spilling out onto a dark linen cloth beneath. The basket is weathered and worn, adding to the rustic ambiance of the scene. The fries are arranged haphazardly, their crispy edges glistening under soft, ambient light. A sprinkle of sea salt highlights their golden hue and savory aroma. The dark backdrop sets a cozy atmosphere, enhancing the visual appeal of the fries and evoking the comforting feeling of indulging in a classic favorite. ",
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
        # print("\nParameters:\n" + json.dumps(image.info, indent=4))

    os.makedirs("output", exist_ok=True)
    image.save(os.path.join("output", str(uuid.uuid4()) + ".png"), pnginfo=pnginfo)

    print("image saved to test_output.png")


for _ in range(4):
    test()
