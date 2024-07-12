import torch
import numpy as np

from transformers import pipeline
from diffusers.utils import load_image, make_image_grid

image = load_image(
    "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
)

def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map

depth_estimator = pipeline("depth-estimation")
depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

output = pipe(
    "lego batman and robin", image=image, control_image=depth_map,
).images[0]

# depth_map to Image
from PIL import Image
depth_map = depth_map.squeeze(0).cpu().numpy().transpose(1, 2, 0)
depth_map = depth_map[:, :, 0]
depth_map = (depth_map * 255).astype(np.uint8)
depth_map = Image.fromarray(depth_map)

img_grid = make_image_grid([image, depth_map, output], rows=1, cols=3)
img_grid.save("i2i_controlnet.png")




