import torch
import cv2
import numpy as np
from PIL import Image

import os
import yaml
import folder_paths

import importlib
from contextlib import nullcontext
is_accelerate_available = importlib.util.find_spec("accelerate") is not None
if is_accelerate_available:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from comfy.model_management import load_models_gpu 

from diffusers.loaders.single_file_utils import create_unet_diffusers_config, convert_ldm_unet_checkpoint
from diffusers.loaders.single_file_utils import create_scheduler_from_ldm, create_diffusers_vae_model_from_ldm
from diffusers.models.modeling_utils import load_model_dict_into_meta

from diffusers import UniPCMultistepScheduler

from .brushnet.brushnet import BrushNetModel
from .brushnet.pipeline_brushnet import StableDiffusionBrushNetPipeline
from .brushnet.unet_2d_condition import UNet2DConditionModel

from typing import Tuple

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")

current_directory = os.path.dirname(os.path.abspath(__file__))
original_config_file = os.path.join(current_directory, 'brushnet', 'v1-inference.yaml')
brushnet_config_file = os.path.join(current_directory, 'brushnet', 'brushnet.json')
torch_dtype = torch.float16

class BrushNetLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "brushnet": (inpaint_safetensors(), ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("BRMODEL",)
    RETURN_NAMES = ("brushnet",)

    FUNCTION = "brushnet_loading"

    def brushnet_loading(self, brushnet):
        brushnet_file = os.path.join(folder_paths.models_dir, "inpaint", brushnet)

        with init_empty_weights():
            brushnet_config = BrushNetModel.load_config(brushnet_config_file)
            brushnet_model = BrushNetModel.from_config(brushnet_config)

        brushnet_model = load_checkpoint_and_dispatch(
            brushnet_model,
            brushnet_file,
            device_map="auto",
            max_memory=None,
            offload_folder=None,
            offload_state_dict=False,
            dtype=torch_dtype,
            force_hooks=False,
        )

        print("BrushNet model is loaded")

        return (brushnet_model,)

    
def inpaint_safetensors():
    inpaint_path = os.path.join(folder_paths.models_dir, 'inpaint')
    brushnet_path = os.path.join(inpaint_path, 'brushnet')
    abs_list = []
    for x in os.walk(brushnet_path):
        for name in x[2]:
            if 'safetensors' in name:
                abs_list.append(os.path.join(x[0], name))
    names = []
    for x in abs_list:
        remain = x
        y = ''
        while remain != inpaint_path:
            remain, folder = os.path.split(remain)
            if len(y) > 0:
                y = os.path.join(folder, y)
            else:
                y = folder
        names.append(y)     
    return names


class BrushNetPipeline:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "model": ("MODEL",),
                        "clip": ("CLIP", ),
                        "vae": ("VAE", ),
                        "brushnet": ("BRMODEL", ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("BRPL",)
    RETURN_NAMES = ("BRPL",)

    FUNCTION = "pipeline_loading"

    def pipeline_loading(self, model, clip, vae, brushnet):
        load_models_gpu([model, clip.load_model()])
        checkpoint = model.model.state_dict_for_saving(clip.get_sd(), vae.get_sd(), None)

        with open(original_config_file, "r") as fp:
            original_config_data = fp.read()
        original_config = yaml.safe_load(original_config_data)
        unet = create_unet(original_config, checkpoint, torch_dtype)

        vae = create_diffusers_vae_model_from_ldm(
            'StableDiffusionBrushNetPipeline',
            original_config,
            checkpoint,
            image_size=None,
            scaling_factor=None,
            torch_dtype=None,
            model_type=None,
        )

        scheduler_type = "ddim"
        prediction_type = None
        scheduler_components = create_scheduler_from_ldm(
            'StableDiffusionBrushNetPipeline',
            original_config,
            checkpoint,
            scheduler_type=scheduler_type,
            prediction_type=prediction_type,
            model_type=None,
        )

        pipe = StableDiffusionBrushNetPipeline(
            unet=unet['unet'], 
            vae=vae['vae'], 
            text_encoder=None, 
            tokenizer=None, 
            scheduler=scheduler_components['scheduler'],
            brushnet=brushnet,
            requires_safety_checker=False, 
            safety_checker=None,
            feature_extractor=None
        )  

        pipe.to(dtype=torch_dtype)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        print("BrushNet pipeline is loaded")

        return (pipe,)


class BrushNetInpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "BRPL": ("BRPL", ),
                        "image": ("IMAGE",),
                        "mask": ("MASK",),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "brushnet_inpaint"

    def brushnet_inpaint(self, BRPL, image: torch.Tensor, mask: torch.Tensor, positive, negative, seed: int, 
                         steps: int, scale: float) -> Tuple[torch.Tensor]:

        print("Working on image")

        init_image = image_from_tensor(image[0])
        mask_image = image_from_tensor(mask[0])

        # resize image and mask

        mask_image = 1.*(mask_image>250)[:,:,np.newaxis]
        init_image = init_image * (1-mask_image)

        init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

        print("Inference started")

        pos_embeds = positive
        while isinstance(pos_embeds, list):
            pos_embeds = pos_embeds[0]
            
        neg_embeds = negative
        while isinstance(neg_embeds, list):
            neg_embeds = neg_embeds[0]

        generator = torch.Generator("cuda").manual_seed(seed)
        result = BRPL(
            image=init_image, 
            mask=mask_image, 
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            num_inference_steps=steps, 
            generator=generator,
            brushnet_conditioning_scale=scale
        ).images[0]

        return (torch.stack([numpy_to_tensor(np.array(result)).squeeze(0)]),)


class BlendInpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "image1": ("IMAGE",),
                        "image2": ("IMAGE",),
                        "mask": ("MASK",),
                        "blur": ("INT", {"default": 21, "min": 0, "max": 1000}),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "blend_inpaint"

    def blend_inpaint(self, image1: torch.Tensor, image2: torch.Tensor, mask: torch.Tensor, blur: int) -> Tuple[torch.Tensor]:

        init_image1 = image_from_tensor(image1[0])
        init_image2 = image_from_tensor(image2[0])
        mask_image = image_from_tensor(mask[0])

        mask_image = 1.*(mask_image>250)[:,:,np.newaxis]

        # blur, you can adjust the parameters for better performance
        mask_blurred = cv2.GaussianBlur(mask_image*255, (blur, blur), 0)/255
        mask_blurred = mask_blurred[:,:,np.newaxis]
        mask_image = 1-(1-mask_image) * (1-mask_blurred)

        image_pasted=init_image1 * (1-mask_image) + init_image2*mask_image
        result=image_pasted.astype(init_image1.dtype)
            
        return (torch.stack([numpy_to_tensor(np.array(result)).squeeze(0)]),)



def image_from_tensor(t: torch.Tensor) -> np.ndarray:
    image_np = t.numpy() 
    # Convert the numpy array back to the original range (0-255) and data type (uint8)
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]

def create_unet(original_config, checkpoint, torch_dtype):
    num_in_channels = 4
    image_size = 512
    
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["in_channels"] = num_in_channels
    diffusers_format_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config, extract_ema=False)
    
    ctx = init_empty_weights if is_accelerate_available else nullcontext

    with ctx():
        unet = UNet2DConditionModel(**unet_config)

    if is_accelerate_available:
        unexpected_keys = load_model_dict_into_meta(unet, diffusers_format_unet_checkpoint, dtype=torch_dtype)
        if unet._keys_to_ignore_on_load_unexpected is not None:
            for pat in unet._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            print(
                f"Some weights of the model checkpoint were not used when initializing {unet.__name__}: \n {[', '.join(unexpected_keys)]}"
            )
    else:
        unet.load_state_dict(diffusers_format_unet_checkpoint)

    if torch_dtype is not None:
        unet = unet.to(torch_dtype)

    return {"unet": unet}
