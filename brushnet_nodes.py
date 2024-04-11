from typing import Tuple
import torch
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import os
import numpy as np
import cv2
from PIL import Image

import folder_paths
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")



class BrushNetLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "brushnet": ([x[0][17:] for x in os.walk("./models/inpaint/")], ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("BRMODEL",)
    RETURN_NAMES = ("brushnet",)

    FUNCTION = "brushnet_loading"

    def brushnet_loading(self, brushnet):
        brushnet_path = "./models/inpaint/"+brushnet
        print("Brushnet... ", end="")
        brush_net = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
        print("loaded")
        return (brush_net,)


class BrushNetPipeline:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "model": (folder_paths.get_filename_list("checkpoints"), ),
                        "brushnet": ("BRMODEL", ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("BRPL",)
    RETURN_NAMES = ("BRPL",)

    FUNCTION = "pipeline_loading"

    def pipeline_loading(self, model, brushnet):
        model_path = "./models/checkpoints/"+model
        print("Model... ", end="")
        pipe = StableDiffusionBrushNetPipeline.from_single_file(
            model_path, 
            brushnet=brushnet, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=False
        )
        pipe.to("cuda")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        print("loaded")
        return (pipe,)



class BrushNetInpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "BRPL": ("BRPL", ),
                        "image": ("IMAGE",),
                        "mask": ("MASK",),
                        "prompt": ("STRING", {"multiline": False, "default": ""}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "brushnet_inpaint"

    def brushnet_inpaint(self, BRPL, image: torch.Tensor, mask: torch.Tensor, prompt: str, seed: int, 
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

        generator = torch.Generator("cuda").manual_seed(seed)
        result = BRPL(
            prompt, 
            init_image, 
            mask_image, 
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
