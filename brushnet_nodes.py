import torch
import torchvision.transforms as T

import os
import folder_paths

import sys
# Get the parent directory of 'comfy' and add it to the Python path
comfy_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(comfy_parent_dir)
import comfy
import nodes
import latent_preview

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

#import numpy as np
#import cv2
#from PIL import Image
#import yaml
#import importlib
#from contextlib import nullcontext
#from diffusers.loaders.single_file_utils import create_diffusers_vae_model_from_ldm
#from diffusers.image_processor import VaeImageProcessor

from .brushnet.brushnet import BrushNetModel

from typing import Tuple

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")

current_directory = os.path.dirname(os.path.abspath(__file__))
#original_config_file = os.path.join(current_directory, 'brushnet', 'v1-inference.yaml')
brushnet_config_file = os.path.join(current_directory, 'brushnet', 'brushnet.json')
brushnet_xl_config_file = os.path.join(current_directory, 'brushnet', 'brushnet_xl.json')

torch_dtype = torch.float16
sd15_scaling_factor = 0.18215
sdxl_scaling_factor = 0.13025

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

        is_SDXL = False
        sd = comfy.utils.load_torch_file(brushnet_file)
        brushnet_down_block, brushnet_mid_block, brushnet_up_block = brushnet_blocks(sd)
        del sd
        if brushnet_down_block == 24 and brushnet_mid_block == 2 and brushnet_up_block == 30:
            print('Loading SD1.5 BrushNet model')
            is_SDXL = False
        elif brushnet_down_block == 18 and brushnet_mid_block == 2 and brushnet_up_block == 22:
            print('Loading SDXL BrushNet model')
            is_SDXL = True
        else:
            raise Exception("Unknown BrushNet model")

        with init_empty_weights():
            if is_SDXL:
                brushnet_config = BrushNetModel.load_config(brushnet_xl_config_file)
            else:
                brushnet_config = BrushNetModel.load_config(brushnet_config_file)
            brushnet_model = BrushNetModel.from_config(brushnet_config)

        print("BrushNet file", brushnet_file)

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

        print("BrushNet model is loaded, SDXL:", is_SDXL)

        return ({"brushnet": brushnet_model, "SDXL": is_SDXL},)

    
def inpaint_safetensors():
    inpaint_path = os.path.join(folder_paths.models_dir, 'inpaint')
    brushnet_path = os.path.join(inpaint_path, 'brushnet')
    abs_list = []
    for x in os.walk(inpaint_path):
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

def brushnet_blocks(sd):
    brushnet_down_block = 0
    brushnet_mid_block = 0
    brushnet_up_block = 0
    for key in sd:
        if 'brushnet_down_block' in key:
            brushnet_down_block += 1
        if 'brushnet_mid_block' in key:
            brushnet_mid_block += 1        
        if 'brushnet_up_block' in key:
            brushnet_up_block += 1
    return (brushnet_down_block, brushnet_mid_block, brushnet_up_block, )


class BrushNet:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "model": ("MODEL",),
                        "vae": ("VAE", ),
                        "image": ("IMAGE",),
                        "mask": ("MASK",),
                        "brushnet": ("BRMODEL", ),
                        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                        "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                        "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("MODEL","LATENT",)
    RETURN_NAMES = ("model","latent",)

    FUNCTION = "model_update"

    def model_update(self, model, vae, image, mask, brushnet, scale, start_at, end_at):

        is_SDXL = False
        if isinstance(model.model.model_config, comfy.supported_models.SD15):
            print('Base model type: SD1.5')
            is_SDXL = False
            if brushnet["SDXL"]:
                raise Exception("Base model is SD15, but BrushNet is SDXL type")    
        elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
            print('Base model type: SDXL')
            is_SDXL = True
            if not brushnet["SDXL"]:
                raise Exception("Base model is SDXL, but BrushNet is SD15 type")    
            raise Exception("SDXL is not implemented yet")
        else:
            print('Base model type: ', type(model.model.model_config))
            raise Exception("Unsupported model type: " + str(type(model.model.model_config)))
        
        # prepare image and mask
        # no batches for original image and mask

        if image.shape[0] > 1:
            image = image[0][None,:,:,:]   
        if mask.shape[0] > 1:
            mask = mask[0][None,:,:]  

        width = image.shape[2]
        height = image.shape[1]

        print("BrushNet image, width", width, "height", height)

        if mask.shape[2] != width or mask.shape[1] != height:
            raise Exception("Image and mask should be the same size")

        masked_image = image * (1.0 - mask[:,:,:,None])

        if hasattr(model.model.model_config, 'latent_format') and hasattr(model.model.model_config.latent_format, 'scale_factor'):
            scaling_factor = model.model.model_config.latent_format.scale_factor
        elif is_SDXL:
            scaling_factor = sdxl_scaling_factor
        else:
            scaling_factor = sd15_scaling_factor

        processed_image = torch.cat([masked_image] * 2).to(vae.device)
        image_latents = vae.encode(processed_image[:,:,:,:3]) * scaling_factor

        processed_mask = torch.cat([1. - mask[None,:,:,:]] * 2)
        interpolated_mask = torch.nn.functional.interpolate(
                    processed_mask, 
                    size=(
                        image_latents.shape[-2], 
                        image_latents.shape[-1]
                    )
                )
        interpolated_mask = interpolated_mask.to(image_latents.device)

        print('BrushNet: image latents shape =', image_latents.shape, 'interpolated mask shape =', interpolated_mask.shape)

        conditioning_latents = torch.concat([image_latents, interpolated_mask], 1).to(dtype=torch_dtype).to(brushnet['brushnet'].device)

        # apply patches to code

        if 'BrushNet' not in nodes.common_ksampler.__doc__:
            nodes.original_common_ksampler = nodes.common_ksampler
            nodes.common_ksampler = modified_common_ksampler

        if 'BrushNet' not in comfy.ldm.modules.diffusionmodules.openaimodel.forward_timestep_embed.__doc__:
            comfy.ldm.modules.diffusionmodules.openaimodel.original_forward_timestep_embed =  \
            comfy.ldm.modules.diffusionmodules.openaimodel.forward_timestep_embed
            comfy.ldm.modules.diffusionmodules.openaimodel.forward_timestep_embed = modified_forward_timestep_embed

        # apply patch to model

        brushnet_conditioning_scale = scale
        control_guidance_start = start_at
        control_guidance_end = end_at

        add_brushnet_patch(model, brushnet["brushnet"], conditioning_latents, 
                           [brushnet_conditioning_scale, control_guidance_start, control_guidance_end])
        
        if is_SDXL:
            add_model_patch(model, brushnet_inference, ('input', 0), (0, 'before'))
            input_blocks = [[0,0],[1,0],[2,0],[3,0],[4,1],[5,1],[6,0],[7,1],[8,1]]
            for i, j in input_blocks:
                add_model_patch(model, apply_brushnet, ('input', i), (j, 'after'))
            add_model_patch(model, apply_brushnet, ('middle', 0), (2, 'after'))
            output_blocks = [[0,1],[1,1],[2,1],[2,2],[3,1],[4,1],[5,1],[5,2],[6,0],[7,0],[8,0]]
            for i, j in output_blocks:
                add_model_patch(model, apply_brushnet, ('output', i), (j, 'after'))
        else:
            add_model_patch(model, brushnet_inference, ('input', 0), (0, 'before'))
            input_blocks = [[0,0],[1,1],[2,1],[3,0],[4,1],[5,1],[6,0],[7,1],[8,1],[9,0],[10,0],[11,0]]
            for i, j in input_blocks:
                add_model_patch(model, apply_brushnet, ('input', i), (j, 'after'))
            add_model_patch(model, apply_brushnet, ('middle', 0), (2, 'after'))
            output_blocks = [[0,0],[1,0],[2,0],[2,1],[3,1],[4,1],[5,1],[5,2],[6,1],[7,1],[8,1],[8,2],[9,1],[10,1],[11,1]]
            for i, j in output_blocks:
                add_model_patch(model, apply_brushnet, ('output', i), (j, 'after'))

        latent = torch.zeros([1, 4, conditioning_latents.shape[2], conditioning_latents.shape[3]], device=brushnet['brushnet'].device)

        return (model, {"samples":latent},)


class BlendInpaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "inpaint": ("IMAGE",),
                        "original": ("IMAGE",),
                        "mask": ("MASK",),
                        "kernel": ("INT", {"default": 10, "min": 1, "max": 1000}),
                        "sigma": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 1000}),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("image","MASK",)

    FUNCTION = "blend_inpaint"

    def blend_inpaint(self, inpaint: torch.Tensor, original: torch.Tensor, mask: torch.Tensor, kernel: int, sigma:int) -> Tuple[torch.Tensor]:

        # no batches over mask and original image
        if len(mask.shape) > 2:
            mask = mask[0]
        if len(original.shape) > 3:
            original = original[0]

        if kernel % 2 == 0:
            kernel += 1
        transform = T.GaussianBlur(kernel_size=(kernel, kernel), sigma=(sigma, sigma))
        blurred_mask = transform(mask[None,None,:,:]).to(original.device).to(original.dtype)

        ret = []
        for result in inpaint:
            ret.append(original * (1.0 - blurred_mask[0][0][:,:,None]) + result.to(original.device) * blurred_mask[0][0][:,:,None])

        return (torch.stack(ret), blurred_mask[0],)

'''
import pip

class TestNode:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "image": ("IMAGE",),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "test_node"

    def test_node(self):
        print('Installed packages')
        print(pip.utils.get_installed_distributions())
'''



# Unfortunately, ModelPatcher does not have necessary hooks to patch, so we have to patch the code instead
def modified_forward_timestep_embed(block, x, emb, context=None, transformer_options={}, 
                           output_shape=None, time_context=None, 
                           num_video_frames=None, image_only_indicator=None):
    '''
    Modified by BrushNet nodes
    '''

    if 'model_patch' not in transformer_options or transformer_options['block'] not in transformer_options['model_patch']:
        return comfy.ldm.modules.diffusionmodules.openaimodel.original_forward_timestep_embed(block, x, emb, context, 
                                                                                              transformer_options,
                                                                                              output_shape, time_context, 
                                                                                              num_video_frames, image_only_indicator)
        
    block_patch = transformer_options['model_patch'][transformer_options['block']]

    for i, layer in enumerate(block):

        if (i, 'before') in block_patch:
            x = block_patch[(i, 'before')](x, emb, context, (transformer_options['block'], i, 'before'), transformer_options)

        if isinstance(layer, comfy.ldm.modules.diffusionmodules.openaimodel.VideoResBlock):
            x = layer(x, emb, num_video_frames, image_only_indicator)
        elif isinstance(layer, comfy.ldm.modules.diffusionmodules.openaimodel.TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, comfy.ldm.modules.attention.SpatialVideoTransformer):
            x = layer(x, context, time_context, num_video_frames, image_only_indicator, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, comfy.ldm.modules.attention.SpatialTransformer):
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, comfy.ldm.modules.diffusionmodules.openaimodel.Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)

        if (i, 'after') in block_patch:
            x = block_patch[(i, 'after')](x, emb, context, (transformer_options['block'], i, 'after'), transformer_options)
        
    return x

# Model needs current step number at inference stepю. It is possible to write a custom KSampler but we'd like to use ComfyUI's one.
def modified_common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, 
                             disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    '''
    Modified by BrushNet nodes
    '''
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    #######################################################################################
    #
    latent_preview_callback = latent_preview.prepare_callback(model, steps)

    to = add_model_patch_option(model)
    to['model_patch']['step'] = 0
    to['model_patch']['total_steps'] = steps

    def callback(step, x0, x, total_steps):
        to['model_patch']['step'] = step + 1
        latent_preview_callback(steps, x0, x, total_steps)
    #
    #######################################################################################
    
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, 
                                  disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

def add_model_patch_option(model):
    if 'transformer_options' not in model.model_options:
        model.model_options['transformer_options'] = {}
    to = model.model_options['transformer_options']
    if "model_patch" not in to:
        to["model_patch"] = {}
    return to

def add_model_patch(model, patch, block, key, replace=True):
    to = add_model_patch_option(model)
    if block not in to["model_patch"]:
        to["model_patch"][block] = {}
    if key not in to["model_patch"][block]:
        to["model_patch"][block][key] = []

    if replace:
        to["model_patch"][block][key] = patch
    else:
        to["model_patch"][block][key].append(patch)

def add_brushnet_patch(model, brushnet, conditioning_latents, controls):
    to = add_model_patch_option(model)
    to['model_patch']['brushnet_model'] = brushnet
    to['model_patch']['brushnet_latents'] = conditioning_latents
    to['model_patch']['brushnet_controls'] = controls
    to['model_patch']['input_samples'] = []
    to['model_patch']['middle_sample'] = 0
    to['model_patch']['output_samples'] = []

def brushnet_inference(x, emb, context, loc, transformer_options):
    # x : sample
    # emb : time embedding
    # context[0] == neg_cond[0][0][0]
    # context[1] == pos_cond[0][0][0]
    # loc = (('input'|'middle'|'output', i), j, 'before'|'after')

    if 'model_patch' not in transformer_options:
        raise Exception('BrushNet inference: there is no model_patch in transformer_options')
    
    mp = transformer_options['model_patch']
    brushnet = mp['brushnet_model']
    if isinstance(brushnet, BrushNetModel):
        conditioning_latents = mp['brushnet_latents']
        step = mp['step']
        total_steps = mp['total_steps']
        brushnet_conditioning_scale, control_guidance_start, control_guidance_end = mp['brushnet_controls']
        brushnet_keep = []
        for i in range(total_steps):
            keeps = [
                1.0 - float(i / total_steps < s or (i + 1) / total_steps > e)
                for s, e in zip([control_guidance_start], [control_guidance_end])
            ]
            brushnet_keep.append(keeps[0])
        cond_scale = brushnet_conditioning_scale * brushnet_keep[step]

        down_samples, mid_sample, up_samples = brushnet(x,
                                                        encoder_hidden_states=context[1],
                                                        brushnet_cond=conditioning_latents,
                                                        time_emb=emb,
                                                        conditioning_scale=cond_scale,
                                                        guess_mode=False,
                                                        return_dict=False,)
        transformer_options['model_patch']['input_samples'] = down_samples
        transformer_options['model_patch']['middle_sample'] = mid_sample
        transformer_options['model_patch']['output_samples'] = up_samples
    else:
        raise Exception('BrushNet model is not a BrushNetModel class')
        
    return x

def apply_brushnet(x, emb, context, loc, transformer_options):
    if loc[0][0] == 'input':
        if len(transformer_options['model_patch']['input_samples']) > 0:
            return x + transformer_options['model_patch']['input_samples'].pop(0)
        else:
            raise Exception('BrushNet: something is not right, input samples are empty, ' + str(loc))
        
    elif loc[0][0] == 'middle':
        return x + transformer_options['model_patch']['middle_sample']
    else:
        if len(transformer_options['model_patch']['output_samples']) > 0:
            return x + transformer_options['model_patch']['output_samples'].pop(0)
        else:
            raise Exception('BrushNet: something is not right, output samples are empty, ' + str(loc))

#def image_from_tensor(t: torch.Tensor) -> np.ndarray:
#    image_np = t.numpy() 
#    # Convert the numpy array back to the original range (0-255) and data type (uint8)
#    image_np = (image_np * 255).astype(np.uint8)
#    return image_np
#
#def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
#    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
#    array = array.astype(np.float32) / 255.0
#    return torch.from_numpy(array)[None,]
