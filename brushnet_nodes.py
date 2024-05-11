import torch
import torchvision.transforms as T
import math

import os
import folder_paths

import sys
from sys import platform
# Get the parent directory of 'comfy' and add it to the Python path
comfy_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(comfy_parent_dir)
import comfy

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .brushnet.brushnet import BrushNetModel
from .brushnet.brushnet_ca import BrushNetModel as PowerPaintModel

from .brushnet.powerpaint_utils import TokenizerWrapper, add_tokens

import types
from typing import Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="safetensors")

current_directory = os.path.dirname(os.path.abspath(__file__))
brushnet_config_file = os.path.join(current_directory, 'brushnet', 'brushnet.json')
brushnet_xl_config_file = os.path.join(current_directory, 'brushnet', 'brushnet_xl.json')
powerpaint_config_file = os.path.join(current_directory,'brushnet', 'powerpaint.json')

sd15_scaling_factor = 0.18215
sdxl_scaling_factor = 0.13025

ComfySDLayers = [comfy.ops.disable_weight_init.Conv2d, 
                 comfy.ldm.modules.attention.SpatialTransformer,
                 comfy.ldm.modules.diffusionmodules.openaimodel.Downsample,
                 comfy.ldm.modules.diffusionmodules.openaimodel.ResBlock
                 ]

class BrushNetLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "brushnet": (get_files_with_extension('inpaint'), ),
                        "dtype": (['float16', 'bfloat16', 'float32', 'float64'], ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("BRMODEL",)
    RETURN_NAMES = ("brushnet",)

    FUNCTION = "brushnet_loading"

    def brushnet_loading(self, brushnet, dtype):
        brushnet_file = os.path.join(folder_paths.models_dir, "inpaint", brushnet)

        is_SDXL = False
        is_PP = False
        sd = comfy.utils.load_torch_file(brushnet_file)
        brushnet_down_block, brushnet_mid_block, brushnet_up_block, keys = brushnet_blocks(sd)
        del sd
        if brushnet_down_block == 24 and brushnet_mid_block == 2 and brushnet_up_block == 30:
            is_SDXL = False
            if keys == 322:
                is_PP = False
                print('BrushNet model type: SD1.5')
            else:
                is_PP = True
                print('PowerPaint model type: SD1.5')
        elif brushnet_down_block == 18 and brushnet_mid_block == 2 and brushnet_up_block == 22:
            print('BrushNet model type: Loading SDXL')
            is_SDXL = True
            is_PP = False
        else:
            raise Exception("Unknown BrushNet model")

        with init_empty_weights():
            if is_SDXL:
                brushnet_config = BrushNetModel.load_config(brushnet_xl_config_file)
                brushnet_model = BrushNetModel.from_config(brushnet_config)
            elif is_PP:
                brushnet_config = PowerPaintModel.load_config(powerpaint_config_file)
                brushnet_model = PowerPaintModel.from_config(brushnet_config)
            else:
                brushnet_config = BrushNetModel.load_config(brushnet_config_file)
                brushnet_model = BrushNetModel.from_config(brushnet_config)

        if is_PP:
            print("PowerPaint model file:", brushnet_file)
        else:
            print("BrushNet model file:", brushnet_file)

        if dtype == 'float16':
            torch_dtype = torch.float16
        elif dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        elif dtype == 'float32':
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float64

        brushnet_model = load_checkpoint_and_dispatch(
            brushnet_model,
            brushnet_file,
            device_map="sequential",
            max_memory=None,
            offload_folder=None,
            offload_state_dict=False,
            dtype=torch_dtype,
            force_hooks=False,
        )

        if is_PP: 
            print("PowerPaint model is loaded")
        elif is_SDXL:
            print("BrushNet SDXL model is loaded")
        else:
            print("BrushNet SD1.5 model is loaded")

        return ({"brushnet": brushnet_model, "SDXL": is_SDXL, "PP": is_PP, "dtype": torch_dtype}, )


class PowerPaintCLIPLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "base": (get_files_with_extension('clip'), ),
                        "powerpaint": (get_files_with_extension('inpaint', ['bin']), ),
                     },
                }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)

    FUNCTION = "ppclip_loading"

    def ppclip_loading(self, base, powerpaint):
        base_CLIP_file = os.path.join(folder_paths.models_dir, "clip", base)
        pp_CLIP_file = os.path.join(folder_paths.models_dir, "inpaint", powerpaint)

        pp_clip = comfy.sd.load_clip(ckpt_paths=[base_CLIP_file])

        print('PowerPaint base CLIP file: ', base_CLIP_file)

        pp_tokenizer = TokenizerWrapper(pp_clip.tokenizer.clip_l.tokenizer)
        pp_text_encoder = pp_clip.patcher.model.clip_l.transformer

        add_tokens(
            tokenizer = pp_tokenizer,
            text_encoder = pp_text_encoder,
            placeholder_tokens = ["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens = ["a", "a", "a"],
            num_vectors_per_token = 10,
        )

        pp_text_encoder.load_state_dict(torch.load(pp_CLIP_file), strict=False)

        print('PowerPaint CLIP file: ', pp_CLIP_file)

        pp_clip.tokenizer.clip_l.tokenizer = pp_tokenizer
        pp_clip.patcher.model.clip_l.transformer = pp_text_encoder

        return (pp_clip,)
    

class PowerPaint:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "model": ("MODEL",),
                        "vae": ("VAE", ),
                        "image": ("IMAGE",),
                        "mask": ("MASK",),
                        "powerpaint": ("BRMODEL", ),
                        "clip": ("CLIP", ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "fitting" : ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                        "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'], ),
                        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                        "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     },
        }
    
    CATEGORY = "inpaint"
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("model","positive","negative","latent",)

    FUNCTION = "model_update"

    def model_update(self, model, vae, image, mask, powerpaint, clip, positive, negative, fitting, function, scale, start_at, end_at):

        is_SDXL, is_PP = check_compatibilty(model, powerpaint)
        if not is_PP:
            raise Exception("BrushNet model was loaded, please use BrushNet node")  

        # Make a copy of the model so that we're not patching it everywhere in the workflow.
        model = model.clone()

        # prepare image and mask
        # no batches for original image and mask
        masked_image, mask = prepare_image(image, mask)

        batch = masked_image.shape[0]
        #width = masked_image.shape[2]
        #height = masked_image.shape[1]

        if hasattr(model.model.model_config, 'latent_format') and hasattr(model.model.model_config.latent_format, 'scale_factor'):
            scaling_factor = model.model.model_config.latent_format.scale_factor
        else:
            scaling_factor = sd15_scaling_factor

        torch_dtype = powerpaint['dtype']

        # prepare conditioning latents
        conditioning_latents = get_image_latents(masked_image, mask, vae, scaling_factor)
        conditioning_latents[0] = conditioning_latents[0].to(dtype=torch_dtype).to(powerpaint['brushnet'].device)
        conditioning_latents[1] = conditioning_latents[1].to(dtype=torch_dtype).to(powerpaint['brushnet'].device)

        # prepare embeddings

        if function == "object removal":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"
            print('You should add to positive prompt: "empty scene blur"')
            #positive = positive + " empty scene blur"
        elif function == "context aware":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = ""
            negative_promptB = ""
            #positive = positive + " empty scene"
            print('You should add to positive prompt: "empty scene"')
        elif function == "shape guided":
            promptA = "P_shape"
            promptB = "P_ctxt"
            negative_promptA = "P_shape"
            negative_promptB = "P_ctxt"
        elif function == "image outpainting":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"
            #positive = positive + " empty scene"
            print('You should add to positive prompt: "empty scene"')
        else:
            promptA = "P_obj"
            promptB = "P_obj"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"

        tokens = clip.tokenize(promptA)
        prompt_embedsA = clip.encode_from_tokens(tokens, return_pooled=False)

        tokens = clip.tokenize(negative_promptA)
        negative_prompt_embedsA = clip.encode_from_tokens(tokens, return_pooled=False)

        tokens = clip.tokenize(promptB)
        prompt_embedsB = clip.encode_from_tokens(tokens, return_pooled=False)

        tokens = clip.tokenize(negative_promptB)
        negative_prompt_embedsB = clip.encode_from_tokens(tokens, return_pooled=False)

        prompt_embeds_pp = (prompt_embedsA * fitting + (1.0 - fitting) * prompt_embedsB).to(dtype=torch_dtype).to(powerpaint['brushnet'].device)
        negative_prompt_embeds_pp = (negative_prompt_embedsA * fitting + (1.0 - fitting) * negative_prompt_embedsB).to(dtype=torch_dtype).to(powerpaint['brushnet'].device)

        # apply patch to code

        if comfy.samplers.sample.__doc__ is None or 'BrushNet' not in comfy.samplers.sample.__doc__:
            comfy.samplers.original_sample = comfy.samplers.sample
            comfy.samplers.sample = modified_sample

        # apply patch to model

        brushnet_conditioning_scale = scale
        control_guidance_start = start_at
        control_guidance_end = end_at

        add_brushnet_patch(model, 
                           powerpaint['brushnet'],
                           torch_dtype,
                           conditioning_latents, 
                           (brushnet_conditioning_scale, control_guidance_start, control_guidance_end), 
                           negative_prompt_embeds_pp, prompt_embeds_pp, 
                           None, None, None)

        latent = torch.zeros([batch, 4, conditioning_latents[0].shape[2], conditioning_latents[0].shape[3]], device=powerpaint['brushnet'].device)

        return (model, positive, negative, {"samples":latent},)

    
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
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                        "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     },
        }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT",)
    RETURN_NAMES = ("model","positive","negative","latent",)

    FUNCTION = "model_update"

    def model_update(self, model, vae, image, mask, brushnet, positive, negative, scale, start_at, end_at):

        is_SDXL, is_PP = check_compatibilty(model, brushnet)

        if is_PP:
            raise Exception("PowerPaint model was loaded, please use PowerPaint node")  

        # Make a copy of the model so that we're not patching it everywhere in the workflow.
        model = model.clone()

        # prepare image and mask
        # no batches for original image and mask
        masked_image, mask = prepare_image(image, mask)

        batch = masked_image.shape[0]
        width = masked_image.shape[2]
        height = masked_image.shape[1]

        if hasattr(model.model.model_config, 'latent_format') and hasattr(model.model.model_config.latent_format, 'scale_factor'):
            scaling_factor = model.model.model_config.latent_format.scale_factor
        elif is_SDXL:
            scaling_factor = sdxl_scaling_factor
        else:
            scaling_factor = sd15_scaling_factor

        torch_dtype = brushnet['dtype']

        # prepare conditioning latents
        conditioning_latents = get_image_latents(masked_image, mask, vae, scaling_factor)
        conditioning_latents[0] = conditioning_latents[0].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        conditioning_latents[1] = conditioning_latents[1].to(dtype=torch_dtype).to(brushnet['brushnet'].device)

        # prepare embeddings

        prompt_embeds = positive[0][0].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        negative_prompt_embeds = negative[0][0].to(dtype=torch_dtype).to(brushnet['brushnet'].device)

        if len(positive[0]) > 1 and 'pooled_output' in positive[0][1] and positive[0][1]['pooled_output'] is not None:
            pooled_prompt_embeds = positive[0][1]['pooled_output'].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        else:
            print('BrushNet: positive conditioning has not pooled_output')
            if is_SDXL:
                print('BrushNet will not produce correct results')
            pooled_prompt_embeds = torch.empty([2, 1280], device=brushnet['brushnet'].device).to(dtype=torch_dtype)

        if len(negative[0]) > 1 and 'pooled_output' in negative[0][1] and negative[0][1]['pooled_output'] is not None:
            negative_pooled_prompt_embeds = negative[0][1]['pooled_output'].to(dtype=torch_dtype).to(brushnet['brushnet'].device)
        else:
            print('BrushNet: negative conditioning has not pooled_output')
            if is_SDXL:
                print('BrushNet will not produce correct results')
            negative_pooled_prompt_embeds = torch.empty([1, pooled_prompt_embeds.shape[1]], device=brushnet['brushnet'].device).to(dtype=torch_dtype)

        time_ids = torch.FloatTensor([[height, width, 0., 0., height, width]]).to(dtype=torch_dtype).to(brushnet['brushnet'].device)

        if not is_SDXL:
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            time_ids = None

        # apply patch to code

        if comfy.samplers.sample.__doc__ is None or 'BrushNet' not in comfy.samplers.sample.__doc__:
            comfy.samplers.original_sample = comfy.samplers.sample
            comfy.samplers.sample = modified_sample

        # apply patch to model

        brushnet_conditioning_scale = scale
        control_guidance_start = start_at
        control_guidance_end = end_at

        add_brushnet_patch(model, 
                           brushnet['brushnet'],
                           torch_dtype,
                           conditioning_latents, 
                           (brushnet_conditioning_scale, control_guidance_start, control_guidance_end), 
                           prompt_embeds, negative_prompt_embeds,
                           pooled_prompt_embeds, negative_pooled_prompt_embeds, time_ids)

        latent = torch.zeros([batch, 4, conditioning_latents[0].shape[2], conditioning_latents[0].shape[3]], device=brushnet['brushnet'].device)

        return (model, positive, negative, {"samples":latent},)


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

        inpaint = torch.nn.functional.interpolate(
            inpaint.permute(0, 3, 1, 2), 
            size=(
                original.shape[0], 
                original.shape[1],
            )
        ).to(original.device).to(original.dtype)

        ret = []
        for result in inpaint.permute(0, 2, 3, 1):
            ret.append(original * (1.0 - blurred_mask[0][0][:,:,None]) + result.to(original.device) * blurred_mask[0][0][:,:,None])

        return (torch.stack(ret), blurred_mask[0],)


#### Utility function

def get_files_with_extension(folder_name, extension=['safetensors']):
    inpaint_path = os.path.join(folder_paths.models_dir, folder_name)
    abs_list = []
    for x in os.walk(inpaint_path):
        for name in x[2]:
            for ext in extension:
                if ext in name:
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
    return (brushnet_down_block, brushnet_mid_block, brushnet_up_block, len(sd))


# Check models compatibility
def check_compatibilty(model, brushnet):
    is_SDXL = False
    is_PP = False
    if isinstance(model.model.model_config, comfy.supported_models.SD15):
        print('Base model type: SD1.5')
        is_SDXL = False
        if brushnet["SDXL"]:
            raise Exception("Base model is SD15, but BrushNet is SDXL type")  
        if brushnet["PP"]:
            is_PP = True
    elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
        print('Base model type: SDXL')
        is_SDXL = True
        if not brushnet["SDXL"]:
            raise Exception("Base model is SDXL, but BrushNet is SD15 type")    
    else:
        print('Base model type: ', type(model.model.model_config))
        raise Exception("Unsupported model type: " + str(type(model.model.model_config)))

    return (is_SDXL, is_PP)

# Prepare image and mask
def prepare_image(image, mask):
    if len(image.shape) < 4:
        # image tensor shape should be [B, H, W, C], but batch somehow is missing
        image = image[None,:,:,:]
    
    if len(mask.shape) > 3:
        # mask tensor shape should be [B, H, W] but we get [B, H, W, C], image may be?
        # take first mask, red channel
        mask = (mask[:,:,:,0])[:,:,:]
    elif len(mask.shape) < 3:
        # mask tensor shape should be [B, H, W] but batch somehow is missing
        mask = mask[None,:,:]

    if image.shape[0] > mask.shape[0]:
        print("BrushNet gets batch of images (%d) but only %d masks" % (image.shape[0], mask.shape[0]))
        if mask.shape[0] == 1: 
            print("BrushNet will copy the mask to fill batch")
            mask = torch.cat([mask] * image.shape[0], dim=0)
        else:
            print("BrushNet will add empty masks to fill batch")
            empty_mask = torch.zeros([image.shape[0] - mask.shape[0], mask.shape[1], mask.shape[2]])
            mask = torch.cat([mask, empty_mask], dim=0)
    elif image.shape[0] < mask.shape[0]:
        print("BrushNet gets batch of images (%d) but too many (%d) masks" % (image.shape[0], mask.shape[0]))
        mask = mask[:image.shape[0],:,:]

    print("BrushNet image.shape =", image.shape, "mask.shape =", mask.shape)

    if mask.shape[2] != image.shape[2] or mask.shape[1] != image.shape[1]:
        raise Exception("Image and mask should be the same size")
    
    # As a suggestion of inferno46n2 (https://github.com/nullquant/ComfyUI-BrushNet/issues/64)
    mask = mask.round()

    masked_image = image * (1.0 - mask[:,:,:,None])

    return (masked_image, mask)


# Prepare conditioning_latents
@torch.inference_mode()
def get_image_latents(masked_image, mask, vae, scaling_factor):
    processed_image = masked_image.to(vae.device)
    image_latents = vae.encode(processed_image[:,:,:,:3]) * scaling_factor
    processed_mask = 1. - mask[:,None,:,:]
    interpolated_mask = torch.nn.functional.interpolate(
                processed_mask, 
                size=(
                    image_latents.shape[-2], 
                    image_latents.shape[-1]
                )
            )
    interpolated_mask = interpolated_mask.to(image_latents.device)

    conditioning_latents = [image_latents, interpolated_mask]

    print('BrushNet CL: image_latents shape =', image_latents.shape, 'interpolated_mask shape =', interpolated_mask.shape)

    return conditioning_latents


# Model needs current step number and cfg at inference step. It is possible to write a custom KSampler but I'd like to use ComfyUI's one.
# The first versions had modified_common_ksampler, but it broke custom KSampler nodes
def modified_sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, 
           latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    '''
    Modified by BrushNet nodes
    '''
    cfg_guider = comfy.samplers.CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)

    ### Modified part ######################################################################
    #
    to = add_model_patch_option(model)
    to['model_patch']['all_sigmas'] = sigmas

    if math.isclose(cfg, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        to['model_patch']['free_guidance'] = False
    else:
        to['model_patch']['free_guidance'] = True
    #
    #######################################################################################
       
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


# Main function where magic happens
@torch.inference_mode()
def brushnet_inference(x, timesteps, transformer_options):
    if 'model_patch' not in transformer_options:
        print('BrushNet inference: there is no model_patch in transformer_options')
        return ([], 0, [])
    mp = transformer_options['model_patch']
    if 'brushnet_model' not in mp:
        print('BrushNet inference: there is no brushnet_model in transformer_options')
        return ([], 0, [])
    brushnet = mp['brushnet_model']
    if isinstance(brushnet, BrushNetModel) or isinstance(brushnet, PowerPaintModel):
        torch_dtype = mp['brushnet_dtype']
        cl_list = mp['brushnet_latents']
        brushnet_conditioning_scale, control_guidance_start, control_guidance_end = mp['brushnet_controls']
        pe = mp['brushnet_prompt']
        npe = mp['brushnet_negative_prompt']
        ppe, nppe, time_ids = mp['brushnet_add_embeds']

        do_classifier_free_guidance = mp['free_guidance']

        x = x.detach().clone()
        x = x.to(torch_dtype).to(brushnet.device)

        timesteps = timesteps.detach().clone()
        timesteps = timesteps.to(torch_dtype).to(brushnet.device)

        all_sigmas = mp['all_sigmas']
        sigma = transformer_options['sigmas'][0].item()
        total_steps = all_sigmas.shape[0]
        step = torch.argmin((all_sigmas - sigma).abs()).item()

        added_cond_kwargs = {}

        if do_classifier_free_guidance and step == 0:
            print('BrushNet inference: do_classifier_free_guidance is True')

        sub_idx = None
        if 'ad_params' in transformer_options and 'sub_idxs' in transformer_options['ad_params']:
            sub_idx = transformer_options['ad_params']['sub_idxs']

        # we have batch input images
        batch = cl_list[0].shape[0]
        # we have incoming latents
        latents_incoming = x.shape[0]
        # and we already got some
        latents_got = mp['brushnet_latent_id']
        if step == 0 or batch > 1:
            print('BrushNet inference, step = %d: image batch = %d, got %d latents, starting from %d' \
                    % (step, batch, latents_incoming, latents_got))

        image_latents = []
        masks = []
        prompt_embeds = []
        negative_prompt_embeds = []
        pooled_prompt_embeds = []
        negative_pooled_prompt_embeds = []
        if sub_idx:
            if step == 0:
                print('BrushNet inference: AnimateDiff indexes detected and applied')

            batch = len(sub_idx)

            if do_classifier_free_guidance:
                for i in sub_idx:
                    image_latents.append(cl_list[0][i][None,:,:,:])
                    masks.append(cl_list[1][i][None,:,:,:])
                    prompt_embeds.append(pe)
                    negative_prompt_embeds.append(npe)
                    pooled_prompt_embeds.append(ppe)
                    negative_pooled_prompt_embeds.append(nppe)
                for i in sub_idx:
                    image_latents.append(cl_list[0][i][None,:,:,:])
                    masks.append(cl_list[1][i][None,:,:,:])
            else:
                for i in sub_idx:
                    image_latents.append(cl_list[0][i][None,:,:,:])
                    masks.append(cl_list[1][i][None,:,:,:])
                    prompt_embeds.append(pe)
                    pooled_prompt_embeds.append(ppe)
        else:
            # do_classifier_free_guidance = 2 passes, 1st pass is cond, 2nd is uncond
            continue_batch = True
            for i in range(latents_incoming):
                number = latents_got + i
                if number < batch:
                    # 1st pass, cond
                    image_latents.append(cl_list[0][number][None,:,:,:])
                    masks.append(cl_list[1][number][None,:,:,:])
                    prompt_embeds.append(pe)
                    pooled_prompt_embeds.append(ppe)
                elif do_classifier_free_guidance and number < batch * 2:
                    # 2nd pass, uncond
                    image_latents.append(cl_list[0][number-batch][None,:,:,:])
                    masks.append(cl_list[1][number-batch][None,:,:,:])
                    negative_prompt_embeds.append(npe)
                    negative_pooled_prompt_embeds.append(nppe)
                else:
                    # latent batch
                    image_latents.append(cl_list[0][0][None,:,:,:])
                    masks.append(cl_list[1][0][None,:,:,:])
                    prompt_embeds.append(pe)
                    pooled_prompt_embeds.append(ppe)
                    latents_got = -i
                    continue_batch = False

            if continue_batch:
                # we don't have full batch yet
                if do_classifier_free_guidance:
                    if number < batch * 2 - 1:
                        mp['brushnet_latent_id'] = number + 1
                    else:
                        mp['brushnet_latent_id'] = 0
                else:
                    if number < batch - 1:
                        mp['brushnet_latent_id'] = number + 1
                    else:
                        mp['brushnet_latent_id'] = 0
            else:
                mp['brushnet_latent_id'] = 0

        cl = []
        for il, m in zip(image_latents, masks):
            cl.append(torch.concat([il, m], dim=1))
        cl2apply = torch.concat(cl, dim=0)

        conditioning_latents = cl2apply.to(torch_dtype).to(brushnet.device)

        prompt_embeds.extend(negative_prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds, dim=0).to(torch_dtype).to(brushnet.device)

        if ppe is not None:
            added_cond_kwargs = {}
            added_cond_kwargs['time_ids'] = torch.concat([time_ids] * latents_incoming, dim = 0).to(torch_dtype).to(brushnet.device)

            pooled_prompt_embeds.extend(negative_pooled_prompt_embeds)
            pooled_prompt_embeds = torch.concat(pooled_prompt_embeds, dim=0).to(torch_dtype).to(brushnet.device)
            added_cond_kwargs['text_embeds'] = pooled_prompt_embeds
        else:
            added_cond_kwargs = None

        if x.shape[2] != conditioning_latents.shape[2] or x.shape[3] != conditioning_latents.shape[3]:
            if step == 0:
                print('BrushNet inference: image', conditioning_latents.shape, 'and latent', x.shape, 'have different size, resizing image')
            conditioning_latents = torch.nn.functional.interpolate(
                conditioning_latents, size=(
                    x.shape[2], 
                    x.shape[3],
                ), mode='bicubic',
            ).to(torch_dtype).to(brushnet.device)

        if step == 0:
            print('BrushNet inference: sample', x.shape, ', CL', conditioning_latents.shape)

        if step < control_guidance_start or step > control_guidance_end:
            cond_scale = 0.0
        else:
            cond_scale = brushnet_conditioning_scale

        return brushnet(x,
                        encoder_hidden_states=prompt_embeds,
                        brushnet_cond=conditioning_latents,
                        timestep = timesteps,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )
    else:
        print('BrushNet model is not a BrushNetModel class')
        return ([], 0, [])


# Check and add 'model_patch' to model.model_options['transformer_options']
def add_model_patch_option(model):
    if 'transformer_options' not in model.model_options:
        model.model_options['transformer_options'] = {}
    to = model.model_options['transformer_options']
    if "model_patch" not in to:
        to["model_patch"] = {}
    return to


# This is main patch function
def add_brushnet_patch(model, brushnet, torch_dtype, conditioning_latents, 
                       controls, 
                       prompt_embeds, negative_prompt_embeds,
                       pooled_prompt_embeds, negative_pooled_prompt_embeds, time_ids):
    to = add_model_patch_option(model)
    to['model_patch']['brushnet_model'] = brushnet
    to['model_patch']['brushnet_dtype'] = torch_dtype
    to['model_patch']['brushnet_latents'] = conditioning_latents
    to['model_patch']['brushnet_controls'] = controls
    to['model_patch']['brushnet_prompt'] = prompt_embeds
    to['model_patch']['brushnet_negative_prompt'] = negative_prompt_embeds
    to['model_patch']['brushnet_add_embeds'] = (pooled_prompt_embeds, negative_pooled_prompt_embeds, time_ids)
    to['model_patch']['brushnet_latent_id'] = 0

    is_SDXL = isinstance(model.model.model_config, comfy.supported_models.SDXL)

    if is_SDXL:
        input_blocks = [[0,0],[1,0],[2,0],[3,0],[4,1],[5,1],[6,0],[7,1],[8,1]]
        output_blocks = [[0,1],[1,1],[2,1],[2,2],[3,1],[4,1],[5,1],[5,2],[6,0],[7,0],[8,0]]
    else:
        input_blocks = [[0,0],[1,1],[2,1],[3,0],[4,1],[5,1],[6,0],[7,1],[8,1],[9,0],[10,0],[11,0]]
        output_blocks = [[0,0],[1,0],[2,0],[2,1],[3,1],[4,1],[5,1],[5,2],[6,1],[7,1],[8,1],[8,2],[9,1],[10,1],[11,1]]
    
    # patch model `forward` so we can call BrushNet inference and add additional samples to layers
    if not hasattr(model.model.diffusion_model, 'original_forward'):
        model.model.diffusion_model.original_forward = model.model.diffusion_model.forward
    def forward_patched_by_brushnet(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        # check if this is brushnet patched model
        if 'model_patch' not in transformer_options or 'brushnet_model' not in transformer_options['model_patch']:
            input_samples = []
            mid_sample = 0
            output_samples = []
        else:    
            # brushnet inference
            input_samples, mid_sample, output_samples = brushnet_inference(x, timesteps, transformer_options)
        # give additional samples to blocks
        for i, block in enumerate(self.input_blocks):
            for j, layer in enumerate(block):
                if [i,j] in input_blocks:
                    layer.brushnet_sample = input_samples.pop(0) if input_samples else 0
        self.middle_block[-1].brushnet_sample = mid_sample
        for i, block in enumerate(self.output_blocks):
            for j, layer in enumerate(block):
                if [i,j] in output_blocks:
                    layer.brushnet_sample = output_samples.pop(0) if output_samples else 0
        return self.original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
    model.model.diffusion_model.forward = types.MethodType(forward_patched_by_brushnet, model.model.diffusion_model)

    # patch layers `forward` so we can apply brushnet
    def forward_patched_by_brushnet(self, x, *args, **kwargs):
        h = self.original_forward(x, *args, **kwargs)
        if hasattr(self, 'brushnet_sample') and type(self) in ComfySDLayers:
            if torch.is_tensor(self.brushnet_sample):
                #print(type(self), h.shape, self.brushnet_sample.shape)
                h += self.brushnet_sample.to(h.dtype).to(h.device)
            else:
                h += self.brushnet_sample
        return h

    for i, block in enumerate(model.model.diffusion_model.input_blocks):
        for j, layer in enumerate(block):
            if not hasattr(layer, 'original_forward'):
                layer.original_forward = layer.forward
            layer.forward = types.MethodType(forward_patched_by_brushnet, layer)

    for j, layer in enumerate(model.model.diffusion_model.middle_block):
        if not hasattr(layer, 'original_forward'):
            layer.original_forward = layer.forward
        layer.forward = types.MethodType(forward_patched_by_brushnet, layer)

    for i, block in enumerate(model.model.diffusion_model.output_blocks):
        for j, layer in enumerate(block):
            if not hasattr(layer, 'original_forward'):
                layer.original_forward = layer.forward
            layer.forward = types.MethodType(forward_patched_by_brushnet, layer)
