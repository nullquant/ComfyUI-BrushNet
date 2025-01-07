import torch.nn.functional as F
import comfy

from .model_patch import add_model_patch_option, patch_model_function_wrapper



class RAUNet:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {    
                        "model": ("MODEL",),
                        "du_start": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "du_end": ("INT", {"default": 4, "min": 0, "max": 10000}),
                        "xa_start": ("INT", {"default": 4, "min": 0, "max": 10000}),
                        "xa_end": ("INT", {"default": 10, "min": 0, "max": 10000}),
                     },
        }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "model_update"

    def model_update(self, model, du_start, du_end, xa_start, xa_end):
        
        model = model.clone()

        add_raunet_patch(model, 
                         du_start,
                         du_end,
                         xa_start,
                         xa_end)
        
        return (model,)


# This is main patch function
def add_raunet_patch(model, du_start, du_end, xa_start, xa_end):
    
    def raunet_forward(model, x, timesteps, transformer_options, control):
        if 'brush_model_patch' not in transformer_options:
            print("RAUNet: 'brush_model_patch' not in transformer_options, skip")
            return

        mp = transformer_options['brush_model_patch']
        is_SDXL = mp['SDXL']

        if is_SDXL and type(model.input_blocks[6][0]) != comfy.ldm.modules.diffusionmodules.openaimodel.Downsample:
            print('RAUNet: model is SDXL, but input[6] != Downsample, skip')
            return

        if not is_SDXL and type(model.input_blocks[3][0]) != comfy.ldm.modules.diffusionmodules.openaimodel.Downsample:
            print('RAUNet: model is not SDXL, but input[3] != Downsample, skip')
            return
        
        if 'raunet' not in mp:
            print('RAUNet: "raunet" not in model_patch options, skip')
            return

        if is_SDXL:
            block = model.input_blocks[6][0]
        else:
            block = model.input_blocks[3][0]

        total_steps = mp['total_steps']
        step = mp['step']

        ro = mp['raunet']
        du_start = ro['du_start']
        du_end = ro['du_end']

        if step >= du_start and step < du_end:
            block.op.stride = (4, 4)
            block.op.padding = (2, 2)
            block.op.dilation = (2, 2)
        else:
            block.op.stride = (2, 2)
            block.op.padding = (1, 1)
            block.op.dilation = (1, 1)

    patch_model_function_wrapper(model, raunet_forward)
    model.set_model_input_block_patch(in_xattn_patch)
    model.set_model_output_block_patch(out_xattn_patch)

    to = add_model_patch_option(model)
    mp = to['brush_model_patch']
    if 'raunet' not in mp:
        mp['raunet'] = {}
    ro = mp['raunet']

    ro['du_start'] = du_start
    ro['du_end'] = du_end
    ro['xa_start'] = xa_start
    ro['xa_end'] = xa_end


def in_xattn_patch(h, transformer_options):
    # both SDXL and SD15 = (input,4)
    if transformer_options["block"] != ("input", 4):
        # wrong block
        return h
    if 'brush_model_patch' not in transformer_options:
        print("RAUNet (i-x-p): 'brush_model_patch' not in transformer_options")
        return h
    mp = transformer_options['brush_model_patch']
    if 'raunet' not in mp:
        print("RAUNet (i-x-p): 'raunet' not in model_patch options")
        return h

    step = mp['step']
    ro = mp['raunet']
    xa_start = ro['xa_start']
    xa_end = ro['xa_end']

    if step < xa_start or step >= xa_end:
        return h
    h = F.avg_pool2d(h, kernel_size=(2,2))
    return h


def out_xattn_patch(h, hsp, transformer_options):
    if 'brush_model_patch' not in transformer_options:
        print("RAUNet (o-x-p): 'brush_model_patch' not in transformer_options")
        return h, hsp
    mp = transformer_options['brush_model_patch']
    if 'raunet' not in mp:
        print("RAUNet (o-x-p): 'raunet' not in model_patch options")
        return h
    
    step = mp['step']
    is_SDXL = mp['SDXL']
    ro = mp['raunet']
    xa_start = ro['xa_start']
    xa_end = ro['xa_end']

    if is_SDXL:
        if transformer_options["block"] != ("output", 5):
            # wrong block
            return h, hsp
    else:
        if transformer_options["block"] != ("output", 8):
            # wrong block
            return h, hsp

    if step < xa_start or step >= xa_end:
        return h, hsp
    #error in hidiffusion codebase, size * 2 for particular sizes only
    #re_size = (int(h.shape[-2] * 2), int(h.shape[-1] * 2))
    re_size = (hsp.shape[-2], hsp.shape[-1])
    h = F.interpolate(h, size=re_size, mode='bicubic')

    return h, hsp


