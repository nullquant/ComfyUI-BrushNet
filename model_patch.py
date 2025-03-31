import torch
import comfy
import types


# Check and add 'brush_model_patch' to model.model_options['transformer_options']
def add_model_patch_option(model):
    if 'transformer_options' not in model.model_options:
        model.model_options['transformer_options'] = {}
    to = model.model_options['transformer_options']
    if "brush_model_patch" not in to:
        to["brush_model_patch"] = {}
    return to


# Patch model with model_function_wrapper
def patch_model_function_wrapper(model, forward_patch, remove=False):

    patch_wrapper = {}
    comfy.patcher_extension.add_wrapper(comfy.patcher_extension.WrappersMP.APPLY_MODEL,
                                        apply_model_function_wrapper,
                                        patch_wrapper,
                                        is_model_options=False)
    to = add_model_patch_option(model)
    brush_model_patch = to['brush_model_patch']

    if isinstance(model.model.model_config, comfy.supported_models.SD15):
        brush_model_patch['SDXL'] = False
    elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
        brush_model_patch['SDXL'] = True
    else:
        print('Base model type: ', type(model.model.model_config))
        raise Exception("Unsupported model type: ", type(model.model.model_config))

    if 'forward' not in brush_model_patch:
        brush_model_patch['forward'] = []

    if remove:
        if forward_patch in brush_model_patch['forward']:
            brush_model_patch['forward'].remove(forward_patch)
    else:
        brush_model_patch['forward'].append(forward_patch)

    brush_model_patch['step'] = 0
    brush_model_patch['total_steps'] = 1

    # apply patches to code
    comfy.patcher_extension.add_wrapper(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                        brushNet_out_sample_wrapper,
                                        patch_wrapper,
                                        is_model_options=False)
    model.wrappers.update(patch_wrapper['wrappers'])

def apply_model_function_wrapper(apply_model_executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
    baseModel = apply_model_executor.class_obj
    to = transformer_options

    timestep = t

    # check if there are patches to execute
    if 'brush_model_patch' not in to or 'forward' not in to['brush_model_patch']:
        return apply_model_executor(x, timestep, c_concat, c_crossattn, control, transformer_options, **kwargs)

    brush_model_patch = to['brush_model_patch']
    unet = baseModel.diffusion_model

    all_sigmas = brush_model_patch['all_sigmas']
    sigma = to['sigmas'][0].item()
    total_steps = all_sigmas.shape[0] - 1
    step = torch.argmin((all_sigmas - sigma).abs()).item()

    brush_model_patch['step'] = step
    brush_model_patch['total_steps'] = total_steps

    # comfy.model_base.apply_model
    xc = baseModel.model_sampling.calculate_input(timestep, x)
    if c_concat is not None:
        xc = torch.cat([xc] + [c_concat], dim=1)
    brush_t = baseModel.model_sampling.timestep(timestep).float()
    # execute all patches
    for method in brush_model_patch['forward']:
        method(unet, xc, brush_t, to, control)

    return apply_model_executor(x, timestep, c_concat, c_crossattn, control, transformer_options, **kwargs)

def brushNet_out_sample_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj
    # set hook
    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    set_brushNet_hook(diffusion_model)
    # Use cfd_guider.model_options, which is copied from modelPatcher.model_options and will be restored after execution without any unexpected contamination
    to = add_model_patch_option(cfg_guider)
    to['brush_model_patch']['all_sigmas'] = sigmas
    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    finally:
        # cleanup hook
        clean_brushNet_hook(diffusion_model)
    return out


def set_brushNet_hook(diffusion_model):
    for i, block in enumerate(diffusion_model.input_blocks):
        for j, layer in enumerate(block):
            if not hasattr(layer, 'original_forward'):
                layer.original_forward = layer.forward
            layer.forward = types.MethodType(forward_patched_by_brushnet, layer)
            layer.add_sample_after = 0
    for j, layer in enumerate(diffusion_model.middle_block):
        if not hasattr(layer, 'original_forward'):
            layer.original_forward = layer.forward
        layer.forward = types.MethodType(forward_patched_by_brushnet, layer)
        layer.add_sample_after = 0
    for i, block in enumerate(diffusion_model.output_blocks):
        for j, layer in enumerate(block):
            if not hasattr(layer, 'original_forward'):
                layer.original_forward = layer.forward
            layer.forward = types.MethodType(forward_patched_by_brushnet, layer)
            layer.add_sample_after = 0

    comfy.ldm.modules.diffusionmodules.openaimodel.original_apply_control = comfy.ldm.modules.diffusionmodules.openaimodel.apply_control
    comfy.ldm.modules.diffusionmodules.openaimodel.apply_control = modified_apply_control

def clean_brushNet_hook(diffusion_model):
    for i, block in enumerate(diffusion_model.input_blocks):
        for j, layer in enumerate(block):
            if hasattr(layer, 'original_forward'):
                layer.forward = layer.original_forward
                del layer.original_forward
                del layer.add_sample_after
    for j, layer in enumerate(diffusion_model.middle_block):
        if hasattr(layer, 'original_forward'):
            layer.forward = layer.original_forward
            del layer.original_forward
            del layer.add_sample_after
    for i, block in enumerate(diffusion_model.output_blocks):
        for j, layer in enumerate(block):
            if hasattr(layer, 'original_forward'):
                layer.forward = layer.original_forward
                del layer.original_forward
                del layer.add_sample_after

    if hasattr(comfy.ldm.modules.diffusionmodules.openaimodel, 'original_apply_control'):
        comfy.ldm.modules.diffusionmodules.openaimodel.apply_control = comfy.ldm.modules.diffusionmodules.openaimodel.original_apply_control
        del comfy.ldm.modules.diffusionmodules.openaimodel.original_apply_control

# patch layers `forward` so we can apply brushnet
def forward_patched_by_brushnet(self, x, *args, **kwargs):
    h = self.original_forward(x, *args, **kwargs)
    if hasattr(self, 'add_sample_after') and type(self):
        to_add = self.add_sample_after
        if torch.is_tensor(to_add):
            # interpolate due to RAUNet
            if h.shape[2] != to_add.shape[2] or h.shape[3] != to_add.shape[3]:
                to_add = torch.nn.functional.interpolate(to_add, size=(h.shape[2], h.shape[3]), mode='bicubic')
            h += to_add.to(h.dtype).to(h.device)
        else:
            h += self.add_sample_after
        self.add_sample_after = 0
    return h

# To use Controlnet with RAUNet it is much easier to modify apply_control a little
def modified_apply_control(h, control, name):
    '''
    Modified by BrushNet nodes
    '''
    if control is not None and name in control and len(control[name]) > 0:
        ctrl = control[name].pop()
        if ctrl is not None:
            if h.shape[2] != ctrl.shape[2] or h.shape[3] != ctrl.shape[3]:
                ctrl = torch.nn.functional.interpolate(ctrl, size=(h.shape[2], h.shape[3]), mode='bicubic').to(h.dtype).to(h.device)                    
            try:
                h += ctrl
            except:
                print.warning("warning control could not be applied {} {}".format(h.shape, ctrl.shape))
    return h    


