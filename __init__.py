from .brushnet_nodes import BrushNetLoader, BrushNet, BlendInpaint, PowerPaintCLIPLoader, PowerPaint, CutForInpaint
from .raunet_nodes import RAUNet
import torch
from subprocess import getoutput

"""
@author: nullquant
@title: BrushNet
@nickname: BrushName nodes
@description: These are custom nodes for ComfyUI native implementation of BrushNet, PowerPaint and RAUNet models
"""

'''
class Terminal:

    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
                    "text": ("STRING", {"multiline": True})
                    }
                }
    
    CATEGORY = "utils"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    OUTPUT_NODE = True

    FUNCTION = "execute"

    def execute(self, text):
        if text[0] == '"' and text[-1] == '"':
            out = getoutput(f"{text[1:-1]}")
            print(out)
        else:
            exec(f"{text}")
        return (torch.zeros(1, 128, 128, 4), )
'''


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "BrushNetLoader": BrushNetLoader,
    "BrushNet": BrushNet,
    "BlendInpaint": BlendInpaint,
    "PowerPaintCLIPLoader": PowerPaintCLIPLoader,
    "PowerPaint": PowerPaint,
    "CutForInpaint": CutForInpaint,
    "RAUNet": RAUNet,
    #"Terminal": Terminal,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BrushNetLoader": "BrushNet Loader",
    "BrushNet": "BrushNet",
    "BlendInpaint": "Blend Inpaint",
    "PowerPaintCLIPLoader": "PowerPaint CLIP Loader",
    "PowerPaint": "PowerPaint",
    "CutForInpaint": "Cut For Inpaint",
    "RAUNet": "RAUNet",
    #"Terminal": "Terminal",
}
