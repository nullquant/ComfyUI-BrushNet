## ComfyUI-BrushNet

Custom nodes for ComfyUI allow to inpaint using Brushnet:  ["BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion"](https://arxiv.org/abs/2403.06976).

My contribution is limited to the ComfyUI adaptation, and all credit goes to the authors of the paper.


## Installation

Clone the repo into the custom_nodes directory and install the requirements:

```
git clone https://github.com/nullquant/ComfyUI-BrushNet.git
pip install -r requirements.txt
```

Checkpoints of BrushNet can be downloaded from [here](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n?usp=drive_link). 
Both `segmentation_mask_brushnet_ckpt` and `random_mask_brushnet_ckpt` should be placed inside `models/inpaint/brushnet` folder.
Base model should be SD1.5 type (realisticVisionV60B1_v51VAE, for example).

## Usage

Below is an example for the intended workflow. The [json file](example/BrushNet_example.json) for the example can be found inside the 'workflow' directory.

![example workflow](example/BrushNet_example.png?raw=true)


## Bugs

1. Nodes use custom `diffusers` library. It can potentially conflicts with other custom nodes.
2. Sometimes `BrushNetPipeline` freezes on loading. Looking into the problem.


## Credits

The code is based on 

- [BrushNet](https://github.com/TencentARC/BrushNet)
- [PowerPaint](https://github.com/zhuang2002/PowerPaint)
- [diffusers](https://github.com/huggingface/diffusers)
