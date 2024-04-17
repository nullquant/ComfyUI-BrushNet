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

The checkpoint in `segmentation_mask_brushnet_ckpt` provides checkpoints trained on BrushData, which has segmentation prior (mask are with the same shape of objects). The `random_mask_brushnet_ckpt` provides a more general ckpt for random mask shape.

Both `segmentation_mask_brushnet_ckpt` and `random_mask_brushnet_ckpt` or only one of them should be placed inside `models/inpaint/brushnet` folder.

Base model should be SD1.5 type (realisticVisionV60B1_v51VAE, for example).

This is complete rework of previous version. There is no more custom `diffusers` library. 
There is no loadings from github. I also changed the workflow so now it is possible to use LoRa models.
To users who already have previoous version: please, uninstall `diffusers` library before installation:

```
pip uninstall diffusers
```

Then install as usual.

## Usage

Below is an example for the intended workflow. The [json file](example/BrushNet_basic.json) for the example can be found inside the 'example' directory.

![example workflow](example/BrushNet_basic.png?raw=true)

This is example with LoRA: [json file](example/BrushNet_with_LoRA.json).

![example workflow](example/BrushNet_with_LoRA.png?raw=true)


## Credits

The code is based on 

- [BrushNet](https://github.com/TencentARC/BrushNet)
- [PowerPaint](https://github.com/zhuang2002/PowerPaint)
- [diffusers](https://github.com/huggingface/diffusers)
