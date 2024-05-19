During investigation of compatibility issues with [WASasquatch's FreeU_Advanced](https://github.com/WASasquatch/FreeU_Advanced/tree/main) and [blepping's jank HiDiffusion](https://github.com/blepping/comfyui_jankhidiffusion) nodes I stumbled upon some quite hard problems. There are `FreeU` nodes in ComfyUI, but no such for HiDiffusion, so I decided to implement RAUNet on base of my BrushNet implementation. **blepping**, I am sorry. :)

### RAUNet

What is RAUNet? I know many of you saw and generate images with a lot of limbs, fingers and faces all morphed together.

The authors of HiDiffusion invent simple, yet efficient trick to alleviate this problem. Here is an example:

![example workflow](example/RAUNet1.png?raw=true)

[workflow](example/RAUNet_basic.json)

The left picture is created using ZavyChromaXL checkpoint on 2048x2048 canvas. The right one uses RAUNet.

In my experience the node is helpful but quite sensitive to its parameters. And there is no universal solution - you should adjust them for every new image you generate. It also lowers model's imagination, you usually get only what you described in the prompt. Look at the example: in first you have a forest in the background, but RAUNet deleted all except fox which is described in the prompt.

From the [paper](https://arxiv.org/abs/2311.17528): Diffusion models denoise from structures to details. RAU-Net introduces additional downsampling and upsampling operations, leading to a certain degree of information loss. In the early stages of denoising, RAU-Net can generate reasonable structures with minimal impact from information loss. However, in the later stages of denoising when generating fine details, the information loss in RAU-Net results in the loss of image details and a degradation in quality.

### Parameters

There are two independent parts in this node: DU (Downsample/Upsample) and XA (CrossAttention). The four parameters are the start and end steps for applying these parts. 

The Downsample/Upsample part lowers models degrees of freedom. If you apply it a lot (for more steps) the resulting images will have a lot of symmetries.

The CrossAttension part lowers number of objects which model tracks in image.

Usually you apply DU and after several steps apply XA, sometimes you will need only XA, you should try it yourself.

### Compatibility

It is compatible with BrushNet and most other nodes.

This is ControlNet example. The lower image is pure model, the upper is after using RAUNet. You can see small fox and two tails in lower image.

![example workflow](example/RAUNet2.png?raw=true)

[workflow](example/RAUNet_with_CN.json)

The node can be implemented for any model. Right now it can be applied to SD15 and SDXL models.