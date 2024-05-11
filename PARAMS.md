## Start At and End At parameters usage

### start_at

Let's start with a ELLA outpaint [workflow](example/BrushNet_with_ELLA.json) and switch off Blend Inpaint node:

![example workflow](example/params1.png?raw=true)

For this example I use "wargaming shop showcase" prompt, `dpmpp_2m` deterministic sampler and `karras` scheduler with 15 steps. This is the result:

![goblin in the shop](example/params2.png?raw=true)

The `start_at` BrushNet node parameter allows us to delay BrushNet inference for some steps, so the base model will do all the job. Let's see what the result will be without BrushNet. For this I set up `start_at` parameter to 20 - it should be more then `steps` in KSampler node:

![the shop](example/params3.png?raw=true)

So, if we apply BrushNet from the beginning (`start_at` equals 0), the resulting scene will be heavily influenced by BrushNet image. The more we increase this parameter, the more scene will be based on prompt. Let's compare:

| `start_at` = 1 | `start_at` = 2 | `start_at` = 3 | 
|:--------------:|:--------------:|:--------------:|
| ![p1](example/params4.png?raw=true) | ![p2](example/params5.png?raw=true) | ![p3](example/params6.png?raw=true) |
| `start_at` = 4 | `start_at` = 5 | `start_at` = 6 |
| ![p1](example/params7.png?raw=true) | ![p2](example/params8.png?raw=true) | ![p3](example/params9.png?raw=true) |
|  `start_at` = 7 |  `start_at` = 8 |  `start_at` = 9 |
| ![p1](example/params10.png?raw=true) | ![p2](example/params11.png?raw=true) | ![p3](example/params12.png?raw=true) |

Look how the floor is aligned with toy's base - at some step it looses consistency. The results will depend on type of sampler and number of KSampler steps, of course.

### end_at

The `end_at` parameter switches off BrushNet at the last steps. If you use deterministic sampler it will only influences details on last steps, but stochastic samplers can change the whole scene. For a description of samplers see, for example, Matteo Spinelli's [video on ComfyUI basics](https://youtu.be/_C7kR2TFIX0?t=516).

Here I use basic BrushNet inpaint [example](example/BrushNet_basic.json), with "intricate teapot" prompt, `dpmpp_2m` deterministic sampler and `karras` scheduler with 15 steps:

![example workflow](example/params13.png?raw=true)

There are almost no changes when we set 'end_at' paramter to 10, but starting from it:

| `end_at` = 10 | `end_at` = 9 | `end_at` = 8 | 
|:--------------:|:--------------:|:--------------:|
| ![p1](example/params14.png?raw=true) | ![p2](example/params15.png?raw=true) | ![p3](example/params16.png?raw=true) |
| `end_at` = 7 | `end_at` = 6 | `end_at` = 5 |
| ![p1](example/params17.png?raw=true) | ![p2](example/params18.png?raw=true) | ![p3](example/params19.png?raw=true) |
|  `end_at` = 4 |  `end_at` = 3 |  `end_at` = 2 |
| ![p1](example/params20.png?raw=true) | ![p2](example/params21.png?raw=true) | ![p3](example/params22.png?raw=true) |

You can see how the scene was completely redrawn.
