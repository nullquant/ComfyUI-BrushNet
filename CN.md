## ControlNet Canny Edge

Let's take the pestered cake and try to inpaint it again. Now I would like to use a sleeping cat for it:

![sleeping cat](example/sleeping_cat.png?raw=true)

I use Canny Edge node from [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux). Don't forget to resize canny edge mask to 512 pixels:

![sleeping cat inpaint](example/sleeping_cat_inpaint1.png?raw=true)

Let's look at the result:

![sleeping cat inpaint](example/sleeping_cat_inpaint2.png?raw=true)

The first problem I see here is some kind of object behind the cat. Such objects appear since the inpainting mask strictly aligns with the removed object, the cake in our case. To remove such artifact we should expand our mask a little:

![sleeping cat inpaint](example/sleeping_cat_inpaint3.png?raw=true)

Now. what's up with cat back and tail? Let's see the inpainting mask and canny edge mask side to side:

![masks](example/sleeping_cat_inpaint4.png?raw=true)

The inpainting works (mostly) only in masked (white) area, so we cut off cat's back. **The ControlNet mask should be inside the inpaint mask.**

To address the issue I resized the mask to 256 pixels:

![sleeping cat inpaint](example/sleeping_cat_inpaint5.png?raw=true)

This is better but still have a room for improvement. The problem with edge mask downsampling is that edge lines tend to be broken and after some size we will got a mess:

![sleeping cat inpaint](example/sleeping_cat_inpaint6.png?raw=true)

Look at the edge mask, at this resolution it is so broken:

![masks](example/sleeping_cat_mask.png?raw=true)




