# GIFfusion
Giffusion is a Web UI for generating GIFs and Videos using Stable Diffusion.

Try it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DN6/giffusion/blob/main/Giffusion.ipynb)

## Features

### Multiframe Generation

Giffusion follows a prompt syntax similar to the one used in [Deforum Art's Stable Diffusion Notebook](https://deforum.github.io/)

Provide prompts for specific key frames in your GIF or Video, and Giffusion will interpolate between them to fill in the rest. For example, the prompt below will generate images for frame 0 and 60. The frames in between will created by interpolating between the prompts.

```
0: a picture of a corgi
60: a picture of a lion
```

![output-corgi-final](https://user-images.githubusercontent.com/7529846/204506200-49f91bd1-396f-4cf1-927c-c91b885f5c4a.gif)

### Composable Diffusion

Giffusion supports [Composable Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) for image generation.

Composable diffusion tends to preserve the components of the individual prompts better than a single text prompt.

To compose your prompts, simply separate them using a '|'. For example

```
0: A red house | a lightning storm
60: A red house | a sunny day
```

![red-house-final](https://user-images.githubusercontent.com/7529846/204506605-f1d89d99-9449-4ba6-82e0-3cc2a1f863c6.gif)

### Inspiration Button

Creating prompts can be challenging. Click the `Give me some inspiration` button to automatically generate prompts for you.

![giffusion-inspo-gif](https://user-images.githubusercontent.com/7529846/191538441-0a27d0f8-f07f-41ea-8653-136f73802fbf.gif)

### Audio Reactive Videos

Drive your GIF and Video animations using audio.

https://user-images.githubusercontent.com/7529846/204550897-70777873-30ca-46a9-a74e-65b6ef429958.mp4

### Video Initialization

You can use frames from an existing video as initial images in the diffusion process.

https://user-images.githubusercontent.com/7529846/204550451-5d2162dc-5d6b-4ecd-b1ed-c15cb56bc224.mp4


### Saving to Comet

GIFfusion also support saving prompts, generated GIFs, images, and settings to [Comet](https://www.comet.com/site/) so you can keep track of your generative experiments.

[Check out an example project here with some of my GIFs!](https://www.comet.com/team-comet-ml/giffusion?shareable=Jf4go5RcGqryr6wq1uBudgVVS)
