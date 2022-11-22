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

![output-corgi](https://user-images.githubusercontent.com/7529846/203226118-cbd83da1-f1d7-47f7-b7a3-e13a9c73d67e.gif)


### Composable Diffusion

Giffusion supports [Composable Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) for image generation.

Composable diffusion tends to preserve the components of the individual prompts better than a single text prompt.

To compose your prompts, simply separate them using a '|'. For example

```
0: A red house | a lightning storm
60: A red house | a sunny day
```
![redhouse-output-small](https://user-images.githubusercontent.com/7529846/191756380-2077f2fb-f39e-4a6f-a4cb-ff4bff3eb8ac.gif)

### Inspiration Button

Creating prompts can be challenging. Click the `Give me some inspiration` button to automatically generate prompts for you.

![giffusion-inspo-gif](https://user-images.githubusercontent.com/7529846/191538441-0a27d0f8-f07f-41ea-8653-136f73802fbf.gif)

### Saving to Comet

GIFfusion also support saving prompts, generated GIFs, images, and settings to [Comet](https://www.comet.com/site/) so you can keep track of your generative experiments.

[Check out an example project here with some of my GIFs!](https://www.comet.com/team-comet-ml/giffusion?shareable=Jf4go5RcGqryr6wq1uBudgVVS)
