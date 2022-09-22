# GIFfusion
Giffusion is an application to generate GIFs using Stable Diffusion.

Try it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DN6/giffusion/blob/main/Giffusion.ipynb)

## Features

### Multiframe Generation

Provide prompts for key frames in your GIF and Giffusion will interpolate between them to fill in the rest. For example, the prompt below will generate images for frame 0 and 60. The frames in between will created based on interpolation between the prompts.

```
0: A picture of a corgi
60: A picture of a labradoodle
```

### Composable Diffusion

Giffusion supports [Composable Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) for image generation.

Composable diffusion tends to preserve the components of the individual prompts better than a single text prompt.

To compose your prompts, simply separate them using an '|'. For example

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
