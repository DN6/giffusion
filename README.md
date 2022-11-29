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

https://user-images.githubusercontent.com/7529846/204581619-d9f3a550-9f0c-4b4b-9783-b558e6ebf109.mp4

You can even provide a list of topics for the inspiration button to use as a starting point.

https://user-images.githubusercontent.com/7529846/204581713-d94f28b3-b7c9-4ad4-8bc3-b287ac849968.mp4

### Audio Reactive Videos

Drive your GIF and Video animations using audio.

https://user-images.githubusercontent.com/7529846/204550897-70777873-30ca-46a9-a74e-65b6ef429958.mp4

In order to use audio, head over to the Audio Settings tab and upload your audio file. Then click `Get Key Frame Information`. This will extract key frames from the audio based on the `Audio Component` you have selected. You can extract key frames from the percussive, harmonic or combined audio components of your file.

**Note:** The key frames will change based the frame rate that you have set in the UI.

https://user-images.githubusercontent.com/7529846/204581783-7fa9ad83-baf2-4293-99e1-9315d6b557c9.mp4

### Video Initialization

You can use frames from an existing video as initial images in the diffusion process.

https://user-images.githubusercontent.com/7529846/204550451-5d2162dc-5d6b-4ecd-b1ed-c15cb56bc224.mp4

To use video initialization, head over to the Video Setting tab and upload your file. The `Strength` parameter controls how well your original video content is preserved by the diffusion process. Setting higher values (greater than 0.5) will lead to the source video being ignored.

### Saving to Comet

GIFfusion also support saving prompts, generated GIFs/Videos, images, and settings to [Comet](https://www.comet.com/site/) so you can keep track of your generative experiments.

[Check out an example project here with some of my GIFs!](https://www.comet.com/team-comet-ml/giffusion?shareable=Jf4go5RcGqryr6wq1uBudgVVS)
