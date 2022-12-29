# GIFfusion
Giffusion is a Web UI for generating GIFs and Videos using Stable Diffusion.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DN6/giffusion/blob/main/Giffusion.ipynb)
[![Open In Comet](https://custom-icon-badges.herokuapp.com/badge/comet__ml-Open_In_Comet-orange?logo=logo_comet_ml)](https://www.comet.com/team-comet-ml/giffusion/view/CzxqbNrydKqHCaYhNEnbyrpnz/panels?utm_source=tds&utm_medium=social&utm_campaign=stable_diffusion)

## Features

### Multiframe Generation

Giffusion follows a prompt syntax similar to the one used in [Deforum Art's Stable Diffusion Notebook](https://deforum.github.io/)

```
0: a picture of a corgi
60: a picture of a lion
```

The first part of the prompt indicates a key frame number, while the text after the colon is the prompt used by the model to generate the image.

In the example above, we're asking the model to generate a picture of a Corgi at frame 0 and a picture of a lion at frame 60. So what about all the images in between these two key frames? How do they get generated?

You might recall that Diffusion Models work by turning noise into images. Stable Diffusion turns a noise tensor into a latent embedding in order to save time and memory when running the diffusion process. This latent embedding is fed into a decoder to produce the image.

The inputs to our model are a noise tensor and text embedding tensor. Using our key frames as our start and end points, we can produce images in between these frames by interpolating these tensors.

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

Additionally, timestamp informtion for these key frames is also extracted to the text input box for reference in case you would like to sync your prompts to a particular time in the audio.

**Note:** The key frames will change based the frame rate that you have set in the UI.

https://user-images.githubusercontent.com/7529846/204581783-7fa9ad83-baf2-4293-99e1-9315d6b557c9.mp4

### Video Initialization

You can use frames from an existing video as initial images in the diffusion process.

https://user-images.githubusercontent.com/7529846/204550451-5d2162dc-5d6b-4ecd-b1ed-c15cb56bc224.mp4

To use video initialization, head over to the Video Setting tab and upload your file. Click `Get Key Frame Information` to extract the maximum number of frames present in the video and to update the frame rate in the UI to match the frame rate of the input video.

The `Strength` parameter controls how well your original video content is preserved by the diffusion process. Setting higher values (greater than 0.5) will lead to source image semantics being ignored.

https://user-images.githubusercontent.com/7529846/204581841-3fa20dd5-7cff-4ec1-aacf-2cb2bfe476b4.mp4

### Saving to Comet

GIFfusion also support saving prompts, generated GIFs/Videos, images, and settings to [Comet](https://www.comet.com/site/) so you can keep track of your generative experiments.

[Check out an example project here with some of my GIFs!](https://www.comet.com/team-comet-ml/giffusion?shareable=Jf4go5RcGqryr6wq1uBudgVVS)

### Additional Resources

1. Prompt format is based on the work from [Deforum Art](https://deforum.github.io/)
2. Inspiration Button uses the [Midjourney Prompt Generator](https://huggingface.co/spaces/doevent/prompt-generator) Space by DoEvent 
3. [Stable Diffusion Videos with Audio Reactivity](https://github.com/nateraw/stable-diffusion-videos)
4. [Comet ML Project with some of the things made with Giffusion](https://www.comet.com/team-comet-ml/giffusion/view/CzxqbNrydKqHCaYhNEnbyrpnz/panels?utm_source=tds&utm_medium=social&utm_campaign=stable_diffusion)
5. [Gradio Docs](https://gradio.app/docs/): The UI for this project is built with Gradio.