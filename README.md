# GIFfusion
Giffusion is a Web UI for generating GIFs and Videos using Stable Diffusion.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DN6/giffusion/blob/main/Giffusion.ipynb)
[![Open In Comet](https://custom-icon-badges.herokuapp.com/badge/comet__ml-Open_In_Comet-orange?logo=logo_comet_ml)](https://www.comet.com/team-comet-ml/giffusion/view/CzxqbNrydKqHCaYhNEnbyrpnz/panels?utm_source=tds&utm_medium=social&utm_campaign=stable_diffusion)

## Features

### Bring Your Own Pipeline

Giffusion supports using any pipeline and compatible checkpoint from the [Diffusers](https://huggingface.co/docs/diffusers/index) library. Simply paste in the checkpoint name and pipeline name in the `Pipeline Settings`

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

### Inspiration Button

Creating prompts can be challenging. Click the `Give me some inspiration` button to automatically generate prompts for you.

You can even provide a list of topics for the inspiration button to use as a starting point.

### Multimedia Support

#### Image Input
You can seed the generation process with an inital image. Upload your file using the, using the `Image Input` dropdown.

#### Audio Input

Drive your GIF and Video animations using audio.

https://user-images.githubusercontent.com/7529846/204550897-70777873-30ca-46a9-a74e-65b6ef429958.mp4

In order to use audio to drive your animations,

1. Head over to the `Audio Input` dropdown and upload your audio file.
2. Click `Get Key Frame Information`. This will extract key frames from the audio based on the `Audio Component` you have selected. You can extract key frames based on the percussive, harmonic or combined audio components of your file.

Additionally, timestamp information for these key frames is also extracted for reference in case you would like to sync your prompts to a particular time in the audio.

**Note:** The key frames will change based the frame rate that you have set in the UI.

#### Video Input

You can use frames from an existing video as initial images in the diffusion process.

https://user-images.githubusercontent.com/7529846/204550451-5d2162dc-5d6b-4ecd-b1ed-c15cb56bc224.mp4

To use video initialization:

1. Head over to the `Video Input` dropdown

2. Upload your file. Click `Get Key Frame Information` to extract the maximum number of frames present in the video and to update the frame rate setting in the UI to match the frame rate of the input video.

#### Resampling Output Generations

You can resample videos and GIFs created in the output tab and send them either to the Image Input or Video Input.

**Resamplng to Image Input**

To sample an image from a video, select the frame id you want to sample from your output video or GIF and click on `Send to Image Input`

**Resampling to Video Input**

**Note:** This option only works if your output format is set to `mp4`

To resample a video, click on `Send to Video Input`

### Saving to Comet

GIFfusion also support saving prompts, generated GIFs/Videos, images, and settings to [Comet](https://www.comet.com/site/) so you can keep track of your generative experiments.

[Check out an example project here with some of my GIFs!](https://www.comet.com/team-comet-ml/giffusion?shareable=Jf4go5RcGqryr6wq1uBudgVVS)

## Diffusion Settings

This section covers all the components in the Diffusion Settings dropdown.

1. **Use Fixed Latent:** Use the same noise latent for every frame of the generation process. This is useful if you want to keep the noise latent fixed while interpolating over just the prompt embeddings.

2. **Use Prompt Embeds:** By default, Giffusion converts your prompts into embeddings and interpolates between the prompt embeddings for the in between frames. If you disable this option, Giffusion will forward fill the text prompts between frames instead. If you are using the `ComposableDiffusion` pipeline or would like to use the prompt embedding function of the pipeline directly, disable this option.

3. **Numerical Seed:** Seed for the noise latent generation process. If `Use Fixed Latent` isn't set, this seed is used to generate a schedule that provides a unique seed for each key frame.

4. **Number of Iteration Steps:** Number of steps to use in the generation process.

5. **Classifier Free Guidance Scale:** Higher guidance scale encourages generated images that are closely linked to the text prompt, usually at the expense of lower image quality.

6. **Image Strength:** Indicates how much to transform the reference image. Must be between 0 and 1. The image will be used as a starting point, adding more noise to it larger the strength. This is only applicable to Pipelines that support images as inputs.

7. **Scheduler:**  Schedulers take in the output of a trained model, a sample which the diffusion process is iterating on, and a timestep to return a denoised sample. The different schedulers require a different number of iteration steps to produce good results. Use this selector to experiment with different schedulers and pipelines.

8. **Batch Size:** Set the batch size used in the generation process. If you have access to a GPU with more memory, increase the batch size to increase the speed of the generation process.

9. **Image Height:** By default, generated images will have a height of 512 pixels. Certain models and pipelines support generating higher resolution images. Adjust this setting to account for those configurations. If an Image or Video input is provided, the height is set to the height of the original input.

10. **Image Width:** By default, generated images will have a width of 512 pixels. Certain models and pipelines support generating higher resolution images. Adjust this setting to account for those configurations. If an Image or Video input is provided, the width is set to the width of the original input.

11. **Number of Latent Channels:** This is used to set the channel dimension of the noise latent. Certain Pipelines, e.g. `InstructPix2Pix` require the number of latent channels to be different from the number of input channels of the Unet model. The default value of `4` should work for a majority of pipelines and models.

## Output Settings

1. **Output Format:** Set the output format to either be a GIF or an MP4 video.
2. **Frame Rate:** Set the frame rate for the output.

## References

1. Prompt format is based on the work from [Deforum Art](https://deforum.github.io/)
2. Inspiration Button uses the [Midjourney Prompt Generator](https://huggingface.co/spaces/doevent/prompt-generator) Space by DoEvent 
3. [Stable Diffusion Videos with Audio Reactivity](https://github.com/nateraw/stable-diffusion-videos)
4. [Comet ML Project with some of the things made with Giffusion](https://www.comet.com/team-comet-ml/giffusion/view/CzxqbNrydKqHCaYhNEnbyrpnz/panels?utm_source=tds&utm_medium=social&utm_campaign=stable_diffusion)
5. [Gradio Docs](https://gradio.app/docs/): The UI for this project is built with Gradio.
6. [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)