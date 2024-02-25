# GIFfusion 💥

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/220882002-72cbfdef-876a-4cb2-9f41-e5989e769868.gif" width="256" title="hover text">
</p>

Giffusion is a Web UI for generating GIFs and Videos using Stable Diffusion.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DN6/giffusion/blob/main/Giffusion.ipynb)
[![Open In Comet](https://custom-icon-badges.herokuapp.com/badge/comet__ml-Open_In_Comet-orange?logo=logo_comet_ml)](https://www.comet.com/team-comet-ml/giffusion/view/CzxqbNrydKqHCaYhNEnbyrpnz/panels?utm_source=tds&utm_medium=social&utm_campaign=stable_diffusion)

## To Run

### In Colab
Open the Colab Notebook linked above and follow the instructions to start the Giffusion UI

### On your local machine

Clone the Giffusion repository
```
git clone https://github.com/DN6/giffusion.git && cd giffusion
```

Install the requirements

```
pip install -r requirements.txt
```

Giffusion uses the Huggingface Hub to download models. Set the following environment variables to ensure that your model cache is set properly.

```shell
export HF_HOME=<path to your Huggingface Hub home directory> # defaults to ~/.cache/huggingface
export MODEL_PATH=$HF_HOME/hub
```

Start the application

```
python app.py
```

## Features

### Saving and Loading Sessions

Giffusion uss the Hugging Face Hub to save output generations and settings. To save and load sessions, you will first need to set your Hugging Face access token using `huggingface-cli login`.

Once set, you can save your session by clicking on the `Save Session` button in the Session Settings. This will create a dataset repo on the Hub and save your settings and output generations to a folder with a randomly generated name. You can also set the Repo ID and Session Name manually in order to save your session to a specific repo.

Loading sessions works in a similar manner. Simply provide the Repo ID and Session Name of the session you would like to load and click on the `Load Session` button. You can filter the settings for the individual components in the UI using the dropdown selector.

### Bring Your Own Pipeline

Giffusion supports using any pipeline and compatible checkpoint from the [Diffusers](https://huggingface.co/docs/diffusers/index) library. Simply paste in the checkpoint name and pipeline name in the `Pipeline Settings`

#### ControlNet Support

Giffusion allows you to use the `StableDiffusionControlNetPipeline`. Simply paste in the ControlNet checkpoint you would like to use to load in the Pipeline.

MultiControlnet's are also supported. Just paste in a list of model checkpoint paths from the Hugging Face Hub

```text
lllyasviel/control_v11p_sd15_softedge, lllyasviel/control_v11f1p_sd15_depth
```

**Notes on Preprocessing:** When using Controlnets, you need to preprocess your inputs before using them as conditioning signals for the model. The Controlnet Preprocessing Settings allow you to choose a set of preprocessing options to apply to your image. Be sure to select them in the same order as your Controlnet models. For example, for the code snippet above, you would have to select the softedge preprocessor before the depth one. If you are using a Controlnet model that requires no processing that in a MultiControlnet setting, a `no-processing` option is also provided.

<p align="center">
  <img width="341" alt="Screenshot 2023-07-26 at 11 41 11 PM" src="https://user-images.githubusercontent.com/7529846/256476148-fc0dc1ad-ed26-435c-9850-8c9cb7f9a789.png">
</p>

#### Custom Pipeline Support

You can use your own custom pipelines with Giffusion as well. Simply paste in the path to your Pipeline file in the `Custom Pipeline` section. The Pipeline file must follow a format similar to the [community pipelines found in Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community)


### Compel Prompt Weighting Support

Prompt Embeds are now generated via [Compel](https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts) and support the weighting syntanx outlined [here](https://github.com/damian0815/compel)

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

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/204506200-49f91bd1-396f-4cf1-927c-c91b885f5c4a.gif" width="256" title="hover text">
</p>

### Inspiration Button

Creating prompts can be challenging. Click the `Give me some inspiration` button to automatically generate prompts for you.

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/220324203-444c1720-c71b-4ccf-b08f-5b20668b7f98.gif" width="800" title="hover text">
</p>

You can even provide a list of topics for the inspiration button to use as a starting point.

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/220324835-fbbae3be-9a9a-48f9-a773-5e45c6274ed2.gif" width="800" title="hover text">
</p>


### Multimedia Support

Augment the image generation process with additional media inputs

<details>
<summary>Image Input</summary>

You can seed the generation process with an inital image. Upload your file using the, using the `Image Input` dropdown.

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/220880564-dba393c5-6023-4539-a59c-c33758769500.gif" width="800" title="hover text">
</p>
<p align="center">
<a align="center" href="https://www.krea.ai/prompt/184bf3cf-ec0d-4ff8-b4f1-45577799700b">Image Source</a>
</p>
</details>

<details>
<summary>Audio Input</summary>

Drive your GIF and Video animations using audio.

https://user-images.githubusercontent.com/7529846/204550897-70777873-30ca-46a9-a74e-65b6ef429958.mp4

In order to use audio to drive your animations,

1. Head over to the `Audio Input` dropdown and upload your audio file.
2. Click `Get Key Frame Information`. This will extract key frames from the audio based on the `Audio Component` you have selected. You can extract key frames based on the percussive, harmonic or combined audio components of your file.

Additionally, timestamp information for these key frames is also extracted for reference in case you would like to sync your prompts to a particular time in the audio.

**Note:** The key frames will change based the frame rate that you have set in the UI.
</details>

<details>
<summary>Video Input</summary>

You can use frames from an existing video as initial images in the diffusion process.

https://user-images.githubusercontent.com/7529846/204550451-5d2162dc-5d6b-4ecd-b1ed-c15cb56bc224.mp4

To use video initialization:

1. Head over to the `Video Input` dropdown

2. Upload your file. Click `Get Key Frame Information` to extract the maximum number of frames present in the video and to update the frame rate setting in the UI to match the frame rate of the input video.

</details>

#### Resampling Output Generations

You can resample videos and GIFs created in the output tab and send them either to the Image Input or Video Input.

<details>
<summary>Resampling to Image Input</summary>

To sample an image from a video, select the frame id you want to sample from your output video or GIF and click on `Send to Image Input`

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/220325938-22438722-d4ac-4a35-995f-51d8dbafaa34.gif" width="800" title="hover text">
</p>
</details>

<details>
<summary>Resampling to Video Input</summary>

To resample a video, click on `Send to Video Input`

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/220322852-f2fab800-43dc-41b8-bdb4-c4057bb65a5f.gif" width="800" title="hover text">
</p>

</details>

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

6. **Image Strength Schedule:** Indicates how much to transform the reference image. Must be between 0 and 1. Larger strength values will perform more denoising steps. This is only applicable to `Img2Img` type Pipelines. The schedule follows a similar format to motion inputs. e.g. `0:(0.5), 10:(0.7)` will ramp up the strength value from `0.5` to `0.7` between frames 0 to 10.

7. **Use Default Pipeline Scheduler:** Select to use the scheduler that has been preconfigured with the Pipeline.

8. **Scheduler:**  Schedulers take in the output of a trained model, a sample which the diffusion process is iterating on, and a timestep to return a denoised sample. The different schedulers require a different number of iteration steps to produce good results. Use this selector to experiment with different schedulers and pipelines.

9. **Scheduler Arguments:** Additional Keyword arguments to pass to the selected scheduler.

10. **Batch Size:** Set the batch size used in the generation process. If you have access to a GPU with more memory, increase the batch size to increase the speed of the generation process.

11. **Image Height:** By default, generated images will have a height of 512 pixels. Certain models and pipelines support generating higher resolution images. Adjust this setting to account for those configurations. If an Image or Video input is provided, the height is set to the height of the original input.

12. **Image Width:** By default, generated images will have a width of 512 pixels. Certain models and pipelines support generating higher resolution images. Adjust this setting to account for those configurations. If an Image or Video input is provided, the width is set to the width of the original input.

13. **Number of Latent Channels:** This is used to set the channel dimension of the noise latent. Certain Pipelines, e.g. `InstructPix2Pix` require the number of latent channels to be different from the number of input channels of the Unet model. The default value of `4` should work for a majority of pipelines and models.

14. **Additional Pipeline Arguments:** Diffuser Pipelines support a wide variety of arguments depending on the task. Use this textbox to input a dictionary of values that will be passed to the pipeline object as keyword arguments. e.g. Passing the Image Guidance Scale parameter to the InstructPix2PixPipeline

## Animation Settings

### Interpolation Type

Giffusion generates animations by first generating prompt embeddings and initial latents for the provided key frames and then interpolating the inbetween values using spherical interpolation. The schedule that controls the rate of change between interpolated values is `linear` by default.

You are free to change this schedule to using this dropdown to either `sine` or `curve`.

**Sine:**

Using the `sine` schedule will interpolate between your start and end latents and embeddings using the following function `np.sin(np.pi * frequency) ** 2` with a default frequency of value of `1.0`. This will produce a single oscillation that will cause the generated output to move from your start prompt to the end prompt and back. Doubling the frequency double the number of oscillations.

Sine interpolation also supports using multiple frequencies. An input of `1.0, 2.0` to the `Interpolation Arguments` will combine two sine waves with those frequencies.

<details>
<summary>Sine Interpolation</summary>

<p align="center">
  <img src="https://user-images.githubusercontent.com/7529846/225011513-cb4a1940-cc15-47b5-8c83-b86e88faeb3a.gif" width="512" title="hover text">
</p>

</details>

**Curve:**

You can also manually define an interpolation curve for your animation using [Chigozie Nri's Keyframe DSL](https://www.chigozie.co.uk/keyframe-string-generator/) which follows the [Deforum format.](https://docs.google.com/document/d/1RrQv7FntzOuLg4ohjRZPVL7iptIyBhwwbcEYEW2OfcI/edit)

An example curve would be

```
0: (0.0), 50: (1.0), 60: (0.5)
```

Curve values must be between 0.0 and 1.0

### Motion Settings

Giffusion allows you to use key frame animation strings to control the angle, zoom and translation of the image across frames. These animation strings follow the exact format as [Deforum](https://docs.google.com/document/d/1RrQv7FntzOuLg4ohjRZPVL7iptIyBhwwbcEYEW2OfcI/edit). Currently, Giffusion only supports 2D animation and allows you to control the following parameters

- Zoom: Scales the canvas size, multiplicatively. 1 is static, with numbers greater than 1 moving forwards and numbers less than 1 moving backward.
- Angle:  Rolls the canvas clockwise or counterclockwise in degrees per frame. This parameter uses positive values to roll counterclockwise and negative values to roll clockwise.
- Translation X: Number of pixels to shift in the X direction. Moves the canvas left or right. This parameter uses positive values to move right and negative values to move left.
- Translation Y: Number of pixels to shift in the Y direction. Moves the canvas up or down. This parameter uses positive values to move up and negative values to move down.

**Zoom Parameter Example**
```
0: (1.05),1: (1.05),2: (1.05),3: (1.05),4: (1.05),5: (1.05),6: (1.05),7: (1.05),8: (1.05),9: (1.05),10: (1.05)
```

**Angle Parameter Example**
```
0: (10.0),1: (10.0),2: (10.0),3: (10.0),4: (10.0),5: (10.0),6: (10.0),7: (10.0),8: (10.0),9: (10.0),10: (10.0)
```

**Translation X/Y Parameter Example**
```
0: (5.0),1: (5.0),2: (5.0),3: (5.0),4: (5.0),5: (5.0),6: (5.0),7: (5.0),8: (5.0),9: (5.0),10: (5.0)
```

### Coherence

Coherence is a method to preserve features across frames when creating animations. It is only applicable to models that produce a latent code while running the diffusion process. In order to do this, we compute the gradient of the current latent with respect to a reference latent (usually the latent of the previous frame)

```python
# compute the gradient for the current latent wrt the reference latent
for step in range(coherence_steps):
    loss = (latents - reference_latent).pow(2).mean()
    cond_grad = torch.autograd.grad(loss, latents)[0]

    latents = latents - (coherence_scale * cond_grad)

# update the reference latent based on coherence alpha value
reference_latent = (coherence_alpha * latents) + (
    1.0 - coherence_alpha
) * reference_latent
```

1. **Coherence Scale:** Increasing this value will make the current frame look more like the reference frame
2. **Coherence Alpha:** Controls how much to blend the current frame's latent code with the reference frame's latent code. Increasing the value will weigh more recent frames when computing the gradient.
3. **Coherence Steps:** How often to apply the callback during the diffusion process. e.g. Setting this to 2, will run the callback on every 2nd step in the diffusion process.
4. **Noise Schedule:** Amount of noise to add to a latent code for diffusion diversity. Higher values lead to more diversity. Noise is only applied if Coherence is greater than 0.0
5. **Apply Color Matching:** Apply LAB histogram color matching to the current frame using the first generated frame as a reference. This can help reduce dramatic changes in color across images during the generation process.

## Output Settings

1. **Output Format:** Set the output format to either be a GIF or an MP4 video.
2. **Frame Rate:** Set the frame rate for the output.

## References

Giffusion would not be possible without the following resources ❤️

1. Prompt format is based on the work from [Deforum Art](https://deforum.github.io/)
2. Inspiration Button uses the [Midjourney Prompt Generator](https://huggingface.co/spaces/doevent/prompt-generator) Space by DoEvent 
3. [Stable Diffusion Videos with Audio Reactivity](https://github.com/nateraw/stable-diffusion-videos)
4. [Comet ML Project with some of the things made with Giffusion](https://www.comet.com/team-comet-ml/giffusion/view/CzxqbNrydKqHCaYhNEnbyrpnz/panels?utm_source=tds&utm_medium=social&utm_campaign=stable_diffusion)
5. [Gradio Docs](https://gradio.app/docs/): The UI for this project is built with Gradio.
6. [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
7. [Keyframed](https://github.com/dmarx/keyframed) for curve interpolation
