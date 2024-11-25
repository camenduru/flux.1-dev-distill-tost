import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS, load_custom_node
from comfy_extras import nodes_flux, nodes_model_advanced, nodes_custom_sampler, nodes_sd3

load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Adaptive-Guidance")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Detail-Daemon")

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
TripleCLIPLoader = nodes_sd3.NODE_CLASS_MAPPINGS["TripleCLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
StyleModelLoader =  NODE_CLASS_MAPPINGS["StyleModelLoader"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()

ModelSamplingFlux = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
AdaptiveGuidance = NODE_CLASS_MAPPINGS["AdaptiveGuidance"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
DetailDaemonSamplerNode = NODE_CLASS_MAPPINGS["DetailDaemonSamplerNode"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("consolidated_s6700.safetensors", "default")[0]
    clip = TripleCLIPLoader.load_clip("t5xxl_fp16.safetensors", "Long-ViT-L-14-BEST-GmP-smooth-ft.safetensors", "clip_g.safetensors")[0]
    unet1, clip1 = LoraLoader.load_lora(unet, clip, "FLUX.1-Turbo-Alpha.safetensors", 1.00, 1.00)
    unet2, clip2 = LoraLoader.load_lora(unet1, clip1, "openflux1-v0.1.0-fast-lora.safetensors", 0.33, 0.33)
    unet3, clip3 = LoraLoader.load_lora(unet2, clip2, "xlabs_flux_realism_lora_comfui.safetensors", 0.70, 0.70)
    unet4, clip4 = LoraLoader.load_lora(unet3, clip3, "detailed_v2_flux_ntc.safetensors", 0.70, 0.70)
    vae = VAELoader.load_vae("ae.sft")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']
    max_shift = values['max_shift']
    base_shift = values['base_shift']
    width = values['width']
    height = values['height']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    positive = CLIPTextEncode.encode(clip4, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip4, negative_prompt)[0]
    unet_flux = ModelSamplingFlux.patch(unet4, max_shift, base_shift, width, height)[0]
    noise = RandomNoise.get_noise(seed)[0]
    guider = AdaptiveGuidance.get_guider(unet_flux, positive, negative, 1.0, cfg, uncond_zero_scale=0.0, cfg_start_pct=0.0)[0]
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(unet_flux, scheduler, steps, 1.0)[0]
    latent_image = EmptyLatentImage.generate(width, height)[0]
    sampler = DetailDaemonSamplerNode.go(sampler=sampler, detail_amount=0.2, start=0.2, end=0.9, bias=1.0, exponent=1.0, start_offset=0.0, end_offset=0.0, fade=0.0, smooth=False, cfg_scale_override=0.0)[0]
    samples, _ = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(f"/content/flux.1-dev-distill-{seed}-tost.png")

    result = f"/content/flux.1-dev-distill-{seed}-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})