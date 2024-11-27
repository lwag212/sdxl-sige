import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig
from PIL import Image
from torch import autocast

from sgm.util import append_dims, default
from sgm.modules.diffusionmodules.util import extract_into_tensor
from sige.utils import dilate_mask, downsample_mask
from sgm.modules.diffusionmodules.sige_openaimodel import SIGEUNetModel
from sgm.models.sige_autoencoder import SIGEAutoencoderKL
from sige.nn import SIGEModel


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [0, 1]

        Returns:
            same as input but watermarked
        """
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange(
            (255 * image).detach().cpu(), "n b c h w -> (n b) h w c"
        ).numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(
            rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)
        ).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        return image


# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({x.input_key for x in conditioner.embedders})


def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watermark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        print("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        print(f"sigmas after pruning: ", sigmas)
        return sigmas


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    batch2model_input: Optional[List] = None,
    return_latents=False,
    filter=None,
    device="cuda",
):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    with torch.no_grad():
        with autocast(device) as precision_scope:
            with model.ema_scope():
                num_samples = [num_samples]
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                )
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, batch[key].shape)
                    elif isinstance(batch[key], list):
                        print(key, [len(l) for l in batch[key]])
                    else:
                        print(key, batch[key])
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to(device), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to(device)

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples


def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_input_image_tensor(image: Image.Image, device="cuda"):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(
        lambda x: x - x % 64, (w, h)
    )  # resize to integer multiple of 64
    image = image.resize((width, height))
    image_array = np.array(image.convert("RGB"))
    image_array = image_array[None].transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(image_array).to(dtype=torch.float32) / 127.5 - 1.0
    return image_tensor.to(device)


def do_img2img(
    img,
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: float = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    device="cuda",
):
    with torch.no_grad():
        with autocast(device) as precision_scope:
            with model.ema_scope():
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [num_samples],
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                
                if isinstance(model.first_stage_model, SIGEAutoencoderKL):
                    assert isinstance(model.first_stage_model.encoder, SIGEModel)
                    model.first_stage_model.encoder.set_mode("full")
                z = model.encode_first_stage(img)

                noise = torch.randn_like(z)
                sigmas = sampler.discretization(sampler.num_steps)
                sigma = sigmas[0].to(z.device)

                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(z.shape[0], device=z.device), z.ndim
                    )
                noised_z = z + noise * append_dims(sigma, z.ndim)
                noised_z = noised_z / torch.sqrt(
                    1.0 + sigmas[0] ** 2.0
                )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)

                samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples

def do_inpaint(
    img,
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: float = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    device="cuda",
    mask=None,
    conv_masks=None,
    shape=None,
):
    with torch.no_grad():
        with autocast(device) as precision_scope:
            with model.ema_scope():
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [num_samples],
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                
                if isinstance(model.first_stage_model, SIGEAutoencoderKL):
                    assert isinstance(model.first_stage_model.encoder, SIGEModel)
                    model.first_stage_model.encoder.set_mode("full")
                z = model.encode_first_stage(img)

                noise = torch.randn_like(z)
                sigmas = sampler.discretization(sampler.num_steps)
                sigma = sigmas[0].to(z.device)

                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(z.shape[0], device=z.device), z.ndim
                    )
                noised_z = z + noise * append_dims(sigma, z.ndim)
                noised_z = noised_z / torch.sqrt(
                    1.0 + sigmas[0] ** 2.0
                )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

                x0 = z
                # x0 = noised_z
                def denoiser(x, sigma, c):
                    # if mask is not None:
                    #     x = x0 * mask + (1.0 - mask) * x
                    #     if isinstance(model.model.diffusion_model, SIGEUNetModel):
                    #         model.model.diffusion_model.set_mode("full")
                    #         model.denoiser(model.model, x0, sigma, c)
                    #         model.model.diffusion_model.set_mode("sparse")
                    #         model.model.diffusion_model.set_masks(conv_masks)

                    return model.denoiser(model.model, x, sigma, c)
                
                def set_mode_masks(mode, set_masks=False):
                    model.model.diffusion_model.set_mode(mode)
                    if set_masks: model.model.diffusion_model.set_masks(conv_masks)
                
                def apply_mask(x, x0):
                    return x0 * mask + (1.0 - mask) * x
                

                samples_z = sampler.inpaint_call(denoiser, set_mode_masks, noised_z, z, cond=c, uc=uc, 
                                                 apply_mask=apply_mask, is_sige=isinstance(model.model.diffusion_model, SIGEUNetModel),
                                                )

                if isinstance(model.first_stage_model, SIGEAutoencoderKL):
                    assert isinstance(model.first_stage_model.decoder, SIGEModel)
                    model.first_stage_model.decoder.set_mode("full")
                    model.decode_first_stage(z)
                    model.first_stage_model.decoder.set_masks(conv_masks)
                    model.first_stage_model.decoder.set_mode("sparse")
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples

def do_sdedit(
    edited_img,
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: float = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    device="cuda",
    init_img=None,
    masks=None,
    is_sige_model=False,
    difference_mask=None,
):
    with torch.no_grad():
        with autocast(device) as precision_scope:
            with model.ema_scope():
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [num_samples],
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                
                if isinstance(model.first_stage_model, SIGEAutoencoderKL):
                    assert isinstance(model.first_stage_model.encoder, SIGEModel)
                    assert init_img is not None, "Must provide an initial image for SIGE model"
                    model.first_stage_model.encoder.set_mode("full")
                    init_latent = model.encode_first_stage(init_img)
                    model.first_stage_model.encoder.set_mode("sparse")
                    model.first_stage_model.encoder.set_masks(masks)
                    edited_latent = model.encode_first_stage(edited_img)
                else:
                    init_latent = None
                    edited_latent = model.encode_first_stage(edited_img)
                # z = model.encode_first_stage(img)
                t_enc = int(value_dict['img2img_strength'] * value_dict['steps'])
                print(f"target t_enc is {t_enc} steps")

                noise = torch.randn_like(edited_latent)
                sigmas = sampler.discretization(sampler.num_steps)
                sigma = sigmas[0].to(edited_latent.device)
                # sigma = sigmas[t_enc - 1].to(edited_latent.device)

                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(edited_latent.shape[0], device=edited_latent.device), edited_latent.ndim
                    )
                noised_edited = edited_latent + noise * append_dims(sigma, edited_latent.ndim)
                z_enc_edited = noised_edited / torch.sqrt(
                    1.0 + sigma ** 2.0
                )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

                # Used for decoding
                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)
                
                def set_mode_masks(mode, set_masks=False):
                    model.model.diffusion_model.set_mode(mode)
                    if set_masks: model.model.diffusion_model.set_masks(masks)
                
                if is_sige_model:
                    noised_init = init_latent + noise * append_dims(sigma, init_latent.ndim)
                    z_enc_init = noised_init / torch.sqrt(
                        1.0 + sigma ** 2.0
                    )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
                    
                    samples_init, samples_edited = sampler.sige_call(denoiser, set_mode_masks, 
                                                                     z_enc_edited, z_enc_init, cond=c, uc=uc, is_sige=True)
                    # # samples = samples_edited
                else:
                    samples_init = None
                    samples_edited = sampler(denoiser, z_enc_edited, cond=c, uc=uc)
                    # samples = sampler.decode(
                    #     z_enc_edited, c, t_enc, unconditional_guidance_scale=args.scale, unconditional_conditioning=uc
                    # )

                if isinstance(model.first_stage_model, SIGEAutoencoderKL):
                    difference_mask = dilate_mask(difference_mask, 40)
                    masks = downsample_mask(difference_mask, min_res=(4, 4), dilation=0)
                    assert isinstance(model.first_stage_model.decoder, SIGEModel)
                    model.first_stage_model.decoder.set_mode("full")
                    model.decode_first_stage(samples_init)
                    model.first_stage_model.decoder.set_masks(masks)
                    model.first_stage_model.decoder.set_mode("sparse")
                # samples = model.decode_first_stage(samples)

                samples_x = model.decode_first_stage(samples_edited)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_edited
                return samples