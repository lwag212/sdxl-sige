import argparse

import numpy as np
import torch
from einops import repeat

# from sgm.models.sige_autoencoder import SIGEAutoencoderKL
# from sdxl.sgm.modules.diffusionmodules.sige_openaimodel import SIGEUNetModel
# from sige.nn import SIGEModel
from sige.utils import downsample_mask
from utils import load_img
from .base_runner import BaseRunner
from sgm.inference.api import *
from sgm.inference.helpers import *


class InpaintingRunner(BaseRunner):
    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super(InpaintingRunner, InpaintingRunner).modify_commandline_options(parser)
        parser.add_argument("--H", type=int, default=512)
        parser.add_argument("--W", type=int, default=512)
        parser.add_argument("--mask_path", type=str, required=True)
        return parser

    def __init__(self, args):
        super().__init__(args)
        assert args.init_img is not None, "Must provide an initial image for inpainting"

    def generate(self):
        # Setup
        args = self.args
        device = self.device

        # Works better with lower image strength
        params = SamplingParams()
        params.img2img_strength = .8
        params.width = args.W
        params.height = args.H

        # Generate the masks
        shape = (args.C, args.H // args.f, args.W // args.f)
        mask = np.load(args.mask_path)
        mask = torch.from_numpy(mask).to(device)
        masks = downsample_mask(mask, min_res=8, dilation=1)
        mask = 1 - masks[tuple(shape[1:])][None, None].float()
        conv_masks=masks

        # Run diffuser
        samples = self.model.inpaint(
          params = params,
          image = load_img(args.init_img).to(device),
        #   image = repeat(load_img(args.init_img).to(device), "1 ... -> b ...", b=1),
          prompt = args.prompt,
          mask=mask,
          conv_masks=conv_masks,
          shape=shape,
        )

        self.save_samples(samples)
        # args = self.args
        # model = self.model
        # sampler = self.sampler
        # device = self.device

        # # Get the batch to pass into conditioner
        # value_dict = asdict(SamplingParams())
        # value_dict["prompt"] = args.prompt
        # value_dict["negative_prompt"] = args.negative_prompt
        # value_dict["target_width"] = args.W
        # value_dict["target_height"] = args.H
        # num_samples = 1
        # batch, batch_uc = get_batch(
        #     get_unique_embedder_keys_from_conditioner(model.conditioner),
        #     value_dict,
        #     [num_samples],
        # )

        # # Condition
        # prompts = [args.prompt]
        # # if args.scale != 1.0:
        # #     uc = model.get_learned_conditioning([""])
        # # c = model.get_learned_conditioning(prompts)
        # c, uc = model.conditioner.get_unconditional_conditioning(
        #     batch,
        #     batch_uc=batch_uc,
        #     force_uc_zero_embeddings = [""],
        #     force_cond_zero_embeddings = prompts
        # )
        # for k in c:
        #     c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))
        
        # init_img = load_img(args.init_img).to(device)
        # init_img = repeat(init_img, "1 ... -> b ...", b=1)

        # if isinstance(model.model.first_stage_model, SIGEAutoencoderKL):
        #     assert isinstance(model.model.first_stage_model.encoder, SIGEModel)
        #     model.model.first_stage_model.encoder.set_mode("full")
        # # init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_img))
        # z = model.encode_first_stage(init_img)

        # del model.first_stage_model.encoder
        # model.first_stage_model.encoder = None
        # torch.cuda.empty_cache()

        # shape = (args.C, args.H // args.f, args.W // args.f)

        # mask = np.load(args.mask_path)
        # mask = torch.from_numpy(mask).to(device)
        # masks = downsample_mask(mask, min_res=8, dilation=1)

        # noise = torch.randn_like(z)
        # sigmas = sampler.discretization(sampler.num_steps)
        # sigma = sigmas[0].to(z.device)

        # noised_z = z + noise * append_dims(sigma, z.ndim)
        # noised_z = noised_z / torch.sqrt(
        #     1.0 + sigmas[0] ** 2.0
        # )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        # x0 = self.guider.prepare_inputs(z, sigma, c, uc)[0]
        # def denoiser(x, sigma, c):
        #     x = x0 * mask + (1.0 - mask) * x
        #     if isinstance(self.model.model.diffusion_model, SIGEUNetModel):
        #         assert masks is not None

        #         model.model.diffusion_model.set_mode("full")
        #         model.denoiser(model.model, x0, sigma, c)

        #         model.model.diffusion_model.set_mode("sparse")
        #         model.model.diffusion_model.set_masks(masks)

        #     return model.denoiser(model.model, x, sigma, c)

        # samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)

        # # samples, _ = sampler.sample(
        # #     S=args.ddim_steps,
        # #     conditioning=c,
        # #     batch_size=1,
        # #     shape=shape,
        # #     verbose=False,
        # #     unconditional_guidance_scale=args.scale,
        # #     unconditional_conditioning=uc,
        # #     eta=args.ddim_eta,
        # #     x_T=None,
        # #     mask=1 - masks[tuple(shape[1:])][None, None].float(),
        # #     x0=init_latent,
        # #     conv_masks=masks,
        # # )
        # # if isinstance(model.first_stage_model, SIGEAutoencoderKL):
        # #     assert isinstance(model.first_stage_model.decoder, SIGEModel)
        # #     model.first_stage_model.decoder.set_mode("full")
        # #     model.decode_first_stage(init_latent)
        # #     model.first_stage_model.decoder.set_masks(masks)
        # #     model.first_stage_model.decoder.set_mode("sparse")
        # if isinstance(model.first_stage_model, SIGEAutoencoderKL):
        #     assert isinstance(model.first_stage_model.decoder, SIGEModel)
        #     model.first_stage_model.decoder.set_mode("full")
        #     model.decode_first_stage(z)
        #     model.first_stage_model.decoder.set_masks(masks)
        #     model.first_stage_model.decoder.set_mode("sparse")
        # samples_x = model.decode_first_stage(samples_z)
        # samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        # # samples = model.decode_first_stage(samples)
        # self.save_samples(samples)
