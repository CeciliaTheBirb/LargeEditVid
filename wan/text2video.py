# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from contextlib import contextmanager
from functools import partial
import torch.nn.functional as F_torch
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import numpy as np
from typing import Optional, Literal
import itertools
from einops import rearrange
from sam2.build_sam import build_sam2_video_predictor 
from pathlib import Path
from .utils.utils import cache_video
from decord import VideoReader
from ultralytics import YOLOWorld
import peft
from torch import nn
sys.path.append('path/to/your/RAFT')
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from torchvision.transforms import Resize

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        
        # condition
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)
        
        
        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        #self.motion_model = self._init_motion_model()
    
    def _init_motion_model(self):
        import argparse
        args = argparse.Namespace(
            model='/xuqianxun/my_models/FlowDirector/RAFT/raft-things.pth',
            small=False, mixed_precision=False, alternate_corr=False)
        
        motion_model = torch.nn.DataParallel(RAFT(args))
        motion_model.load_state_dict(torch.load(args.model))
        motion_model = motion_model.module                       # unwrap DP
        motion_model.eval()                                      # freeze BN / dropout
        for p in motion_model.parameters():
            p.requires_grad = False
        motion_model.to(self.device)                             # move once
        return motion_model

    def compute_motion(self, x):
        if x.ndim == 4:                                  # (T, H, W, C)
            x = rearrange(x, 't h w c -> 1 c t h w')     # add batch & move channels

        assert len(x.shape) == 5, "Input should have len = 5, represent (b, c, t, w, h)"
        b = x.shape[0]
        t = x.shape[2]
        with torch.no_grad():
            #x = rearrange(x, 'b c t w h -> (b t) c w h', b=b, t=t)
            #x = denomalizing_img(x)
            #if(self.cfg.MODEL.VISUAL_MODEL == "sapiens"):
              #x = Resize(224)(x)
            #x = rearrange(x, '(b t) c w h -> b t c w h', b=b, t=t)
            #if(self.cfg.MODEL.MOTION_MODEL == "opencv"):
               #ret = extract_optical_flow_sparse_to_dense_video(x)
               #return ret
            # print(f'b = {b}, t = {t}')
            # flow_embs_batches = []
            flow_embs = []
            for index in range(t-1): #loop over time
                image1 = x[:,:,index]
                image2 = x[:,:,index+1]
                # print(image1.shape)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = self.motion_model(image1, image2, iters=1, test_mode=True)
                #if(self.cfg.MODEL.VISUAL_MODEL == "sapiens"):
                    #flow_up = Resize(1024)(flow_up)
                flow_embs.append(flow_up)
            flow_embs = torch.stack(flow_embs)
            ret = rearrange(flow_embs, 't b c w h -> b t c w h') #torch.stack(flow_embs)
            print(f'flow shape {ret.shape}')
            ret = ret.squeeze(0)
        return ret

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond, _ = self.model(
                    latent_model_input, t=timestep, **arg_c)
                noise_pred_uncond, _ = self.model(
                    latent_model_input, t=timestep, **arg_null)
                
                noise_pred_cond, noise_pred_uncond = noise_pred_cond[0], noise_pred_uncond[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def pick_point_from_yolo(self,frame_rgb, yolo_model, prompt: str | None = None, conf=0.25):
        """
        Detect objects in `frame_rgb` with YOLO and return (cx, cy)
        of the largest matching box.

        Args
        ----
        frame_rgb : np.ndarray   # shape (H,W,3), RGB uint8
        yolo_model: ultralytics.YOLO | YOLOWorld  # already on the right device
        prompt    : str | None   # if given, use `model.set_classes([prompt])`
        conf      : float        # confidence threshold

        Returns
        -------
        (cx, cy) : tuple[int,int]   # centre of the chosen box
        """
        # 1) Set open-vocabulary classes if a prompt is provided
        preds = yolo_model.predict(frame_rgb, conf=conf, verbose=False)[0]

        if len(preds.boxes) == 0:
            raise RuntimeError("YOLO found no objects in the frame")

        # 2. Filter boxes by class name
        matched_idx = [
            i for i in range(len(preds.boxes))
            if yolo_model.names[int(preds.boxes.cls[i])] == prompt
        ]
        if not matched_idx:
            raise RuntimeError(f"No '{prompt}' boxes found in the frame")

        # 3. Pick the largest area among matches
        boxes = preds.boxes.xyxy[matched_idx].cpu().numpy()  # (M,4)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best  = boxes[areas.argmax()]                        # (x1,y1,x2,y2)

        # 4. Centre point
        x1, y1, x2, y2 = best
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        return best


    def sam2_masks_from_point(self, frames_rgb, box_xyxy, sam2_predictor):
        """
        1) Pick the center of a YOLO person-box (in original image coords)
        2) Scale to SAM2’s 1024×1024 crop
        3) Add as a single positive point
        4) Propagate & resize masks back to original size
        """
        T, H, W, _ = frames_rgb.shape

        # — initialize state with your native-resolution frames —
        inference_state = sam2_predictor.init_state(video_path='/xuqianxun/my_models/FlowDirector/video_list/dog')
        sam2_predictor.reset_state(inference_state)
        obj_id = 1  # SAM2 uses a single object ID for all frames
        # — seed with a box on frame 0 —
        box = box_xyxy.astype(np.float32)
        print(box)
        _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=box,
        )
        
        masks = np.zeros((T, H, W), dtype=np.uint8)
        for f_idx, obj_ids, logits in sam2_predictor.propagate_in_video(inference_state):
            if obj_id not in obj_ids:
                continue
            loc = obj_ids.index(obj_id)
            mask_np = (logits[loc] > 0).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(f'out_masks/raw_{f_idx}.png',mask_np*255)
            # logits[loc] is already the mask at your original (H, W)
            #mask_np = (logits[loc] > 0).detach().cpu().numpy().astype(np.uint8)
            mask_raw = (logits[loc] > 0).float()#.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
            if mask_raw.dim() == 2:                       # [h, w]  ➜ add N and C
                mask_raw = mask_raw.unsqueeze(0).unsqueeze(0)      # [1,1,h,w]
            elif mask_raw.dim() == 3:                     # [1, h, w] ➜ add N only
                mask_raw = mask_raw.unsqueeze(0)
            mask_resized = F.interpolate(mask_raw, size=(H, W), mode="nearest")[0, 0]
            masks[f_idx] = mask_resized.byte().cpu().numpy()

        return masks

    def load_video_frames(self, video_path, size=(832, 480)):
        r"""
        Load video frames from the given path and preprocess them.

        Args:
            video_path (str): Path to the video file.
            size (tuple[`int`], *optional*, defaults to (1280,720)): Target resolution for resizing frames.

        Returns:
            torch.Tensor: Tensor of video frames with shape (frame_num, C, H, W).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to target size
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            # Convert to tensor and normalize to [-1, 1]
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1.0
            # Convert to tensor and normailize to [0, 1]
            # frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255
            frames.append(frame)

        cap.release()
        if not frames:
            raise ValueError(f"No frames found in video: {video_path}")

        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3).to(self.device)
        #gt_motion= self.compute_motion(frames_tensor)
        #print("Frames tensor shape:", frames_tensor.shape)
        #mask_noise= torch.randn_like(frames_tensor)
        #frames_tensor= frames_tensor * (1-mask) + mask_noise*mask  # Apply mask to source latent
        latents = self.vae.video_encode(frames_tensor)
        
        
        return latents, frames_tensor#, gt_motion # [C, F, H, W]


    def find_subtokens_range(self, source_tokens, target_tokens):
        """
        查找 target_tokens 在 source_tokens 中的位置，不包括 '
        返回起始索引和结束索引（类似 slice），找不到返回 None
        """
        valid_len = len(source_tokens)
        # 在 source 的有效范围内滑动窗口匹配 target
        for i in range(valid_len - len(target_tokens) + 1):
            if source_tokens[i:i + len(target_tokens)] == target_tokens:
                # return (i, i + len(target_tokens)-1)
                return list(range(*(i, i + len(target_tokens))))
            

        return None


    def create_binary_mask( # Renamed slightly for clarity
        self,
        attn_map: torch.Tensor,
        n: int, 
        pooling_mode: Literal['max', 'avg'] = 'max',
        threshold: Optional[float] = 0.5,
        threshold_method: Literal['fixed', 'otsu'] = 'fixed',
        normalize_per_slice=False
    ) -> torch.Tensor:

        # --- Preparation ---
        original_shape = attn_map.shape
        C, T, H, W = original_shape
        device = attn_map.device

        map_batched = attn_map.reshape(C * T, 1, H, W)

        padding = (n - 1) // 2

        if pooling_mode == 'max':
            smoothed_map_batched = torch.nn.functional.max_pool2d(map_batched, kernel_size=n, stride=1, padding=padding)
        elif pooling_mode == 'avg':
            smoothed_map_batched = torch.nn.functional.avg_pool2d(map_batched, kernel_size=n, stride=1, padding=padding)

        smoothed_map = smoothed_map_batched.squeeze(1).view(C, T, H, W)
        # smoothed_map has shape (C, T, H, W)

        map_to_binarize = smoothed_map_batched # Start with the smoothed map
        if normalize_per_slice:

            flat_map = map_to_binarize.view(C * T, -1)

            # Calculate min and max per slice (image in the batch C*T)
            min_vals = torch.min(flat_map, dim=1, keepdim=True)[0]
            max_vals = torch.max(flat_map, dim=1, keepdim=True)[0]

            # Calculate range, handle the case where min == max (flat slice)
            range_vals = max_vals - min_vals

            range_vals = torch.where(range_vals == 0,
                                    torch.tensor(1.0, device=device, dtype=map_to_binarize.dtype),
                                    range_vals)

            min_vals_b = min_vals.view(C * T, 1, 1, 1)
            range_vals_b = range_vals.view(C * T, 1, 1, 1)

            normalized_map_batched = (map_to_binarize - min_vals_b) / range_vals_b

            map_to_binarize = torch.clamp(normalized_map_batched, 0.0, 1.0)

        map_to_binarize = map_to_binarize.squeeze(1).view(C, T, H, W)
        binary_mask = torch.zeros_like(map_to_binarize, dtype=torch.bool, device=device)
        
        if threshold_method == 'fixed':
            if normalize_per_slice: 
                binary_mask = map_to_binarize > torch.mean(map_to_binarize).item()
            else:
                binary_mask = map_to_binarize > threshold

        return binary_mask.to(device=device, dtype=torch.float32)          
   

    def soften_mask_edges(
        self,
        binary_mask: torch.Tensor,
        decay_factor: float = 0.5
    ) -> torch.Tensor:
        """
        Softens the edges of a binary mask by assigning values to background pixels (0)
        based on their distance to the nearest foreground pixel (1).

        Pixels originally equal to 1 remain 1.
        Pixels originally equal to 0 get a value exp(-decay_factor * distance),
        where distance is the Euclidean distance to the nearest 1.
        This means pixels closer to the original mask edge get values closer to 1,
        and pixels farther away get values closer to 0.

        Args:
            binary_mask (torch.Tensor): The input binary mask. Expected to be a 4D
                                        tensor (C, T, H, W) with values 0.0 or 1.0.
            decay_factor (float): Controls how quickly the softened value decays
                                with distance. A larger value means a faster decay
                                (sharper transition near the edge), a smaller value
                                means a slower decay (softer, more spread-out transition).
                                Defaults to 0.1.

        Returns:
            torch.Tensor: A mask tensor with the same dimensions as the input,
                        where original mask areas are 1.0 and background areas
                        have softened values based on distance. Output is float32.

        Raises:
            TypeError: If binary_mask is not a PyTorch Tensor.
            ValueError: If binary_mask is not 4D.
            ImportError: If scipy is not installed.
        """
        from scipy.ndimage import distance_transform_edt
        # --- Input Validation ---
        if not isinstance(binary_mask, torch.Tensor):
            raise TypeError("binary_mask must be a PyTorch Tensor.")
        if binary_mask.ndim != 4:
            raise ValueError(f"Input binary_mask must be 4D (C, T, H, W), got {binary_mask.ndim}D")
        if not decay_factor > 0:
            raise ValueError("decay_factor must be positive.")

        # --- Preparation ---
        original_shape = binary_mask.shape
        C, T, H, W = original_shape
        device = binary_mask.device
        dtype = torch.float32 # Ensure output is float

        # Create an output tensor initialized with zeros
        softened_mask = torch.zeros_like(binary_mask, dtype=dtype, device=device)

        # --- Process each slice (C, T) independently ---
        for c in range(C):
            for t in range(T):
                # Extract the 2D slice
                mask_slice = binary_mask[c, t, :, :]

                # Move to CPU and convert to NumPy boolean array for SciPy
                # distance_transform_edt expects background (0) to be True
                # and foreground (1) to be False.
                inverted_mask_slice_np = (mask_slice == 0).cpu().numpy()

                # Compute Euclidean Distance Transform
                # distance_map_np contains the distance from each True pixel (background)
                # to the nearest False pixel (foreground/mask).
                # Pixels that were originally part of the mask (False in inverted) will have distance 0.
                distance_map_np = distance_transform_edt(inverted_mask_slice_np)

                # Convert distances to softened values (0 to 1) using exponential decay
                # exp(-k * distance). Larger distance -> smaller value.
                # Distance 0 (original mask) -> exp(0) = 1.
                # Add a small epsilon to distance before applying exp if you want to strictly avoid 1.0 in non-mask areas
                # but exp(-k*dist) naturally handles this.
                softened_values_np = np.exp(-decay_factor * distance_map_np)

                # Convert back to PyTorch tensor and move to the original device
                softened_values_slice = torch.from_numpy(softened_values_np).to(device=device, dtype=dtype)

                # --- Combine original mask and softened background ---
                # Use torch.where for clarity and efficiency:
                # Where the original mask was 1, keep 1.0.
                # Where the original mask was 0, use the calculated softened value.
                final_slice = torch.where(
                    mask_slice.bool(), # Condition: True where original mask was 1
                    torch.tensor(1.0, device=device, dtype=dtype), # Value if True
                    softened_values_slice # Value if False (use calculated softened value)
                )

                # Place the processed slice into the output tensor
                softened_mask[c, t, :, :] = final_slice

        return softened_mask                

    
    def generate_conflict_map( # Renamed slightly for clarity
        self,
        vsrc: torch.Tensor,
        vtar: torch.Tensor,
        normalize: bool = True, # Setting default to True as it simplifies tuning 'k' later
        norm_type: int = 2,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Generates a conflict map indicating the magnitude of difference between
        two velocity fields (vsrc and vtar) and returns it as a 4D tensor.

        The conflict at each spatial-temporal location is defined as the L-p norm
        (default L2, Euclidean distance) of the difference vector between vsrc and
        vtar along the channel dimension. The resulting scalar map (F, H, W) is
        then expanded to match the input shape (C, F, H, W) by repeating the
        scalar value across the channel dimension.

        Args:
            vsrc (torch.Tensor): The source velocity field tensor. Expected shape:
                                (Channels, Frames, Height, Width).
            vtar (torch.Tensor): The target velocity field tensor. Must have the
                                same shape and device as vsrc.
            normalize (bool, optional): If True, normalize the underlying 3D conflict
                                        map to the range [0, 1] using min-max scaling
                                        before expanding it to 4D. Recommended for
                                        easier tuning of 'k' in downstream functions.
                                        Defaults to True.
            norm_type (int, optional): The order of the norm (p-norm). Default is 2 (L2 norm).
                                    Use 1 for L1 norm (Manhattan distance), etc.
            epsilon (float, optional): A small value added to the denominator during
                                    normalization to prevent division by zero if
                                    all conflict values are identical. Defaults to 1e-6.

        Returns:
            torch.Tensor: The conflict map tensor, expanded to 4D.
                        Shape: (Channels, Frames, Height, Width). The value is
                        uniform across the channel dimension for each (F,H,W).

        Raises:
            TypeError: If inputs are not PyTorch tensors.
            ValueError: If input tensors do not have the same shape or are not 4D.
        """
        # --- Input Validation ---
        if not isinstance(vsrc, torch.Tensor) or not isinstance(vtar, torch.Tensor):
            raise TypeError("Inputs vsrc and vtar must be PyTorch tensors.")

        if vsrc.ndim != 4 or vtar.ndim != 4:
            raise ValueError(f"Input tensors must be 4D (C, F, H, W), "
                            f"got shapes {vsrc.shape} and {vtar.shape}")

        if vsrc.shape != vtar.shape:
            raise ValueError(f"Input tensors vsrc and vtar must have the same shape, "
                            f"got {vsrc.shape} and {vtar.shape}")

        if vsrc.device != vtar.device:
            logging.warning(f"Input tensors are on different devices ({vsrc.device}, {vtar.device}). "
                            f"Proceeding with calculations, but ensure this is intended.")

        # --- Conflict Calculation ---
        vsrc_float = vsrc.float()
        vtar_float = vtar.float()
        num_channels = vsrc.shape[0]

        difference = vtar_float - vsrc_float

        # Calculate the norm along the channel dimension -> shape (F, H, W)
        conflict_map_3d = torch.norm(difference, p=norm_type, dim=0)

        # --- Optional Normalization (applied to the 3D map) ---
        if normalize:
            # Avoid normalization issues on empty tensors
            if conflict_map_3d.numel() > 0:
                min_val = torch.min(conflict_map_3d)
                max_val = torch.max(conflict_map_3d)
                denominator = max_val - min_val

                if denominator < epsilon:
                    logging.warning(f"Conflict map values are nearly constant ({min_val.item()} to {max_val.item()}). "
                                    f"Normalization results in a map of all zeros.")
                    conflict_map_3d = torch.zeros_like(conflict_map_3d)
                else:
                    conflict_map_3d = (conflict_map_3d - min_val) / denominator
            else:
                logging.warning("Conflict map has zero elements, skipping normalization.")


        # --- Expand to 4D ---
        # Add channel dim and expand
        conflict_map_4d = conflict_map_3d.unsqueeze(0).expand(num_channels, -1, -1, -1)

        return conflict_map_4d


    def compute_dynamic_source_mask( # Renamed slightly for clarity
        self,
        conflict_map_4d: torch.Tensor,
        k: float,
        function_type: str = 'exponential_squared',
        clamp_output: bool = True
        # warn_threshold removed as normalization is best done in the generating function
    ) -> torch.Tensor:
        """
        Computes the Dynamic Source Mask M(p, t) based on a 4D conflict map input.

        Applies a chosen function element-wise to the conflict map values to generate
        the mask. High conflict should result in a mask value near 0, while low
        conflict should result in a value near 1. Assumes the input conflict map
        may have redundant values across the channel dimension.

        Args:
            conflict_map_4d (torch.Tensor): A tensor representing the conflict C(p, t)
                                            between Vsrc and Vtar. Expected shape:
                                            (Channels, Frames, Height, Width).
                                            Normalizing this input (e.g., via
                                            generate_conflict_map_4d with normalize=True)
                                            is recommended for easier tuning of 'k'.
            k (float): A positive sensitivity parameter. Controls how aggressively
                    the mask value decreases as conflict increases.
            function_type (str, optional): Specifies the function f(C) used to map
                                        conflict C to the mask value M element-wise.
                                        Options:
                                        - 'exponential_squared': M = exp(-k * C^2)
                                        - 'exponential': M = exp(-k * C)
                                        - 'inverse': M = 1 / (1 + k * C)
                                        - 'inverse_squared': M = 1 / (1 + k * C^2)
                                        Defaults to 'exponential_squared'.
            clamp_output (bool, optional): If True, clamps the output mask values
                                        strictly to the range [0.0, 1.0].
                                        Defaults to True.

        Returns:
            torch.Tensor: The dynamic source mask tensor M(p, t).
                        Shape: (Channels, Frames, Height, Width).

        Raises:
            TypeError: If conflict_map_4d is not a PyTorch tensor.
            ValueError: If conflict_map_4d is not 4D, k is not positive, or an
                        invalid function_type is provided.
        """
        # --- Input Validation ---
        if not isinstance(conflict_map_4d, torch.Tensor):
            raise TypeError("Input conflict_map_4d must be a PyTorch tensor.")

        if conflict_map_4d.ndim != 4:
            raise ValueError(f"Input conflict_map_4d must be 4D (C, F, H, W), "
                            f"got {conflict_map_4d.ndim}D shape {conflict_map_4d.shape}")

        # num_channels = conflict_map_4d.shape[0] # Get C if needed elsewhere

        if not isinstance(k, (int, float)) or k <= 0:
            raise ValueError(f"Parameter k must be a positive number, got {k}")

        supported_functions = ['exponential_squared', 'exponential', 'inverse', 'inverse_squared']
        if function_type not in supported_functions:
            raise ValueError(f"Invalid function_type '{function_type}'. "
                            f"Supported types are: {supported_functions}")

        # --- Mask Calculation (Applied element-wise on 4D tensor) ---
        # Note: conflict_map_4d already includes the (potentially redundant) channel dim
        conflict_map_float = conflict_map_4d.float()
        k_float = float(k)

        if function_type == 'exponential_squared':
            # Applies exp(-k * C^2) element-wise
            mask_4d = torch.exp(-k_float * torch.pow(conflict_map_float, 2))
        elif function_type == 'exponential':
            # Applies exp(-k * C) element-wise
            mask_4d = torch.exp(-k_float * conflict_map_float)
        elif function_type == 'inverse':
            # Applies 1 / (1 + k * C) element-wise
            mask_4d = 1.0 / (1.0 + k_float * conflict_map_float)
        elif function_type == 'inverse_squared':
            # Applies 1 / (1 + k * C^2) element-wise
            mask_4d = 1.0 / (1.0 + k_float * torch.pow(conflict_map_float, 2))
        else:
            raise NotImplementedError(f"Function type {function_type} calculation not implemented.")

        # --- Optional Clamping ---
        if clamp_output:
            mask_4d = torch.clamp(mask_4d, min=0.0, max=1.0)

        # --- Return the 4D Mask ---
        # No expansion needed, it's already 4D
        return mask_4d
    
    def visualize_and_save_masks(self, v_bg_mask, v_tar_mask, vae_stride, patch_size, suffix=".mp4"):
            """
            v_bg_mask:   Tensor [C, F, H_lat, W_lat]  ⟵ your latent‐space mask
            v_tar_mask:  same shape
            vae_stride:  (stride_t, stride_h, stride_w)
            patch_size:  (p_t, p_h, p_w)
            """

            bg = v_bg_mask.mean(dim=0)  # [F, H_lat, W_lat]
            tar = v_tar_mask.mean(dim=0)

            # Step 2: add channel dim → [1, F, H_lat, W_lat] → then permute to [1, 1, F, H_lat, W_lat]
            bg = bg.unsqueeze(0).unsqueeze(0)   # [1, 1, F, H_lat, W_lat]
            tar = tar.unsqueeze(0).unsqueeze(0)

            stride_t, stride_h, stride_w = vae_stride
            F_lat, H_lat, W_lat = bg.shape[-3:]
            H = H_lat * stride_h
            W = W_lat * stride_w
            F_up = F_lat * stride_t

            # Step 3: Interpolate (upsample) in 3D (T x H x W)
            bg_up = F.interpolate(bg, size=(F_up, H, W), mode='trilinear', align_corners=False)
            tar_up = F.interpolate(tar, size=(F_up, H, W), mode='trilinear', align_corners=False)

            # Step 4: squeeze batch & channel → [F, H, W]
            bg_up = bg_up.squeeze(0).squeeze(0)
            tar_up = tar_up.squeeze(0).squeeze(0)

            # Step 5: turn into 3-channel RGB heatmaps → [3, F, H, W]
            bg_rgb = torch.stack([bg_up]*3, dim=0)
            tar_rgb = torch.stack([tar_up]*3, dim=0)

            # Step 6: add batch dim and clamp → [1, 3, F, H, W]
            bg_rgb = bg_rgb.clamp(0, 1).unsqueeze(0)
            tar_rgb = tar_rgb.clamp(0, 1).unsqueeze(0)

            # Step 7: Save to disk
            save_bg = cache_video(bg_rgb, save_file="tmp_src_mask.mp4", fps=12, suffix=".mp4")
            save_tar = cache_video(tar_rgb, save_file="tmp_tar_mask.mp4", fps=12, suffix=".mp4")

            print("Saved mask videos:", save_bg, save_tar)
                    
    def edit(self,
             target_prompt,
             size=(832, 480),
             frame_num=81,
             shift=5.0,
             sample_solver='unipc',
             sampling_steps=50,
             guide_scale=5.0,
             tar_guide_scale=10.0,
             n_prompt="",
             seed=-1,
             offload_model=True,
             source_video_path=None,
             source_prompt=None,
             nmax_step=50,
             nmin_step=0,     
             n_avg=5,
             worse_avg=3,
             omega=3,
             source_words=None,
             target_words=None,
             window_size=11,
             decay_factor=0.5,
             tmd_window_size=11,
             tmd_stride=8
             ):
        '''from ultralytics.nn.tasks import WorldModel
        sam2_checkpoint = "/ssdwork/xuqianxun/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        sam2 = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        _orig_torch_load = torch.load

        # 2) wrap it so weights_only=False is the default
        def _torch_load_force_full(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_force_full    # <-- monkey-patch
        # Option 1: using `add_safe_globals` globally
        torch.serialization.add_safe_globals([WorldModel]) 
        yolo=YOLOWorld('yolov8l.pt').to(self.device)
        vr = VideoReader(source_video_path)
        frames = vr.get_batch(range(len(vr))).asnumpy()        # (T,H,W,3) BGR
        frames_rgb = frames[..., ::-1].copy()                  # RGB
        T, H, W, _ = frames_rgb.shape

        box = self.pick_point_from_yolo(frames_rgb[0], yolo, prompt='dog',conf=0.3)
        #print("Chosen point:", point_xy)

        # 5. run SAM2 propagation
        masks = self.sam2_masks_from_point(frames_rgb, box, sam2)
        out_dir = Path('out_masks')
        out_dir.mkdir(exist_ok=True, parents=True)
        # 6. save masks + overlay video
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        overlay = cv2.VideoWriter(str(out_dir/"overlay.mp4"),
                                fourcc, vr.get_avg_fps(), (W, H))
        for t in range(T):
            cv2.imwrite(str(out_dir/f"{t:05d}.png"), masks[t]*255)
            frame_bgr = frames_rgb[t][..., ::-1].copy()
            red = np.zeros_like(frame_bgr); red[...,2] = 255
            alpha = 0.45
            frame_bgr = np.where(masks[t][...,None].astype(bool),
                                cv2.addWeighted(red,alpha,frame_bgr,1-alpha,0),
                                frame_bgr)
            overlay.write(frame_bgr)
        overlay.release()
        print(f"✅ Masks saved to {out_dir} (PNG + overlay.mp4)")
'''
        # preprocess
        F = frame_num
        W, H = size
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1], size[0] // self.vae_stride[2])
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) * 
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        # seed_g = torch.Generator(device=self.device)
        torch.manual_seed(seed)

        # 加载源视频潜在表示和参考图像
        #mask=torch.from_numpy(masks).unsqueeze(0).to(self.device)
        #print(f"mask shape: {mask.shape}")
        x_src, org= self.load_video_frames(source_video_path, size=size)
        #x_src, gt_motion = self.load_video_frames(source_video_path, size=size)
        C_latent, F_latent, H_latent, W_latent = x_src.shape
        import torch.nn.functional as F_torch
        #gt_motion = F_torch.interpolate(gt_motion, size=(F_latent, H_latent, W_latent), mode='nearest')  # [1, 1, F, H_latent, W_latent]
        #mask_tensor = torch.from_numpy(masks).unsqueeze(0).unsqueeze(0).float()  # [1, 1, F, H, W]
        #mask_resized = F_torch.interpolate(mask_tensor, size=(F_latent, H_latent, W_latent), mode='nearest')  # [1, 1, F, H_latent, W_latent]
        #final_mask = mask_resized.squeeze(0).expand(C_latent, -1, -1, -1).to(self.device)
        
        #x_src = x_src * (1-final_mask)+mask_noise*final_mask  # Apply mask to source latent
        # Validate TMD parameters
        if tmd_window_size > F_latent:
            logging.warning(f"tmd_window_size ({tmd_window_size}) > latent frames ({F_latent}). Using full sequence as one window.")
            tmd_window_size = F_latent
            tmd_stride = F_latent
        elif tmd_stride <= 0:
             logging.warning(f"Invalid tmd_stride ({tmd_stride}). Setting stride to window size / 2.")
             tmd_stride = max(1, tmd_window_size // 2)
             
             
        # 计算提示词相对位置
        source_words_idx = None
        target_words_idx = None 
        if source_words:
            tk1 = self.text_encoder.tokenizer.tokenizer.tokenize(source_prompt, add_special_tokens=True)
            tk2 = self.text_encoder.tokenizer.tokenizer.tokenize(source_words, add_special_tokens=True)
            source_words_idx = self.find_subtokens_range(tk1[:-1], tk2[:-1])
            
        if target_words:
            tk1 = self.text_encoder.tokenizer.tokenizer.tokenize(target_prompt, add_special_tokens=True)
            tk2 = self.text_encoder.tokenizer.tokenizer.tokenize(target_words, add_special_tokens=True)
            target_words_idx = self.find_subtokens_range(tk1[:-1], tk2[:-1])

        # 编码源和目标文本提示
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_src = self.text_encoder([source_prompt], self.device)
            context_tar = self.text_encoder([target_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context_tar = self.text_encoder([target_prompt], torch.device('cpu'))
            context_src = self.text_encoder([source_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context_src = [t.to(self.device) for t in context_src]
            context_tar = [t.to(self.device) for t in context_tar]
            context_null = [t.to(self.device) for t in context_null]

        arg_src_c = {'context': context_src, 'seq_len': seq_len, 'words_indices': source_words_idx, 'block_id': 18, 'type': 'src'}
        arg_tar_c = {'context': context_tar, 'seq_len': seq_len, 'words_indices': target_words_idx, 'block_id': 18, 'type': 'tar'}
        arg_unc = {'context': context_null, 'seq_len': seq_len}

        
        # 初始化编辑路径
        zt_edit = x_src.clone() # [16, 21, 60, 104]: C x Frames x H x W 
        conflict_mask = torch.ones_like(x_src)

        # 设置采样调度器
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False, solver_order=2)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False, solver_order=1)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        self.model.to(self.device)
        self.model.train()  # Now this is your LoRA-wrapped model

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5) 
        #timesteps = timesteps.flip(0)
        # 编辑过程
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            fwd_noise0 = torch.randn_like(zt_edit[:, 0:tmd_window_size, :, :], device=self.device)
            for index, t in enumerate(tqdm(timesteps)):
                  # torch tensor reverse, shape: [steps]
                t_next = timesteps[timesteps.tolist().index(t) + 1] if t > timesteps[-1] else 0
                '''                if index < len(timesteps) - 1:
                    t_i    = timesteps[index].item() / 1000.0
                    t_im1  = timesteps[index + 1].item() / 1000.0
                else:
                    t_im1 = 0.0 #if needed'''
                arg_src_c["timestep"] = t
                arg_tar_c["timestep"] = t
                arg_unc["timestep"] = t
                timestep = torch.tensor([t], device=self.device)
                relative_index = nmax_step - (sampling_steps - index)
                v_list = []
                v_src_list = []
                
                if sampling_steps - (index + 1) >= nmax_step:
                    continue
                if sampling_steps - (index + 1) >= nmin_step:
                    t_i = t / 1000.0
                    t_im1 = t_next / 1000.0
                    v_delta_sum = torch.zeros_like(zt_edit)
                    v_src_sum = torch.zeros_like(zt_edit)
                    v_worse = torch.zeros_like(zt_edit)
                    v_mask = torch.zeros_like(zt_edit)
                    v_tar_mask = torch.zeros_like(zt_edit)
                    v_bg_mask = torch.zeros_like(zt_edit)
                    for time in range(n_avg):
                        # --- Temporal MultiDiffusion Logic ---
                        V_delta_accumulator = torch.zeros_like(zt_edit)
                        V_src_accumulator = torch.zeros_like(zt_edit)
                        window_counts = torch.zeros_like(zt_edit) # Use float for division later
                        window_mask_sum = torch.zeros_like(zt_edit)    
                        window_tar_mask_sum = torch.zeros_like(zt_edit)  
                        window_bg_mask_sum = torch.zeros_like(zt_edit)  
                        #fwd_noise = torch.randn_like(zt_edit[:, 0:tmd_window_size, :, :], device=self.device)
                        # for f_start in range(0, F_latent, tmd_stride):
                        if tmd_window_size >= F_latent:
                            window_starts = [0]
                        else:
                            window_starts = list(range(0, F_latent - tmd_window_size, tmd_stride))
                            last_possible_start = F_latent - tmd_window_size
                            if not window_starts or window_starts[-1] < last_possible_start:
                                if last_possible_start >= 0:
                                    window_starts.append(last_possible_start)

                        for f_start in window_starts:
                            f_end = F_latent if tmd_window_size >= F_latent else f_start + tmd_window_size

                            # --- Calculate V_delta within the window ---
                            # Extract window slices
                            zt_edit_w = zt_edit[:, f_start:f_end, :, :]
                            x_src_w = x_src[:, f_start:f_end, :, :]  

                            # 计算 zt_src
                            if index < 30:  # Use fwd_noise0 for the first 30 iterations
                                fwd_noise = torch.randn_like(zt_edit[:, 0:tmd_window_size, :, :], device=self.device)
                                zt_src = (1 - t_i) * x_src_w + t_i * fwd_noise
                            else:
                                #fwd_noise = torch.randn_like(zt_edit[:, 0:tmd_window_size, :, :], device=self.device)
                                zt_src = (1 - t_i) * x_src_w + t_i * fwd_noise0
                            zt_tar = zt_edit_w + zt_src - x_src_w
                                          
                            #if index<30:     
                                #noise_pred_src, src_attn_map = self.model([zt_src], t=timestep, **arg_tar_c)
                            # 计算源和目标噪声预测
                            #else:
                            noise_pred_src, src_attn_map = self.model([zt_src], t=timestep, **arg_src_c)
                            noise_pred_tar, tar_attn_map = self.model([zt_tar], t=timestep, **arg_tar_c)
                            noise_pred_src, noise_pred_tar = noise_pred_src[0], noise_pred_tar[0]
                            # uncond
                            noise_pred_uncond_src, _ = self.model([zt_src], t=timestep, **arg_unc)
                            noise_pred_uncond_tar, _ = self.model([zt_tar], t=timestep, **arg_unc)
                            noise_pred_uncond_src, noise_pred_uncond_tar = noise_pred_uncond_src[0], noise_pred_uncond_tar[0]
                            #noise_pred_tar_cfg, tar_attn_map_cfg = self.model([zt_tar], t=timestep, **arg_src_c)
                            #noise_pred_tar_cfg=noise_pred_tar_cfg[0]

                            # 计算引导后的噪声预测
                            noise_pred_src_guided = noise_pred_uncond_src + guide_scale * (noise_pred_src - noise_pred_uncond_src)
                            noise_pred_tar_guided = noise_pred_uncond_tar + tar_guide_scale * (noise_pred_tar - noise_pred_uncond_tar)
                            #noise_pred_tar_guided = noise_pred_uncond_tar + tar_guide_scale * (noise_pred_tar - noise_pred_tar_cfg)
                            #noise_pred_src_guided = noise_pred_src
                            sum_attn_mask = torch.zeros_like(zt_edit_w)
                            sum_tar_attn_mask = torch.zeros_like(zt_edit_w)
                            sum_src_attn_mask = torch.zeros_like(zt_edit_w)
                            restore_mask = torch.zeros_like(zt_edit_w)
                            
                            conflict_map = self.generate_conflict_map(noise_pred_src, noise_pred_tar_guided)
                            raw_mask = self.compute_dynamic_source_mask(conflict_map, 0.5)
                            clamped_mask = torch.clamp(raw_mask, min=0.0, max=1.0)
                            conflict_mask = 5.0 - 4.0 * clamped_mask
                            
                            E_p=zt_tar-t_i * noise_pred_tar_guided
                            E_q=zt_src-t_i * noise_pred_src_guided
                                                        
                            if src_attn_map is not None:
                                src_attn_mask = self.create_binary_mask(src_attn_map,
                                                                        n=window_size,
                                                                        pooling_mode='avg',
                                                                        threshold=torch.mean(src_attn_map).item(),
                                                                        threshold_method='fixed')
                                
                                sum_attn_mask += src_attn_mask
                                
                            if tar_attn_map is not None:
                                tar_attn_mask = self.create_binary_mask(tar_attn_map,
                                                                        n=window_size,
                                                                        pooling_mode='avg',
                                                                        threshold=torch.mean(tar_attn_map).item(),
                                                                        threshold_method='fixed')
                                
                                sum_attn_mask += tar_attn_mask
                                
                            sum_attn_mask = torch.clamp(sum_attn_mask, min=0.0, max=1.0)
                            sum_tar_attn_mask = torch.clamp(tar_attn_mask, min=0.0, max=1.0)
                            sum_src_attn_mask = torch.clamp(src_attn_mask, min=0.0, max=1.0)
                            restore_mask  = sum_src_attn_mask * (1.0 - sum_tar_attn_mask)
                            #if index<30: 
                                #V_delta = noise_pred_src_guided
                            #else:
                            #vel_src = (1 - t_i) * noise_pred_src_guided - t_i * x_src_w
                            #vel_tar = (1 - t_i) * noise_pred_tar_guided - t_i * zt_edit_w
                            #V_delta = vel_tar - vel_src
                            V_delta = noise_pred_tar_guided - noise_pred_src_guided
                            print(f'max difference: {torch.max(V_delta)}')

                            def cosine_similarity_4d(x, y):
                                """
                                Compute cosine similarity between two tensors of shape (C, F, H, W)
                                """
                                assert x.shape == y.shape, "Input shapes must match"

                                x_flat = x.view(-1)
                                y_flat = y.view(-1)

                                x_norm = F_torch.normalize(x_flat.unsqueeze(0), dim=1)
                                y_norm = F_torch.normalize(y_flat.unsqueeze(0), dim=1)

                                sim = (x_norm * y_norm).sum().item()
                                return sim

                            # Example usage
                            sim = cosine_similarity_4d(noise_pred_src_guided, noise_pred_tar_guided)
                            print(f"Cosine similarity: {sim:.4f}")
                            # Accumulate results
                            V_delta_accumulator[:, f_start:f_end, :, :] += V_delta
                            V_src_accumulator[:, f_start:f_end, :, :] += noise_pred_src_guided
                            window_counts[:, f_start:f_end, :, :] += 1.0
                            window_mask_sum[:, f_start:f_end, :, :] += sum_attn_mask
                            window_tar_mask_sum[:, f_start:f_end, :, :] += sum_tar_attn_mask
                            window_bg_mask_sum[:, f_start:f_end, :, :] += restore_mask

                        V_delta_final = V_delta_accumulator / torch.clamp(window_counts, min=1.0) # Avoid division by zero
                        V_src_final = V_src_accumulator / torch.clamp(window_counts, min=1.0) # Avoid division by zero
                        v_delta_sum += V_delta_final
                        v_src_sum += V_src_final
                        v_list.append(V_delta_final)
                        v_src_list.append(V_src_final)
                        v_mask += window_mask_sum
                        v_tar_mask += window_tar_mask_sum
                        v_bg_mask += window_bg_mask_sum
                        if time < worse_avg:
                            v_worse += V_delta_final

                    v_list = torch.stack(v_list, dim=0) 
                    v_src_list = torch.stack(v_src_list, dim=0) 
                    V_delta_better = v_list.mean(dim=0)
                    V_src_better = v_src_list.mean(dim=0)
                    
                    v_trend = []
                    for worse_set in itertools.combinations(v_list, worse_avg):
                        v_worse = torch.zeros_like(V_delta_better)
                        for worse in worse_set:
                            v_worse += worse
                        v_worse = v_worse / worse_avg
                        v_trend.append(V_delta_better - v_worse)
                    v_trend = torch.stack(v_trend, dim=0)
                    v_trend = v_trend.mean(dim=0)
                    
                    v_mask = torch.clamp(v_mask, min=0.0, max=1.0)
                    v_mask = self.soften_mask_edges(v_mask, decay_factor=decay_factor)
                    v_tar_mask = torch.clamp(v_tar_mask, min=0.0, max=1.0)
                    v_tar_mask = self.soften_mask_edges(v_tar_mask, decay_factor=decay_factor)
                    v_bg_mask = torch.clamp(v_bg_mask, min=0.0, max=1.0)
                    v_bg_mask = self.soften_mask_edges(v_bg_mask, decay_factor=decay_factor)
                    
                    #new_mask=final_mask#+v_tar_mask
                    #new_mask = torch.clamp(new_mask, min=0.0, max=1.0)
                    #new_mask = self.soften_mask_edges(new_mask, decay_factor=decay_factor)
                    #self.visualize_and_save_masks(final_mask, v_tar_mask, self.vae_stride, self.patch_size, suffix=f"_{index}.mp4")
                    V_delta_final = V_delta_better + (omega - 1) * v_trend
                    #V_delta_final = V_delta_final 
                    #zt_edit_lora = zt_edit + (t_im1 - t_i) * V_delta_final 
                    zt_edit = zt_edit + (t_im1 - t_i) * V_delta_final * v_mask#- (E_q-E_p)*(t_im1 - t_i)#* new_mask 
                    '''                    
                    with torch.set_grad_enabled(True):  # ensure not inside a torch.no_grad() block
                        zt_edit_lora.requires_grad_()  # make sure latent has grad
                            #videos_lora = self.vae.decode([zt_edit_lora])[0]  # keep grad path
                        #motion=self.compute_motion(videos_lora[0])
                        #mask_lora=torch.from_numpy(masks).unsqueeze(0)
                        #mask_lora = self.soften_mask_edges(mask_lora, decay_factor=decay_factor).to(self.device)
                        x_src=x_src.to(self.device)
                        #print(f'mask shape: {mask_lora.shape}')
                        #print(f'motion shape: {motion.shape}')
                        #motion = motion.to(self.device) * mask_lora[:,1:,:,:].to(self.device)
                        #gt_motion = gt_motion.to(self.device) * mask_lora[:,1:,:,:].to(self.device)
                        loss = F_torch.smooth_l1_loss(zt_edit_lora*(1-final_mask), x_src*(1-final_mask), reduction='mean')
                        #print(zt_edit_lora.requires_grad)  # True
                        #print(motion.requires_grad)  # True
                        #print(loss.requires_grad)    # True
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()'''
                    #videos_lora = self.vae.decode([zt_edit_lora])
                    videos = self.vae.decode([zt_edit])
                    cache_video(
                        tensor=videos[0][None],
                        save_file=f'videos/step_{index}.mp4',
                        fps=30,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
                    
                    def save_velocity_maps(zt, V_edit, output_dir, prefix="vel", quantile=0.995):
                        os.makedirs(output_dir, exist_ok=True)
                        B, F, H, W = V_edit.shape
                        V_mag = V_edit.abs()  # or use torch.norm(V_edit, dim=1)
                        
                        vmax = torch.quantile(V_mag.flatten(1), quantile).item()
                        zmax = torch.quantile(zt.flatten(1), quantile).item()
                        for f in range(F):
                            frame_v = V_mag[0, f].cpu().numpy()
                            plt.imshow(frame_v, cmap='hot', vmin=0, vmax=vmax)
                            plt.axis('off')
                            plt.savefig(f"{output_dir}/{prefix}_f{f:03d}.jpg", bbox_inches='tight', pad_inches=0)
                            plt.close()
                            
                            frame_z = zt[0, f].cpu().numpy()
                            plt.imshow(frame_z, cmap='hot')
                            plt.axis('off')
                            plt.savefig(f"{output_dir}/{prefix}_z{f:03d}.jpg", bbox_inches='tight', pad_inches=0)
                            plt.close()
                            
                    save_velocity_maps(zt_edit, V_delta_final, output_dir=f"vel_vis2/step_{index}", prefix="v_edit")
                    #compute freq for all pixels inside v_bg_mask and (1-v_mask) for the whole video
                    # use svd or pca to find the main frequency component in (1-v_mask)
                    #apply a small freq_nudge in that direction for all pixels inside v_bg_mask while keeping the other areas fixed
                    #V_delta_final = V_delta_final * freq_nudge 
                    '''
                    # ─────────────────────────────────────────────────────────────────────────────
                    # >>> FIXED 4-D LATENT-SPACE FREQUENCY NUDGE (no time/freq mismatch)
                    print(zt_edit.shape)
                    freq_indices = [0, 1, 2, 3,4,5,6,7,8,9,10]  # Frequencies to visualize
                    # ------------------------------------------------------------------
                    out_dir = "heatmaps"
                    os.makedirs(out_dir, exist_ok=True)
                    # ------------------------------------------------------------------

                    # -------------  FFT over the **time** axis only  ------------------
                    zt_fft   = torch.fft.fft(zt_edit, dim=1)     # shape [B, F, H, W]
                    power    = zt_fft.abs() ** 2                 # Power spectrum

                    # -------------  Save one heat-map per frequency  ------------------
                    for f in freq_indices:
                        heatmap = power[0, f].cpu().numpy()      # pick batch idx 0
                        plt.figure(figsize=(6, 6))
                        plt.imshow(heatmap, cmap="inferno")
                        plt.title(f"Temporal Frequency {f}", fontsize=9)
                        plt.axis("off")
                        plt.colorbar(shrink=0.7)

                        fname = os.path.join(out_dir, f"frequency_{index}_{f}.png")
                        plt.savefig(fname, bbox_inches="tight", dpi=150)
                        plt.close()
                        print(f"Saved {fname}")
                    
                    nudge_strength = 0.10  # γ
                    nudge_clip     = 0.25  # clamp δ to ±0.25
                    eps            = 1e-8

                    # 1) shapes
                    B, F, H, W = zt_edit.shape

                    # 2) FFT on the latent (time→freq)
                    latents = zt_edit.unsqueeze(2)                       # [B,F,1,H,W]
                    fft_vol  = torch.fft.rfft(latents, dim=1, norm="ortho")  # [B,Ff,1,H,W]
                    power    = fft_vol.abs()[:, 1:]                          # [B,Ff-1,1,H,W]

                    # 3) build a SPATIAL hole/background mask
                    #    collapse over time so it's [B,H,W] and broadcastable
                    hole_spatial = (v_bg_mask.mean(dim=1) > 0.5).float()      # [B,H,W]
                    bg_spatial   = ((1 - v_mask).mean(dim=1) > 0.5).float()    # [B,H,W]

                    # 4) expand to match power dims: [B, Ff-1, 1, H, W]
                    hole_exp = hole_spatial.unsqueeze(1).unsqueeze(2)         # [B,1,1,H,W] → broadcast to freq
                    bg_exp   = bg_spatial  .unsqueeze(1).unsqueeze(2)         # same

                    # 5) compute mean high-freq energy in each region
                    hole_E = (power * hole_exp).sum() / (hole_exp.sum() + eps)
                    bg_E   = (power *   bg_exp).sum() / (  bg_exp.sum()   + eps)

                    # 6) signed difference δ, clamp, and form nudge factor
                    delta = (bg_E - hole_E).clamp(-nudge_clip, nudge_clip)  # scalar
                    nudge = 1.0 + nudge_strength * delta                    # scalar

                    # 7) apply *after* your integration step to the state zt_edit,
                    #    not to V_delta_final, so you’re nudging the latent itself:
                    zt_edit = zt_edit * (1 + (nudge - 1) * hole_spatial.unsqueeze(1))
                    # ─────────────────────────────────────────────────────────────────────────────
                    # ─────────────────────────────────────────────────────────────────────────────

                                   '''
        
        def visualize_and_save_masks(v_bg_mask, v_tar_mask, vae_stride, patch_size, suffix=".mp4"):
            """
            v_bg_mask:   Tensor [C, F, H_lat, W_lat]  ⟵ your latent‐space mask
            v_tar_mask:  same shape
            vae_stride:  (stride_t, stride_h, stride_w)
            patch_size:  (p_t, p_h, p_w)
            """

            bg = v_bg_mask.mean(dim=0)  # [F, H_lat, W_lat]
            tar = v_tar_mask.mean(dim=0)

            # Step 2: add channel dim → [1, F, H_lat, W_lat] → then permute to [1, 1, F, H_lat, W_lat]
            bg = bg.unsqueeze(0).unsqueeze(0)   # [1, 1, F, H_lat, W_lat]
            tar = tar.unsqueeze(0).unsqueeze(0)

            stride_t, stride_h, stride_w = vae_stride
            F_lat, H_lat, W_lat = bg.shape[-3:]
            H = H_lat * stride_h
            W = W_lat * stride_w
            F_up = F_lat * stride_t

            # Step 3: Interpolate (upsample) in 3D (T x H x W)
            bg_up = F_torch.interpolate(bg, size=(F_up, H, W), mode='trilinear', align_corners=False)
            tar_up = F_torch.interpolate(tar, size=(F_up, H, W), mode='trilinear', align_corners=False)

            # Step 4: squeeze batch & channel → [F, H, W]
            bg_up = bg_up.squeeze(0).squeeze(0)
            tar_up = tar_up.squeeze(0).squeeze(0)

            # Step 5: turn into 3-channel RGB heatmaps → [3, F, H, W]
            bg_rgb = torch.stack([bg_up]*3, dim=0)
            tar_rgb = torch.stack([tar_up]*3, dim=0)

            # Step 6: add batch dim and clamp → [1, 3, F, H, W]
            bg_rgb = bg_rgb.clamp(0, 1).unsqueeze(0)
            tar_rgb = tar_rgb.clamp(0, 1).unsqueeze(0)

            # Step 7: Save to disk
            save_bg = cache_video(bg_rgb, save_file="walking_src_mask.mp4", fps=12, suffix=".mp4")
            save_tar = cache_video(tar_rgb, save_file="walking_tar_mask.mp4", fps=12, suffix=".mp4")

            print("Saved mask videos:", save_bg, save_tar)
            
        # 解码编辑结果
        if offload_model:
            self.model.cpu()
        if self.rank == 0:
            videos = self.vae.decode([zt_edit])
            visualize_and_save_masks(v_bg_mask, v_tar_mask,
                         vae_stride=self.vae_stride,
                         patch_size=self.patch_size)

            return videos[0]
        
        return None
    
    def configure_adapter(self, adapter_config, num_last_blocks=20):

        # 1) Discover all block indices in the model
        block_ids = set()
        for name, module in self.model.named_modules():
            if name.startswith("blocks."):
                parts = name.split(".")
                try:
                    idx = int(parts[1])
                    block_ids.add(idx)
                except:
                    pass
        block_ids = sorted(block_ids)
        last_blocks = set(block_ids[-num_last_blocks:])  # e.g. {24,25,26,27,28,29}

        # 2) Define which submodules inside each block to LoRA
        target_substrs = [
            "self_attn.q",   "self_attn.k",  "self_attn.v",  "self_attn.o",
            # optionally cross-attn:
            "cross_attn.q","cross_attn.k","cross_attn.v","cross_attn.o",
            "ffn.0",         "ffn.2",
        ]

        target_linear_modules = set()
        for name, module in self.model.named_modules():
            # must be in one of the last blocks
            if not name.startswith("blocks."):
                continue
            block_idx = int(name.split(".")[1])
            if block_idx not in last_blocks:
                continue

            # match one of our target substrings
            if any(substr in name for substr in target_substrs):
                # and it must be an nn.Linear
                if isinstance(module, nn.Linear):
                    target_linear_modules.add(name)

        if not target_linear_modules:
            raise ValueError(f"No Linear layers matched in last {num_last_blocks} blocks.")

        print("✅ LoRA will be injected into:")
        for m in sorted(target_linear_modules):
            print("   ", m)

        # 3) Build and attach LoRA
        peft_config = peft.LoraConfig(
            r=adapter_config["rank"],
            lora_alpha=adapter_config["alpha"],
            lora_dropout=adapter_config["dropout"],
            bias="none",
            target_modules=list(target_linear_modules),
        )
        self.lora_model = peft.get_peft_model(self.model, peft_config)
        self.model = self.lora_model
        self.lora_model.print_trainable_parameters()

    def configure_adapter_(self, adapter_config):
        target_modules = ["v", "q", "ffn.0", "o", "k", "ffn.2"]
        target_linear_modules = set()
        
        exclude_linear_modules = adapter_config.get('exclude_linear_modules', [])

        for name, module in self.model.named_modules():
            if module.__class__.__name__ not in target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    should_exclude = any(exclude_name in full_submodule_name for exclude_name in exclude_linear_modules)
                    if not should_exclude:
                        target_linear_modules.add(full_submodule_name)

        target_linear_modules = list(target_linear_modules)

        if adapter_config['type'] == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f"Adapter type {adapter_config['type']} is not implemented")

        self.peft_config = peft_config

        # 🔄 Replace original model with LoRA-wrapped version
        self.lora_model = peft.get_peft_model(self.model, peft_config)
        self.model = self.lora_model  # Use this going forward

        self.lora_model.print_trainable_parameters()

        # Optional: move trainable LoRA weights to correct dtype
        for name, p in self.lora_model.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])
