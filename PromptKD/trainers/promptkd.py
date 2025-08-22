import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
from tqdm import tqdm
import math

from clip.model import VisionTransformer, convert_weights

_tokenizer = _Tokenizer()

class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat):
        
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        
        return final_feat.squeeze(-1).squeeze(-1)
        
def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.PROMPTKD.TEACHER_NAME
    # url = clip._MODELS[backbone_name]
    
    if backbone_name == "ViT-B/16":
        model_path = './clip/ViT-B-16.pt'
    elif backbone_name == "ViT-L/14":
        model_path = './clip/ViT-L-14.pt'
    elif backbone_name == "ViT-B/32":
        model_path = './clip/ViT-B-32.pt'
    else:
        print('enter the wrong teacher name.')
    
    print(f"CLIP Teacher name is {backbone_name}")
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PromptSRC to pretrain our teacher model
    design_details = {"trainer": 'IVLP',
                        "vision_depth": 9,
                        "language_depth": 9,
                        "vision_ctx": 4,
                        "language_ctx": 4}
    
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    model_path = './clip/ViT-B-16.pt'
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.PROMPTKD.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        
        # print(f'------prompts size is {prompts.size()}------')
        # print(f'------tokenized prompts size is {tokenized_prompts.size()}------')

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL
        
        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTKD.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        
        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS
            
        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            
            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # print(f'label is {label}')
        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # print(f'ctx size is {ctx.size()}')

        prefix = self.token_prefix
        # print(f'prefix size is {prefix.size()}')
        
        suffix = self.token_suffix
        # print(f'suffix size is {suffix.size()}')

        if self.trainer_name == "PromptKD_Refined" and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)
            # No need to modify ctx as prefix and suffix concatenation results in the same size as n_cls

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # This is for transforming global features, as in the original PromptKD
        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 768)
        
        # This is for transforming local features, as in TextRefiner
        self.mid_image_trans = Feature_Trans_Module_two_layer(512, 512)

        self.cfg = cfg
        
        self.VPT_image_trans = self.VPT_image_trans.cuda()
        convert_weights(self.VPT_image_trans)
        self.mid_image_trans = self.mid_image_trans.cuda()
        convert_weights(self.mid_image_trans)

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
        
        # Get both global and local features
        image_features, image_fine = self.image_encoder(image.type(self.dtype), all_layer_outputs=True)
        
        # Global feature transformation (from original PromptKD)
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Local feature processing (from TextRefiner)
        image_fine_list = []
        top_image_fine_list = []
        B, d_global = image_features.shape # d_global is 768
        
        image_fine_features = [x[0] for x in image_fine] # B,N,d_local (d_local is 512)
        image_fine_attns =  [x[1] for x in image_fine] # B,1,N
        layers=[-1] 
        _, _, d_local = image_fine_features[0].shape # d_local is 512
        
        for i, layer in enumerate(layers):
            x = image_fine_features[layer]
            x = x.reshape(-1, d_local)

            # Transform local features
            x = self.mid_image_trans(x)
            x = x.reshape(B, -1, d_local)

            image_fine_feature = x
            image_fine_list.append(image_fine_feature)
            
            k = 5
            _, indices = torch.topk(image_fine_attns[layer], k=k, dim=-1)
            indices += 1
            indices = torch.cat((torch.zeros(B, 1, dtype=torch.int64).cuda(), indices), dim=1)
            top_image_fine_feature = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(B, k + 1, d_local))
            avg_image_feature = torch.mean(x, dim=1, keepdim=True)
            top_image_fine_feature = torch.cat((top_image_fine_feature, avg_image_feature), dim=1)

            top_image_fine_list.append(top_image_fine_feature.reshape(-1, d_local))

        if len(image_fine_list) > 0:
            image_fine_list = torch.cat(image_fine_list)
            top_image_fine_list = torch.cat(top_image_fine_list)

        return image_features, logit_scale, image_fine_list, top_image_fine_list


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, True)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
  
    def forward(self, image=None, label=None):
        
        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        
        logits = logit_scale * image_features @ text_features.t()
        
        return image_features, text_features, logits


class Memory(nn.Module):
    def __init__(self, clip_model,feature_dim = 512, memory_size = 25, reduction = 4, frozen_text_embedding=None, alpha=0.2,momentum=0.8):
        super().__init__()
        self.device = clip_model.dtype
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.text_fine_cache = F.normalize(torch.rand(self.memory_size, feature_dim), dim = -1)
        self.text_fine_cache = self.text_fine_cache.to(self.device)
        self.text_fine_cache = self.text_fine_cache.cuda()
        self.alpha = alpha
        self.momentum = momentum

        if frozen_text_embedding is not None:
            self.frozen_text_embedding = frozen_text_embedding

        
        self.extractor = nn.Linear(2 * feature_dim , feature_dim, bias=False)
        self.extractor = self.extractor.to(self.device)
        self.extractor = self.extractor.cuda()


        self.writeTF = lambda x: x.clone()
        
    def forward(self, text_token=None, image_token=None, training=False):
        fine_feature = self.read(text_token, training)

        text_fine_feature = torch.cat((text_token, fine_feature), dim = -1)
        text_fine_feature = self.alpha * self.extractor(text_fine_feature) + text_token
        if training:
            _ = self.write(image_token)
            normalized_text_features = F.normalize(text_fine_feature, dim = -1)
            loss = F.l1_loss(normalized_text_features, text_token, reduction='mean') 
        else:
            loss = 0.0
        return text_fine_feature, loss

    def get_score(self, query, mem):
        score = query @ mem.t() 
        score_query = F.softmax(score, dim = 0)
        score_mem = F.softmax(score, dim = 1)
        return score_query, score_mem
    
    def read(self, x, training=False):
        base_features = F.normalize(x, dim = -1) 
        C, d = x.shape
        if training:
            self.text_fine_cache = self.text_fine_cache.detach()
        _, softmax_score_cache = self.get_score(base_features, self.text_fine_cache)
        fine_feature = softmax_score_cache @ self.text_fine_cache  # (N, d)
        
        return fine_feature

    def write(self, x):
        m, d = self.text_fine_cache.shape
        ratio = 0.2
        base_features = x.clone()
        base_features = self.writeTF(base_features)


        base_features = base_features.reshape(-1, d) # (B * P, d)
        base_features = F.normalize(base_features, dim = -1)


        softmax_score_query, softmax_score_cache = self.get_score(base_features, self.text_fine_cache) #(B*P, 50)
        _, updating_indices = torch.topk(softmax_score_cache, 1, dim=1)
        

        updated_cache = self.text_fine_cache.clone().detach()
        for i in range(m):
            idx = torch.nonzero(updating_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                score = (softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
                updated_cache[i] = self.momentum * self.text_fine_cache[i] + (1 - self.momentum) * torch.sum(score * base_features[idx.squeeze(1)], dim=0)
        
        updated_cache = F.normalize(updated_cache, dim = -1)

        loss = 0.0
        self.text_fine_cache = updated_cache.to(self.device)
        return loss
    def diversityloss(self, mem):
        # it is same with orthonomal constraints.
        cos_sim = torch.matmul(mem,torch.t(mem))
        margin = 0 
        cos_sim_pos = cos_sim-margin
        cos_sim_pos[cos_sim_pos<0]=0
        loss = (torch.sum(cos_sim_pos)-torch.trace(cos_sim_pos))/(self.memory_size*(self.memory_size-1))
        return loss


@TRAINER_REGISTRY.register()
class PromptKD_Refined(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTKD.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.PROMPTKD.PREC == "fp32" or cfg.TRAINER.PROMPTKD.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Building teacher model")
        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)

        print("Building Memory")
        # Note: The feature dimension for memory should match the teacher's text feature dimension
        teacher_feature_dim = self.model_teacher.text_encoder.text_projection.shape[1]
        
        # Set default values for missing configuration parameters
        memory_size = getattr(cfg.TRAINER.PROMPTKD, 'MEMORY_SIZE', 25)
        alpha = getattr(cfg.TRAINER.PROMPTKD, 'ALPHA', 0.2)
        
        self.memory = Memory(clip_model, feature_dim=teacher_feature_dim, 
                             memory_size=memory_size, 
                             alpha=alpha)
        
        if cfg.TRAINER.MODAL == "base2novel":
            model_path = './teacher_model/'+str(cfg.DATASET.NAME)+'/VLPromptLearner/model-best.pth.tar'
        elif cfg.TRAINER.MODAL == "cross":
            model_path = './teacher_model/ImageNet-xd/VLPromptLearner_large/model.pth.tar-20'
            
        self.train_modal = cfg.TRAINER.MODAL
        
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]
        
        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device)
        self.model_teacher.eval()
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        # Add memory parameters to the trainable set
        for name, param in self.memory.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.memory.to(self.device)
        # NOTE: only give prompt_learner to the optimizer

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)
        self.trainable_list.append(self.memory)

        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        self.register_model("Memory", self.memory, self.optim, self.sched)
        
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.temperature = getattr(cfg.TRAINER.PROMPTKD, 'TEMPERATURE', 4.0)
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTKD.PREC
        
        # Define weights at the beginning for use in both AMP and non-AMP paths
        kd_weight = getattr(self.cfg.TRAINER.PROMPTKD, 'KD_WEIGHT', 1.0)
        local_weight = getattr(self.cfg.TRAINER.PROMPTKD, 'LOCAL_WEIGHT', 1.0)
        mem_weight = getattr(self.cfg.TRAINER.PROMPTKD, 'MEM_WEIGHT', 0.1)

        with torch.no_grad():
            tea_image_features, tea_text_features, tea_logits = self.model_teacher(image)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        if prec == "amp":
            with autocast():
                image_features, logit_scale, image_fine_list, top_image_fine_list = model(image, label)
                
                # Use teacher's text features as the base for refinement
                refined_text_features, loss_mem = self.memory(text_token=tea_text_features, 
                                                              image_token=image_fine_list, 
                                                              training=True)

                # 1. Logits distillation loss (original PromptKD loss)
                stu_logits = logit_scale * image_features @ refined_text_features.t()
                L_kd = F.kl_div(
                    F.log_softmax(stu_logits / self.temperature, dim=1),
                    F.softmax(tea_logits / self.temperature, dim=1),
                    reduction='sum',
                ) * (self.temperature * self.temperature) / stu_logits.numel()

                # 2. Local alignment loss (from TextRefiner)
                top_local_logit = logit_scale * top_image_fine_list @ tea_text_features.t()
                L_local = F.cross_entropy(top_local_logit, label.repeat_interleave(dim=0, repeats=7))

                # 3. Total loss
                loss = (kd_weight * L_kd +
                        local_weight * L_local +
                        mem_weight * loss_mem)
                        
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            image_features, logit_scale, image_fine_list, top_image_fine_list = model(image, label)
            
            # Use teacher's text features as the base for refinement
            refined_text_features, loss_mem = self.memory(text_token=tea_text_features, 
                                                          image_token=image_fine_list, 
                                                          training=True)

            # 1. Logits distillation loss (original PromptKD loss)
            # Student's global features align with refined text features
            stu_logits = logit_scale * image_features @ refined_text_features.t()
            L_kd = F.kl_div(
                F.log_softmax(stu_logits / self.temperature, dim=1),
                F.softmax(tea_logits / self.temperature, dim=1),
                reduction='sum',
            ) * (self.temperature * self.temperature) / stu_logits.numel()

            # 2. Local alignment loss (from TextRefiner)
            # Student's local features align with original teacher text features
            top_local_logit = logit_scale * top_image_fine_list @ tea_text_features.t()
            # The number of local tokens is 7 (k=5 + 1 for CLS + 1 for avg)
            L_local = F.cross_entropy(top_local_logit, label.repeat_interleave(dim=0, repeats=7))

            # 3. Total loss
            loss = (kd_weight * L_kd +
                    local_weight * L_local +
                    mem_weight * loss_mem)

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "L_kd": L_kd.item() * kd_weight,
            "L_local": L_local.item() * local_weight,
            "L_mem": loss_mem.item() * mem_weight,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]
                
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]
                
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            
            with torch.no_grad():
                tea_image_features, tea_text_features, tea_logits = self.model_teacher(image, label)
                
            image_ft, logit_scale, _, _ = self.model(image, label)
            
            # Use memory module to refine text features for evaluation
            refined_text_features, _ = self.memory(text_token=tea_text_features, training=False)

            if self.train_modal == "base2novel":
                if split == "val":
                    output = logit_scale * image_ft @ refined_text_features[:math.ceil(self.n_cls / 2),:].t()
                elif split == "test":
                    output = logit_scale * image_ft @ refined_text_features[math.ceil(self.n_cls / 2):,:].t()
            elif self.train_modal == "cross" :
                output = logit_scale * image_ft @ refined_text_features.t()
            
            self.evaluator.process(output, label) 

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

