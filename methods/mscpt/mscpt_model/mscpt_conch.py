import torch
import torch.nn as nn
from torchvision import transforms
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from torch_geometric.nn import GCNConv
import json
import torch.nn.functional as F
import os


def create_model(
    model_name: str,
    override_image_size = 224,
    pretrain_path = "path/to/weights"
    ):  
    model = create_model_from_pretrained("conch_ViT-B-16", 
                                 checkpoint_path=pretrain_path, 
                                 force_image_size=override_image_size,
                                 return_transform=False)
    return model

class GcnPromptLearner(nn.Module):
    def __init__(self, traing):
        super(GcnPromptLearner, self).__init__()
        self.conv1 = GCNConv(512, 512)
        self.conv2 = GCNConv(512, 512)
        self.training = traing

    def forward(self, x, edge_index, edge_attr):

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class MYPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_tpro, n_high, tokenizer):
        super().__init__()
        self.n_tpro = n_tpro # prompt length
        self.n_high = n_high # number of high-level prompts
        self.ctx_dim = clip_model.text.ln_final.weight.shape[0]
        self.layers = len(clip_model.text.transformer.resblocks)
        self.dtype = clip_model.text.ln_final.weight.dtype
        self.text = clip_model.text                            
        # global prompt for text encoder (except for the first layer)   全局prompt，为每一层设置可训练parametr
        self.p_uni = nn.ParameterList([nn.Parameter(torch.empty(self.n_tpro, self.ctx_dim).type(self.dtype))
                                                      for _ in range(self.layers - 1)]) #[11,2,512]
        for p in self.p_uni:
            nn.init.normal_(p, std=0.02)
            
        # projector for learning high-level prompt (a.k.a p_ins)
        self.p_ins_projector = nn.Linear(self.ctx_dim, self.ctx_dim)
        
        # global prompt for the first layer of the text encoder
        self.p_input = nn.Parameter(torch.empty(self.n_tpro+self.n_high, self.ctx_dim))  #7,512
        nn.init.normal_(self.p_input, std=0.02)
        
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(classnames)
        self.clip_model = clip_model
        self.tokenizer = tokenizer

    def forward(self, feats, desc, embed_cls=True):
        p_uni = self.p_uni
        prompts = []
        prompt_prefix = " ".join(["X"] * (self.n_tpro+self.n_high))

        for name in self.classnames:
            # We leverage all structures from descriptions as a part of input respectively during evaluation.
            for id in range(len(desc[name]['big_mag'])):
                p = prompt_prefix + ' ' + desc[name]['big_mag'][id]
                prompts.append(p)
        
        tokenized_prompts = tokenize(
            texts=prompts, tokenizer=self.tokenizer).to(self.p_input.device)
        tokenized_prompts = tokenized_prompts[:, :-1] if embed_cls else tokenized_prompts
        seq_len = tokenized_prompts.shape[1]

        cast_dtype = self.text.transformer.get_cast_dtype()
        
        with torch.no_grad():
            attn_mask = self.text.attn_mask
            embedding = self.text.token_embedding(tokenized_prompts).to(cast_dtype)
            if self.text.cls_emb is not None:
                seq_len += 1
                embedding = torch.cat([embedding, self.text._repeat(self.text.cls_emb, embedding.shape[0])], dim=1)
                cls_mask = self.text.build_cls_mask(tokenized_prompts, cast_dtype)
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
            embedding = embedding + self.text.positional_embedding[:seq_len].to(cast_dtype)

        p_input = self.p_input.unsqueeze(0).expand(len(prompts), -1, -1)
        prefix = embedding[:, :1]
        suffix = embedding[:, 1+self.n_tpro+self.n_high:]

        p_ori = torch.cat([prefix, p_input, suffix], dim=1)

        p_ins = []
        feats = feats.permute(1, 0, 2, 3)
        (l, c, n, d) = feats.shape
        feats = feats.reshape(l, c*n, d)
        for idx in range(self.layers - 1):
            feat = feats[idx].float()
            feat = feat + self.p_ins_projector(feat) 
            p_ins.append(feat)
        p_ins = torch.stack(p_ins, dim=0)

        return p_ori, p_ins, p_uni, attn_mask

class VisionPromptLearner(nn.Module):
    def __init__(self, clip_model, n_vpro):
        super().__init__()
        self.n_vpro = n_vpro
        self.pro_dim = clip_model.visual.trunk.norm.weight.shape[0]
        self.dtype = clip_model.visual.trunk.norm.weight.dtype
        self.layers = len(clip_model.visual.trunk.blocks)
        self.embeddings = clip_model.visual.trunk.patch_embed
        self.pos_embed = clip_model.visual.trunk._pos_embed
        self.p_visual = nn.ParameterList([nn.Parameter(torch.empty(self.n_vpro, self.pro_dim).type(self.dtype)) for _ in range(self.layers-1)])

        for p in self.p_visual:
            nn.init.normal_(p, std=0.02)
            
        # global prompt for the first layer of image encoder
        self.p_input = nn.Parameter(torch.empty(self.n_vpro, self.pro_dim))
        nn.init.normal_(self.p_input, std=0.02)

    def forward(self, x):
        x = x.type(self.dtype)
        x = self.embeddings(x)
        x = self.pos_embed(x)
        p_input = self.p_input.unsqueeze(0).expand(len(x), -1, -1)
        x = torch.cat([x, p_input], dim=1)

        return x, self.p_visual
    
class VisionEncoder(nn.Module):
    def __init__(self, clip_model, n_vpro):
        super().__init__()
        self.n_vpro = n_vpro
        self.pre_layrnorm = clip_model.visual.trunk.norm_pre
        self.post_layernorm = clip_model.visual.trunk.norm
        self.dtype = clip_model.visual.trunk.norm.weight.dtype
        self.layers = clip_model.visual.trunk.blocks
        self.attn_pool_contrast = clip_model.visual.attn_pool_contrast
        self.ln_contrast = clip_model.visual.ln_contrast
        self.proj_contrast = clip_model.visual.proj_contrast
    
    def forward(self, x, p_visual):
        hidden_states = self.pre_layrnorm(x).type(self.dtype)
        for layer_idx, encoder_layer in enumerate(self.layers):
            if layer_idx > 0:
                hidden_states[:,-self.n_vpro:] = p_visual[layer_idx-1].unsqueeze(0)
            hidden_states = encoder_layer(hidden_states)
        
        pooled_output = self.attn_pool_contrast(hidden_states)[:, 0]
        pooled_output = self.ln_contrast(pooled_output)
        out_put = pooled_output @ self.proj_contrast
        return out_put
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_tpro, n_high):
        super().__init__()
        self.n_tpro = n_tpro # prompt length
        self.n_high = n_high
        self.layers = len(clip_model.text.transformer.resblocks)
        self.dtype = clip_model.text.ln_final.weight.dtype
        self.text = clip_model.text

    def forward(self, x, p_ins, p_uni, attn=None):  # x是第一层的global prompts和low-level prompts
        # p_ins: instance-specific prompt, a.k.a high-level prompt from descriptions
        # p_uni: task-unified prompt, a.k.a global-level prompt
        # flag: True when training and False when testing
        # Since we use all (self.n_set) descriptions for learning high-level prompt, we should reshape p_ins first.
        (l, k, d) = p_ins.shape
        p_ins = p_ins.reshape(l, k//self.n_high, self.n_high, d) # [11, 3, 5, 512]   [num_layers, num_class, num_high_desc, dim]
        p_ins = p_ins.unsqueeze(2).expand(-1, -1, x.shape[0]//(k//self.n_high), -1, -1) # [11, 3, 20, 5, 512] [num_layers, num_class, num_low_desc, num_high_desc, dim]
        (l, num_class, num_low_desc, num_high_desc, d) = p_ins.shape
        p_ins = p_ins.reshape(l, num_class*num_low_desc, num_high_desc, d) # [11, 60, 5, 512] [num_layers, num_class*num_low_desc, num_high_desc, dim]
        p_ins = p_ins.type(self.dtype)

        x = x.permute(1, 0, 2)
        for layer_idx, layer in enumerate(self.text.transformer.resblocks):
            if layer_idx > 0:               
                prefix = x[:1]
                suffix = x[1+self.n_tpro+self.n_high:]
                
                # global-level prompt
                ctx_g = p_uni[layer_idx - 1].unsqueeze(1).expand(self.n_tpro, prefix.shape[1] , -1)
                
                # high-level prompt
                ctx_l = p_ins[layer_idx - 1].permute(1, 0, 2)
                x = torch.cat([prefix, ctx_g, ctx_l, suffix], dim=0)
                
                # 'attn' is attention matrix from topological prompt learner, 
                # considering as low-level prompt which models relationships in an explicit way.
                x = layer(x, attn)
                
            elif layer_idx == 0:
                x = layer(x, attn)
            else:
                x = layer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.text.cls_emb is not None:
            pooled_output, _ = x[:, -1], x[:, :-1]
            pooled_output = self.text.ln_final(pooled_output)
        out_put = pooled_output @ self.text.text_projection

        return out_put

class TextEncoderZS(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text = clip_model.text

    def forward(self, input_ids, attention_mask=None, embed_cls=True):
        input_ids = input_ids.to(self.text.token_embedding.weight.device)
        input_ids = input_ids[:, :-1] if embed_cls else input_ids # make space for CLS token
        seq_len = input_ids.shape[1]
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        cast_dtype = self.text.transformer.get_cast_dtype()
        hidden_states = self.text.token_embedding(input_ids).to(cast_dtype)
        attn_mask = self.text.attn_mask
        if self.text.cls_emb is not None:
            seq_len += 1
            hidden_states = torch.cat([hidden_states, self.text._repeat(self.text.cls_emb, hidden_states.shape[0])], dim=1)
            cls_mask = self.text.build_cls_mask(input_ids, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        hidden_states = hidden_states + self.text.positional_embedding[:seq_len].to(cast_dtype)
        hidden_states = hidden_states.permute(1, 0, 2)  # NLD -> LND
        feats = []
        for _, layer in enumerate(self.text.transformer.resblocks):
            hidden_states = layer(hidden_states, attn_mask)
            feats.append(hidden_states[-1,:,:])
        hidden_states = hidden_states.permute(1, 0, 2)

        if self.text.cls_emb is not None:
            pooled_output, _ = hidden_states[:, -1], hidden_states[:, :-1]
            pooled_output = self.text.ln_final(pooled_output)
        out_put = pooled_output @ self.text.text_projection
        txt_feats = torch.stack(feats)

        return out_put, txt_feats

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, gpt_dir, dataset_name,
                 base_model, n_high, n_tpro, n_vpro, tokenizer=None,
                 high_patch_topk=5, low_patch_topk=100):
        super().__init__()

        for p in clip_model.parameters():
            p.requires_grad = False
        # Load description and structure from gpt
        f_json = os.path.join(gpt_dir+'/description', dataset_name+'.json')
        with open(f_json, 'r') as f:
            text_prompts = json.load(f)

        tokenizer = tokenizer or get_tokenizer()
        self.prompt_learner = MYPromptLearner(classnames, clip_model, n_tpro, n_high, tokenizer)
        self.gcn_prompt_learner_big = GcnPromptLearner(self.training)
        self.gcn_prompt_learner_small = GcnPromptLearner(self.training)
        self.vision_prompt_learner = VisionPromptLearner(clip_model, n_vpro)
        self.image_encoder = VisionEncoder(clip_model, n_vpro)
        self.text_encoder = TextEncoder(clip_model, n_tpro, n_high)
        self.text_encoder_zs = TextEncoderZS(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.model = clip_model
        self.n_class = len(classnames)
        self.high_patch_topk = int(high_patch_topk)
        self.low_patch_topk = int(low_patch_topk)
        if self.high_patch_topk <= 0 or self.low_patch_topk <= 0:
            raise ValueError("MSCPT patch top-k values must be positive")

        with torch.no_grad():
            # zs_feats: layer-wise class embeddings from frozen text encoder
            # zs_repres: final representations from frozen text encoder
            zs_feats, zs_repres = [], []
            for classname in classnames:
                texts = text_prompts[classname]
                class_texts = texts['small_mag']

                class_texts = tokenize(texts=class_texts, tokenizer=tokenizer)

                class_embeddings, features = self.text_encoder_zs(class_texts)
                class_embeddings = F.normalize(class_embeddings, dim=-1)
                features = F.normalize(features, dim=-1)
                zs_feats.append(features)
                zs_repres.append(class_embeddings)
            model_device = next(clip_model.parameters()).device
            self.text_features_zs = torch.stack(zs_repres, dim=0).to(model_device)
            self.text_features_ft = torch.stack(zs_feats, dim=0).to(model_device) # [3, 11, 5, 512] [num_class, num_layers, num_desc, dim]
            self.text_prompts = text_prompts
            self.clip_model_proj = clip_model.visual.proj_contrast

    def forward(self, big_image, small_embeddings, train=True):
        big_image = big_image.squeeze(0)
        small_embeddings = small_embeddings.squeeze(0)
        projection = self.clip_model_proj
        if small_embeddings.shape[-1] == projection.shape[-1]:
            # Benchmark bags may already be in CONCH's shared 512-D space.
            image_features_zs = F.normalize(small_embeddings, dim=1)
        elif small_embeddings.shape[-1] == projection.shape[0]:
            image_features_zs = F.normalize(
                small_embeddings @ projection, dim=1)
        else:
            raise ValueError(
                "MSCPT-CONCH low features must be pre-projection width "
                f"{projection.shape[0]} or shared width {projection.shape[-1]}, "
                f"got {small_embeddings.shape[-1]}")
        logit_scale = self.logit_scale.exp()

        text_features_zs = self.text_features_zs
        text_features_zs = text_features_zs.reshape(-1, text_features_zs.shape[-1])
        image_features_zs = image_features_zs.reshape(-1, image_features_zs.shape[-1])

        p_ori, p_ins, p_uni, attention_mask = self.prompt_learner(self.text_features_ft, self.text_prompts)
        text_features = self.text_encoder(p_ori, p_ins, p_uni, attention_mask)
        text_features = F.normalize(text_features, dim=1)

        # Since we use multiple structures for producing representations of one category, 
        # we should take their mean value as the final representation.
        if big_image.ndim == 2 and big_image.shape[-1] == projection.shape[-1]:
            # Feature-only benchmark mode: high-resolution bags have already
            # passed through the paired CONCH visual tower.
            image_features = F.normalize(big_image, dim=1)
        else:
            x, p_visual = self.vision_prompt_learner(big_image)
            image_features = self.image_encoder(x, p_visual)
            image_features = F.normalize(image_features, dim=1)

        # asymmetric loss 
        sim_big = image_features @ text_features_zs.t() #归一化后（nrom）的余弦相似度与点积计算公式相同
        sim_big_ = sim_big

        sim_small = image_features_zs @ text_features.t()
        sim_small_ = sim_small

        sim_big_ = sim_big_.reshape(sim_big_.shape[0], self.n_class, -1).permute(1,0,2)
        A_big = torch.bmm(sim_big_, sim_big_.transpose(1,2))
        A_big = torch.max(A_big, dim=0)[0]
        A_big = torch.softmax(A_big, dim=-1)
        num_nodes_big = A_big.shape[0]
        edge_index_big = torch.tensor([[i, j] for i in range(num_nodes_big) for j in range(i+1, num_nodes_big)], dtype=torch.long).t().contiguous()
        edge_attr_big = A_big[edge_index_big[0], edge_index_big[1]]
        x_big = self.gcn_prompt_learner_big(
            image_features, edge_index_big.to(A_big.device), edge_attr_big.to(A_big.device))

        sim_small_ = sim_small_.reshape(sim_small_.shape[0], self.n_class, -1)
        sim_small_ = sim_small_.permute(1,2,0)
        x_small = image_features_zs
        A_small = torch.bmm(sim_small_, sim_small_.transpose(1,2))
        A_small = torch.max(A_small, dim=0)[0]
        A_small = torch.softmax(A_small, dim=-1)
        num_nodes_small = A_small.shape[0]
        edge_index_small = torch.tensor([[i, j] for i in range(num_nodes_small) for j in range(i+1, num_nodes_small)], dtype=torch.long).t().contiguous()
        edge_attr_small = A_small[edge_index_small[0], edge_index_small[1]]
        x_small = self.gcn_prompt_learner_small(
            x_small, edge_index_small.to(A_small.device), edge_attr_small.to(A_small.device))

        logits_i = x_big @ text_features_zs.t()
        return_sim_big = logits_i.reshape(logits_i.shape[0], self.n_class, -1).cpu().detach().numpy()
        logits_i = logits_i.reshape(-1, self.n_class)
        high_k = min(self.high_patch_topk, logits_i.shape[0])
        logits_i = logit_scale * torch.topk(
            logits_i, high_k, dim=0)[0].mean(0)
        text_features_i = text_features.reshape(self.n_class, -1, text_features.shape[-1])
        text_features_i = text_features_i.mean(1)
        logits_i_cross = x_big @ text_features_i.t()
        logits_i_cross = logit_scale * torch.topk(
            logits_i_cross, high_k, dim=0)[0].mean(0)
        logits_i = logits_i + logits_i_cross
        
        logits_t = x_small @ text_features.t()
        return_sim_small = logits_t.reshape(logits_t.shape[0], self.n_class, -1).cpu().detach().numpy()
        logits_t = logits_t.reshape(-1, self.n_class)
        low_k = min(self.low_patch_topk, logits_t.shape[0])
        logits_t = logit_scale * torch.topk(
            logits_t, low_k, dim=0)[0].mean(0)
        text_features_t = text_features_zs.reshape(self.n_class, -1, text_features_zs.shape[-1])
        text_features_t = text_features_t.mean(1)
        logits_t_cross = x_small @ text_features_t.t()
        logits_t_cross = logit_scale * torch.topk(
            logits_t_cross, low_k, dim=0)[0].mean(0)
        logits_t = logits_t + logits_t_cross
        

        logits = (logits_i + logits_t)/2

        if train:
            return logits, logits_i, logits_t
        else:
            return logits, (return_sim_small, return_sim_big)

class MscptConch(nn.Module):

    def __init__(self, base_model='plip', base_pretrain_path='', trainer_perc='fp16', dataset_name='RCC',
                 gpt_dir='', label_dicts={}, n_set=5, n_tpro=2, n_vpro=2,
                 n_high=10, n_topk=100, high_patch_topk=5,
                 clip_model=None, tokenizer=None):
        super().__init__()

        classnames = [name for name in label_dicts.keys()]
        clip_model = clip_model or create_model(
            model_name=base_model, pretrain_path=base_pretrain_path)
        if trainer_perc in ['fp32', 'amp']:
            clip_model.float()

        print("Building custom CLIP")
        self.Custom_model = CustomCLIP(classnames, clip_model, gpt_dir, dataset_name,
                                        base_model, n_high, n_tpro, n_vpro,
                                        tokenizer=tokenizer,
                                        high_patch_topk=high_patch_topk,
                                        low_patch_topk=n_topk)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.Custom_model.named_parameters():
            if "prompt_learner"  not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.Custom_model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.Custom_model.parameters() if p.requires_grad)}")


    def forward(self, data, train=True):
        big_img = data[0]
        small_embeddings = data[1]
        if train:
            logits, logits_i, logits_t = self.Custom_model(big_img, small_embeddings, True)
            return logits.unsqueeze(0), logits_i.unsqueeze(0), logits_t.unsqueeze(0)
        else:
            logits, return_sim = self.Custom_model(big_img, small_embeddings, False)
            return logits.unsqueeze(0), return_sim

    
