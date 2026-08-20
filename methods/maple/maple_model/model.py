import torch
from torch import nn
import json
import os
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from .model_utils import MultiheadAttention

import math 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_model(
    model_name: str,
    override_image_size = None,
    pretrain_path = None
    ):  
    def clean_state_dict_ctranspath(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'attn_mask' in k:
                continue
            new_state_dict[k.replace('module.', '')] = v
        return new_state_dict
    
    if model_name == 'PLIP':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('vinid/plip')
        return model
    elif model_name == 'CLIP':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
        return model


def load_pretrained_tokenizer(model_name):
    if model_name == 'PLIP':
        model_name = 'vinid/plip'
        tokenizer = AutoTokenizer.from_pretrained('vinid/plip')
    elif model_name == 'CLIP':
        model_name = 'openai/clip-vit-base-patch16'
        tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16', use_fast=False, TOKENIZERS_PARALLELISM=True)
    
    return tokenizer

class PromptLearner(nn.Module):
    def __init__(self, prompt_path, clip_model, tokenizer, mode, n_ctx=0, all_ctx_trainable=True, csc=True, p_drop_out=0.0, attr_edge_topk=3):

        super().__init__()
        embedding_device = next(
            clip_model.text_model.embeddings.parameters()).device

        json_data = json.load(open(os.path.expanduser(prompt_path), "r"))
        
        text_ctx_list, text_num_list = [], []
        attr_list = []
        for level in ['low', 'high']:
            data = json_data[level]
            entities = data['entities']
            entities_num = len(entities)

            if mode == 'attribute':
                templete = 'an H&E stained image of {} at {} resolution.'

                text_feature_list, text_attrs_list = [], []
                for idx in range(entities_num):
                    entities_dict = entities[idx]
                    entities_name = entities_dict['name']
                    general_attr = entities_dict['general_feature']
                    text_feature = templete.format(entities_name, level).replace(".", ", {}".format(general_attr))
                    text_attrs = [templete.format(entities_name, level).replace(".", ", {}".format(value)) for key, value in entities_dict['attributes'].items()]

                    text_feature_list.append(text_feature)
                    text_attrs_list.extend(text_attrs) 
                
                text_ctx_list.extend(text_feature_list + text_attrs_list)
                text_num_list.extend([len(text_feature_list), len(text_attrs_list)])
                
                attr_list.extend(text_feature_list)

                # csc = csc 
            elif mode == 'region':
                templete = 'an H&E stained image of {} at {} resolution.'
                key, value = 'tumor', data['tumor']
                text_positive = templete.format(key, level).replace(".", ", {}".format(value))

                text_ctx_list.append(text_positive)
                text_num_list.append(1)

            else: 
                templete_wsi = 'a whole slide image of {} at {} resolution.'
                
                text_global = [templete_wsi.format(key, level).replace(".", ", {}".format(value)) for key, value in data['global_info'].items()]
                text_global_prototype = 'an H&E stained image at {} resolution.'.format(level)

                text_ctx_list.extend(text_global + [text_global_prototype]) 
                text_num_list.extend([len(text_global), 1])

                csc = False 

        if mode == 'attribute': 
            self.text_encoder = TextEncoder(clip_model) 
            for param in self.text_encoder.parameters(): 
                param.requires_grad = False 

            with torch.no_grad():
                tokens_dict = tokenizer(attr_list, max_length=77, padding='max_length', return_tensors='pt')
                tokenized_prompts = tokens_dict['input_ids'].to(
                    embedding_device)
                attention_mask = tokens_dict['attention_mask'].to(
                    embedding_device)
                embedding = clip_model.text_model.embeddings(tokenized_prompts).type(clip_model.dtype)
                causal_attention_mask = _create_4d_causal_attention_mask(tokenized_prompts.size(), embedding.dtype, device=embedding.device)
                if attention_mask is not None:
                    attention_mask = _prepare_4d_attention_mask(attention_mask, embedding.dtype)
                text_features_attr = self.text_encoder(embedding, tokenized_prompts, attention_mask, causal_attention_mask)
                
                topk = attr_edge_topk
                text_features_attr = F.normalize(text_features_attr)
                sim = torch.matmul(text_features_attr, text_features_attr.T)
                sim = torch.clamp(sim, min=0.0)
                neighbor_count = (sim.shape[0] if topk is None else
                                  min(topk + 1, sim.shape[0]))
                attr_edge = torch.zeros_like(sim, dtype=torch.bool)
                topk_idx = torch.topk(
                    sim, k=neighbor_count, dim=-1).indices
                row_idx = torch.arange(
                    sim.size(0), device=sim.device).unsqueeze(1).expand(
                        -1, neighbor_count)
                attr_edge[row_idx, topk_idx] = True
                attr_edge = attr_edge | attr_edge.T  # undirected

                self.attr_edge = attr_edge


        text_ctx = text_ctx_list
        text_ctx_X = [" ".join(["X"]*n_ctx) + " " + i for i in text_ctx]

        self.num_list = text_num_list 

        self.all_ctx_trainable = all_ctx_trainable
        self.n_prompts = len(text_ctx)

        dtype = clip_model.dtype 
        ctx_dim = clip_model.text_model.final_layer_norm.weight.shape[0]

        if csc == True: 
            ctx_init = text_ctx_X 
        else: 
            ctx_init = '' 

        if type(ctx_init) == list:
            self.use_class_specific_ctx = True
        else:
            self.use_class_specific_ctx = False

        if ctx_init:
            if not self.use_class_specific_ctx:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                n_fixed_ctx = len(ctx_init.replace(" *", "").split(" "))
                n_learnable_ctx = ctx_init.count("*")

                tokens = tokenizer(ctx_init, return_tensors='pt')
                prompt = tokens['input_ids'].to(embedding_device)
                attention_mask = tokens['attention_mask'].to(
                    embedding_device)
                # prompt = torch.from_numpy(np.array(prompt))

                with torch.no_grad():
                    embedding = clip_model.text_model.embeddings(prompt).type(dtype)
                    
                num_nonzero_token = prompt.nonzero().max()
                ctx_vectors = embedding[0, 1: num_nonzero_token , :]
                prompt_prefix = ctx_init
            else:
                prompt_prefix = []
                ctx_vectors = []
                for ctx_init_i in ctx_init: 
                    # use given words to initialize context vectors
                    ctx_init_i = ctx_init_i.replace("_", " ")
                    # tokens_i = tokenizer(ctx_init_i, return_tensors='pt')
                    tokens_i = tokenizer(ctx_init_i, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
                    prompt_i = tokens_i['input_ids'].to(embedding_device)
                    attention_mask_i = tokens_i['attention_mask'].to(
                        embedding_device)
                    # prompt_i = torch.from_numpy(np.array(prompt_i))
                    with torch.no_grad():
                        embedding_i = clip_model.text_model.embeddings(prompt_i).type(dtype)
                    num_nonzero_token = prompt_i.nonzero().max()
                    idx_special_character_i = torch.where(prompt_i == 343)[1]
                    if all_ctx_trainable: 
                        ctx_vectors_i = embedding_i[0, 1: num_nonzero_token, :]  # keep ctx between SOS and EOS
                    else:
                        ctx_vectors_i = embedding_i[0, idx_special_character_i, :]
                    prompt_prefix_i = ctx_init_i
                    prompt_prefix.append(prompt_prefix_i)
                    ctx_vectors.append(ctx_vectors_i)

        else:
            # random initialization
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        if not self.use_class_specific_ctx:
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        else:
            self.ctx = nn.ParameterList([nn.Parameter(ctx_vector) for ctx_vector in ctx_vectors])

        # classnames = [name.replace("_", " ") for name in classnames]
        # # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        if not self.use_class_specific_ctx:
            prompts = [prompt_prefix + " " + name for name in text_ctx]
        else:
            prompts = prompt_prefix 
        
        tokens_dict = tokenizer(prompts, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_prompts = tokens_dict['input_ids'].to(embedding_device)
        attention_mask = tokens_dict['attention_mask'].to(embedding_device)
        # tokenized_prompts = torch.from_numpy(np.array(tokenized_prompts)).cuda()
        # attention_mask = torch.from_numpy(np.array(attention_mask)).cuda()
        with torch.no_grad():
            embedding = clip_model.text_model.embeddings(tokenized_prompts).type(dtype)
        causal_attention_mask = _create_4d_causal_attention_mask(tokenized_prompts.size(), embedding.dtype, device=embedding.device)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, embedding.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        if not self.use_class_specific_ctx:
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS 
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS 
            self.n_ctx = n_ctx
        else:
            if all_ctx_trainable:
                self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            else:
                
                for i in range(self.n_prompts):
                    special_character_pos = torch.where(tokenized_prompts[i] == 343)[0].min()  # find first character "*"
                    self.register_buffer("token_prefix_{}".format(i), embedding[i, :special_character_pos, :])
            for i in range(self.n_prompts):
                CLS_pos = torch.where(tokenized_prompts[i] == 343)[0].max()  # find last character "*"
                self.register_buffer("token_suffix_{}".format(i), embedding[i, CLS_pos+1:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        
        self.causal_attention_mask = causal_attention_mask
        self.attention_mask = attention_mask 
        # self.name_lens = name_lens

        self.drop_layer = torch.nn.Dropout(p=p_drop_out)

    def forward(self):
        if not self.use_class_specific_ctx:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_prompts, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            ctx = self.ctx
            prompts = []
            for i in range(self.n_prompts):
                if self.all_ctx_trainable:
                    prompt_i = torch.cat(
                        [
                            self.token_prefix[i].unsqueeze(0),  # (n_cls, 1, dim)
                            ctx[i].unsqueeze(0),  # (n_cls, n_ctx, dim)
                            getattr(self, "token_suffix_{}".format(i)).unsqueeze(0)  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                else:
                    prompt_i = torch.cat(
                        [
                            getattr(self, "token_prefix_{}".format(i)).unsqueeze(0),  # (n_cls, 1, dim)
                            ctx[i].unsqueeze(0),  # (n_cls, n_ctx, dim)
                            getattr(self, "token_suffix_{}".format(i)).unsqueeze(0)  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                prompts.append(prompt_i)
            prompts = torch.cat(prompts, dim=0)
            prompts = self.drop_layer(prompts)
        return prompts
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.num_layers = clip_model.text_model.encoder.layers # clip_model.transformer.resblocks
        self.positional_embedding = clip_model.text_model.embeddings.position_embedding #clip_model.positional_embedding
        self.final_layer_norm =  clip_model.text_model.final_layer_norm #clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = self.final_layer_norm.weight.dtype

    def forward(self, x, tokenized_prompts, attn, causal_attention_mask):
        attn, causal_attention_mask = attn.to(x.device), causal_attention_mask.to(x.device)
        for layer_idx, layer in enumerate(self.num_layers):
            x = layer(x, attn, causal_attention_mask)[0]

        x = self.final_layer_norm(x)
        x = x[
                torch.arange(x.shape[0], device=x.device),
                tokenized_prompts.to(dtype=torch.int, device=x.device).argmax(dim=-1),
            ]
        x = self.text_projection(x)

        return x

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, x, adj):
        # x: [B, N, in_dim], adj: [B, N, N]
        B, N, _ = x.size()
        H = self.num_heads
        x_proj = self.linear(x).view(B, N, H, self.out_dim)  # [B, N, H, out_dim]

        # Compute attention scores
        a_l = (x_proj * self.attn_l).sum(dim=-1)  # [B, N, H]
        a_r = (x_proj * self.attn_r).sum(dim=-1)  # [B, N, H]

        a_l = a_l.unsqueeze(2)  # [B, N, 1, H]
        a_r = a_r.unsqueeze(1)  # [B, 1, N, H]
        attn_scores = a_l + a_r  # [B, N, N, H]
        attn_scores = F.leaky_relu(attn_scores)

        # Masked attention: only compute for connected nodes
        mask = adj.unsqueeze(-1)  # [B, N, N, 1]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=2)  # [B, N, N, H]
        attn_probs = self.dropout(attn_probs)

        # Attention-weighted sum
        x_proj = x_proj.permute(0, 2, 1, 3)  # [B, H, N, out_dim]
        attn_probs = attn_probs.permute(0, 3, 1, 2)  # [B, H, N, N]
        out = torch.matmul(attn_probs, x_proj)  # [B, H, N, out_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, H * self.out_dim)  # [B, N, H*out_dim]

        return F.elu(out)

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(GATLayer(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x  # [B, N, out_dim]


def _entity_major_attribute_view(attrs_f, n_cls, num_entities):
    """Restore the entity-major order produced by ``PromptLearner``.

    PromptLearner appends every class attribute for entity 0, then every class
    attribute for entity 1, and so on. The released reshape treated that tensor
    as class-major, silently pairing most entities with another entity's class
    descriptions.
    """
    expected = n_cls * num_entities
    if attrs_f.ndim != 2 or attrs_f.shape[0] != expected:
        raise ValueError(
            "MAPLE attribute prompt tensor has incompatible shape: "
            f"got {list(attrs_f.shape)}, expected [{expected}, D]")
    return attrs_f.reshape(num_entities, n_cls, attrs_f.shape[1])


class MAPLE(nn.Module):
    def __init__(self, args, clip_model=None, tokenizer=None):
        super().__init__()
        self.weight = args.weight
        self.pos_ratio = args.pos_ratio
        self.neg_ratio = args.neg_ratio
        
        # base_model_name = 'clip'
        if clip_model is None:
            clip_model = create_model(model_name=args.base_model)
        if tokenizer is None:
            tokenizer = load_pretrained_tokenizer(args.base_model)
        # freaze all parmeters
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip_visual_proj = clip_model.visual_projection # project to VL space, form 768 to 512

        self.prompt_learner_attr = PromptLearner( 
                    args.text_path,
                    clip_model, 
                    tokenizer, 
                    mode = 'attribute',
                    n_ctx=args.attr_n_ctx,
                    attr_edge_topk=args.attr_edge_topk,
                    all_ctx_trainable=args.all_ctx_trainable, 
                    csc=args.csc, 
                    # classnames=["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"], 
                    p_drop_out=args.p_drop_out)
        
        self.prompt_learner_reg = PromptLearner( 
                    args.text_path,
                    clip_model, 
                    tokenizer, 
                    mode = 'region',
                    n_ctx=args.attr_n_ctx,
                    all_ctx_trainable=args.all_ctx_trainable, 
                    csc=args.csc, 
                    # classnames=["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"], 
                    p_drop_out=args.p_bag_drop_out) 
            
        self.prompt_learner_global = PromptLearner(
                    args.text_path,
                    clip_model,
                    tokenizer,
                    mode = 'global',
                    n_ctx=args.bagLevel_n_ctx,
                    all_ctx_trainable=args.all_ctx_trainable,
                    csc=args.csc,
                    p_drop_out=args.p_bag_drop_out)

        self.text_encoder = TextEncoder(clip_model) 
        for param in self.text_encoder.parameters(): 
            param.requires_grad = False 

        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad = False
        self.dtype = clip_model.dtype

        self.input_dim = clip_model.visual_projection.weight.shape[0]
        self.cross_attention = MultiheadAttention(embed_dim=self.input_dim, num_heads=1)

        self.graph_dim = clip_model.visual_projection.weight.shape[0]
        self.graph_learner = GNNEncoder(in_dim=self.graph_dim, hidden_dim=self.graph_dim, out_dim=self.graph_dim, num_layers=1)

        self.input_size = self.clip_visual_proj.weight.shape[0]
        self.L = self.input_size
        self.D = self.input_size
        self.K = 1
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        self.norm = nn.LayerNorm(self.input_size)

        self.loss_ce = nn.CrossEntropyLoss()

    def get_topk_indices(self, x, positive_f):
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("MAPLE requires a non-empty rank-2 patch bag")
        x_norm = F.normalize(x, dim=-1)
        positive_f_norm = F.normalize(positive_f, dim=-1)
        positive_sim = self.logit_scale.exp() * x_norm @ positive_f_norm.t()

        positive_sim = positive_sim.max(dim=-1)[0]

        pos_top_k = max(
            1, min(len(x_norm), int(self.pos_ratio * len(x_norm))))
        pos_indices = torch.topk(positive_sim, pos_top_k, dim=-1).indices
        
        return pos_indices

    def obtain_entities_attr(self, x_pos, attr_proto_f, attrs_f, n_cls, num_entities):
        # attr_proto_f: num_entities x dim
        entities_feature, _ = self.cross_attention(attr_proto_f.unsqueeze(1), x_pos, x_pos)
        entities_feature = self.norm(entities_feature + attr_proto_f.unsqueeze(1))
        entities_feature = entities_feature.squeeze(1)

        entities_feature_norm = entities_feature / entities_feature.norm(dim=-1, keepdim=True) # num_entities x dim
        attrs_f_norm = attrs_f / attrs_f.norm(dim=-1, keepdim=True) # numcls*num_entities x dim

        attrs_f_norm = _entity_major_attribute_view(
            attrs_f_norm, n_cls, num_entities)
        # entities_feature_norm = entities_feature_norm[:, None, :] # num_entities x 1 x dim
        
        if self.weight > 0.:
            entities_feature_norm = entities_feature_norm.detach()

        attr_score = self.logit_scale.exp() * torch.einsum('ncd,nd->nc', attrs_f_norm, entities_feature_norm)
        # attr_score = torch.matmul(attrs_f, entities_feature.permute(0,2,1)).squeeze(-1)

        return entities_feature, attr_score

    def build_graph(self, attr_edge, entities_fl, entities_fh):
        """
        entities_fl: (Kl, dim)
        entities_fh: (Kh, dim)
        global_proto_fl/fh: (1, dim)

        Returns:
            feat: (1, N, dim)
            adj:  (1, N, N)
        """
        Kl, dim = entities_fl.shape
        Kh, dim = entities_fh.shape
        device = entities_fl.device

        # concat features
        feat = torch.cat([entities_fl, entities_fh], dim=0)  # (Kl+Kh, dim)

        N = Kl + Kh

        # adjacency
        adj = torch.zeros((N, N), device=device)
        adj[:N, :N] = attr_edge

        adj.fill_diagonal_(1.0)

        feat = feat.unsqueeze(0)  # [1, N+2, dim]
        adj = adj.unsqueeze(0)    # [1, N+2, N+2]

        return feat, adj

    def forward(self, x_l, coords_l, x_h, coord_h, label, return_features=False):

        # x_l = self.clip_visual_proj(x_l) # image: N x dim
        # x_h = self.clip_visual_proj(x_h) # image: N x dim

        # prompt attr
        prompts_attr = self.prompt_learner_attr()
        text_features_attr = self.text_encoder(prompts_attr, self.prompt_learner_attr.tokenized_prompts, self.prompt_learner_attr.attention_mask, self.prompt_learner_attr.causal_attention_mask)
        attr_proto_fl, attrs_fl, attr_proto_fh, attrs_fh = torch.split(text_features_attr, self.prompt_learner_attr.num_list)

        # prompt region 
        prompts_reg = self.prompt_learner_reg()
        text_features_reg = self.text_encoder(prompts_reg, self.prompt_learner_reg.tokenized_prompts, self.prompt_learner_reg.attention_mask, self.prompt_learner_reg.causal_attention_mask)
        positive_fl, positive_fh = torch.split(text_features_reg, self.prompt_learner_reg.num_list)

        # prompt global
        prompts_global = self.prompt_learner_global()
        text_features_global = self.text_encoder(prompts_global, self.prompt_learner_global.tokenized_prompts, self.prompt_learner_global.attention_mask, self.prompt_learner_global.causal_attention_mask)
        global_fl, global_proto_fl, global_fh, global_proto_fh = torch.split(text_features_global, self.prompt_learner_global.num_list)

        # slected 
        n_cls = self.prompt_learner_global.num_list[0]
        num_entities_low, num_entities_high = attr_proto_fl.shape[0], attr_proto_fh.shape[0]
        
        pos_indices = self.get_topk_indices(x_l, positive_fl)
        x_l_pos = x_l[pos_indices, :]
        
        pos_indices = self.get_topk_indices(x_h, positive_fh)
        x_h_pos = x_h[pos_indices, :]

        entities_fl, attr_sl = self.obtain_entities_attr(x_l_pos, attr_proto_fl, attrs_fl, n_cls, num_entities_low)
        entities_fh, attr_sh = self.obtain_entities_attr(x_h_pos, attr_proto_fh, attrs_fh, n_cls, num_entities_high)

        # # # attr similarity
        entities_score_all = torch.cat([attr_sl, attr_sh], dim=0)
        entities_logits = torch.mean(entities_score_all, dim=0, keepdim=True)

        loss_attr = self.loss_ce(entities_score_all, label.expand(len(entities_score_all)))
        # loss_attr = self.loss_ce(entities_logits, label)
        if self.weight <= 0.:
            loss = loss_attr 
            Y_prob = F.softmax(entities_logits, dim = 1)
            Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
            return Y_prob, Y_hat, loss

        # entities graph         # entities_fl, entities_fh, K x dim global_proto_fl, global_proto_fh 1xdim

        feat, adj = self.build_graph(self.prompt_learner_attr.attr_edge, entities_fl, entities_fh)
        graph_feat = self.graph_learner(feat, adj) 
        # # Split back to low / high level
        entities_fl, entities_fh = torch.split(graph_feat, [entities_fl.shape[0], entities_fh.shape[0]], dim=1)
        
        # Remove only the synthetic graph-batch dimension. ``squeeze()``
        # also removed the entity axis for one-entity task prompts (for
        # example CAMELYON16), leaving a 1-D tensor for the attention head.
        H_l = entities_fl.squeeze(0).float()
        A_V_l = self.attention_V(H_l)  
        A_U_l = self.attention_U(H_l)  
        A_l = self.attention_weights(A_V_l * A_U_l) 
        A_l = torch.transpose(A_l, 1, 0)  
        A_l = F.softmax(A_l, dim=1)  
        graph_global_fl = torch.mm(A_l, H_l)  

        H_h = entities_fh.squeeze(0).float()
        A_V_h = self.attention_V(H_h)  
        A_U_h = self.attention_U(H_h)  
        A_h = self.attention_weights(A_V_h * A_U_h) 
        A_h = torch.transpose(A_h, 1, 0)  
        A_h = F.softmax(A_h, dim=1)  
        graph_global_fh = torch.mm(A_h, H_h)  


        # # graph_global_fl, graph_global_fh = self.norm(graph_global_fl), self.norm(graph_global_fh)
        graph_global_fl_norm, graph_global_fh_norm = F.normalize(graph_global_fl, dim=-1), F.normalize(graph_global_fh, dim=-1)
        global_fl_norm, global_fh_norm = F.normalize(global_fl, dim=-1), F.normalize(global_fh, dim=-1)
        
        golbal_fl_sim = self.logit_scale.exp() * graph_global_fl_norm @ global_fl_norm.t()
        global_fh_sim = self.logit_scale.exp() * graph_global_fh_norm @ global_fh_norm.t()


        global_logits = (golbal_fl_sim + global_fh_sim) / 2
        logits = self.weight * global_logits + (1 - self.weight) * entities_logits

        loss = self.loss_ce(logits, label)

        # if (1 - self.weight) > 0.:
        #     loss_global = self.loss_ce(global_logits, label)
        #     loss = loss + loss_global + loss_attr 

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]

        return Y_prob, Y_hat, loss
