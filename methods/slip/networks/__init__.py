
def get_clip_network(model_name : str):
    import clip

    if model_name == "CLIP":
        model, embed_dim, preprocess = clip.load("ViT-B/16")
        preprocess_train = preprocess_test = preprocess
        tokenizer = clip.tokenize

    elif model_name == "CLIP-RN50":
        model, embed_dim, preprocess = clip.load("RN50")
        preprocess_train = preprocess_test = preprocess
        tokenizer = clip.tokenize

    elif model_name == "PLIP":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("vinid/plip")
        preprocess = CLIPProcessor.from_pretrained("vinid/plip")
        def preprocess_image(image):
            return preprocess.image_processor(
                image, return_tensors="pt")["pixel_values"][0]
        def tokenize_text(text):
            return preprocess.tokenizer(
                text, return_tensors="pt", padding=True)["input_ids"]
        preprocess_train = preprocess_test = preprocess_image
        tokenizer = tokenize_text
        embed_dim = 768
        model = model.cuda()

    elif model_name == "BiomedCLIP":
        import open_clip
        model, preprocess_train, preprocess_test = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', precision='fp32')
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # print(model.state_dict().keys())
        # print(model)
        embed_dim = 512 #model.state_dict()["text_projection"].shape[1]
        model = model.cuda()
    else:
        raise NotImplementedError
    
    return model, embed_dim, tokenizer, preprocess_train, preprocess_test

def get_ctranspath(save_path):
    from .ctranspath import ctranspath
    return ctranspath(save_path)
