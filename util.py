def compute_similarity(image_features, prompt_features):
    return (
        (100.0 * image_features @ prompt_features.T)
        .softmax(dim=-1)
        .detach()
        .cpu()
        .numpy()
    )
