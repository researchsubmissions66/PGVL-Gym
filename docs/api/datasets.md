# Feature-loader API

## Patch bags

::: common.datasets.bag_features
    options:
      members:
        - BagFeaturesDataset
        - build_bag_loader

## Slide embeddings

::: common.datasets.slide_embeddings
    options:
      members:
        - normalise_slide_id
        - infer_slide_embedding_source_type
        - SlideEmbeddingSource
        - SlideEmbeddingDataset
        - split_csv
        - build_slide_embedding_loader
