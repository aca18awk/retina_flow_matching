import os

from distance_ImageNet_normalised import generate_ImageNet_Embeddings
from distance_RetFound_DinoV2 import generate_DINO_embeddings
from distance_RetFound_MAE import generate_MAE_embeddings

if __name__ == "__main__":
    SPLIT = "hidden"
    datasetName = "Messidor"

    savedir = f"embeddings/{datasetName}/{SPLIT}"
    os.makedirs(savedir, exist_ok=True)

    generate_MAE_embeddings(SPLIT, datasetName, useRetFoundPreprocessing=False, savedir=savedir)
    generate_MAE_embeddings(SPLIT, datasetName, useRetFoundPreprocessing=True, savedir=savedir)

    generate_DINO_embeddings(SPLIT, datasetName, useRetFoundPreprocessing=False, savedir=savedir)
    generate_DINO_embeddings(SPLIT, datasetName, useRetFoundPreprocessing=True, savedir=savedir)

    generate_ImageNet_Embeddings(
        SPLIT, datasetName, SAME_LABEL=True, useRetFoundPreprocessing=True, savedir=savedir
    )
    generate_ImageNet_Embeddings(
        SPLIT, datasetName, SAME_LABEL=True, useRetFoundPreprocessing=False, savedir=savedir
    )
    generate_ImageNet_Embeddings(
        SPLIT, datasetName, SAME_LABEL=False, useRetFoundPreprocessing=True, savedir=savedir
    )
    generate_ImageNet_Embeddings(
        SPLIT, datasetName, SAME_LABEL=False, useRetFoundPreprocessing=False, savedir=savedir
    )
