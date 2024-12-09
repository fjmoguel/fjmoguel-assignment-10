import torch
from sklearn.decomposition import PCA
from PIL import Image
import pandas as pd
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F

def initialize_model():
    """
    Initialize the model, preprocess function, and embeddings.
    """
    model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
    model.eval()
    df = pd.read_pickle('image_embeddings.pickle')
    embeddings = torch.tensor([row['embedding'] for _, row in df.iterrows()])
    return model, preprocess, embeddings, df

def search(query_embedding, embeddings, df, top_k=5, use_pca=False, n_components=5):
    """
    Perform a search for the most similar images based on embeddings.
    Optionally applies PCA for dimensionality reduction.
    """
    if use_pca:
        embeddings_np = embeddings.detach().numpy()
        query_np = query_embedding.detach().numpy()

        pca = PCA(n_components=min(n_components, embeddings_np.shape[1]))
        reduced_embeddings = pca.fit_transform(embeddings_np)
        reduced_query = pca.transform(query_np)

        reduced_embeddings = torch.tensor(reduced_embeddings)
        reduced_query = torch.tensor(reduced_query)
        cos_similarities = torch.mm(reduced_query, reduced_embeddings.T)
    else:
        cos_similarities = torch.mm(query_embedding, embeddings.T)

    top_k_indices = torch.topk(cos_similarities, top_k, dim=1).indices[0]
    results = []
    for idx in top_k_indices:
        file_name = df.iloc[idx.item()]['file_name']
        score = cos_similarities[0, idx].item()
        results.append((f"/coco_images_resized/{file_name}", score))
    return results

def preprocess_image(image_path, preprocess, model):
    """
    Preprocess an image and generate its embedding.
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    return F.normalize(model.encode_image(image))

def tokenize_text(text_query, model):
    """
    Tokenize text and generate its embedding.
    """
    text_tokens = tokenizer.tokenize([text_query]).to("cpu")
    return F.normalize(model.encode_text(text_tokens))