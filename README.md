# intel_vllm_project
# CLIP-based Visual Search Engine

## Project Overview  
This project is a *Visual Search Engine* powered by *CLIP (Contrastive Language–Image Pretraining)*. It allows users to:

- Search images using *text descriptions*
- Find similar images using an *image input*

We used:
- *CLIP (ViT-B/32)* from HuggingFace Transformers  
- *FAISS* for similarity search  
- *Gradio* for the user interface  

---

## Dataset Used: Flickr8k  
We worked with the *Flickr8k Dataset*, which contains:

- Approximately 8,000 images  
- 5 captions per image → ~40,000 image-caption pairs  
- Caption file: Flickr8k.token.txt  

---

## Embedding Generation (Jupyter Notebook)  
Due to hardware limits, we generated image and text embeddings on *local Jupyter Notebook* in *batches*.

### Batch Processing Logic  
- *Batch size:* 500 image-caption pairs  
- Each batch creates:
  - image_embeddings_batch_X.pt
  - text_embeddings_batch_X.pt  
- A total of *60 batches* was generated (≈30,000 embeddings)

This batching strategy helps:
- Handle large datasets without GPU memory overflow  
- Resume from a failed batch easily (e.g., start from batch 9 if interrupted)

---

## FAISS Indexing & Visual Search (Google Colab)

Once all embeddings were saved, we switched to *Google Colab* for building the visual search engine.

### Steps in Colab
- Upload all *.pt embedding files and zipped images to Colab  
- Load image-caption pairs from the same order (unsorted)  
- Concatenate batches using torch.cat()  
- Build FAISS index from image embeddings  
- Implement search:
  - *Text → Image* using CLIP.text_encoder  
  - *Image → Image* using CLIP.image_encoder  

```python
# Example for FAISS indexing
import faiss
index = faiss.IndexFlatL2(512)
index.add(image_embeddings.numpy())
```
## Features

- Text-to-Image Search  
- Image-to-Image Search  
- Shows Top 10 closest matches  
- Clean Gradio Interface  
- Robust design using pre-generated embedding batches
  
> NOTE:  due to hardware and time constraints i wasnt able to create embeddings for all 8k imgs hences its a bit messy with its results 
> STATUS: Yet to re start the training.
