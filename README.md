# Adapting CLIP for image classification and image–text retrieval

NUS IT5005 · Group 7

How much can a pretrained vision–language model improve when most of its weights stay frozen? We tested CLIP ViT-B/32 on two tasks. A classifier trained on frozen image embeddings raised ImageNet-1k test accuracy from **59.44% to 69.33%**. On Flickr8k, tuning the projection layers and adding rank-2 LoRA adapters raised mean test recall from **0.7424 to 0.8107** with **0.47%** of parameters trainable.

## 1. Image classification on ImageNet-1k

We split the 50,000 images from the ImageNet-1k validation set into 35,000 training, 5,000 validation and 10,000 test images, stratified across 1,000 classes. We used the validation set to select the zero-shot prompt and the classifier checkpoint. The accuracy figures below use the same held-out 10,000 images.

| Method | Test accuracy |
| --- | ---: |
| Zero-shot CLIP, cosine similarity with the prompt `an image of a {}` | 59.44% |
| Residual classifier trained on frozen CLIP image embeddings | 69.33% |

The trained classifier gained **9.89 percentage points** and reached a test macro-F1 of **0.6866**. The notebook also compares eight prompt templates and tests how normalising embeddings changes zero-shot predictions. [Classification notebook](Part_1.ipynb)

## 2. Image–text retrieval on Flickr8k

Given an image, can CLIP retrieve its caption? Given a caption, can it find the image? We evaluated both directions on 1,091 held-out image–caption pairs after using 6,000 pairs for training and 1,000 for validation.

| Method | Image → text R@1 | Text → image R@1 | Mean recall |
| --- | ---: | ---: | ---: |
| Zero-shot CLIP | 0.5298 | 0.5380 | 0.7424 |
| Projection layers, triplet loss | 0.5857 | 0.5885 | 0.7874 |
| Projection layers, InfoNCE loss | 0.6169 | 0.5811 | 0.7994 |
| Projection layers + rank-2 text LoRA, merged | **0.6288** | **0.5958** | **0.8107** |

R@1 is the fraction of queries whose matching item ranked first. Mean recall averages Recall@1, @5 and @10 across both directions. The best model trained 655,360 projection weights and 49,152 adapter parameters in the text encoder’s query and value projections. Section 9.7 contains the full test table, and section 9.8 shows example retrievals. [Retrieval notebook](Part_2.ipynb)

## Run the notebooks

From the repository root, create a Python 3.12 environment and launch JupyterLab:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r project_requirements.txt
jupyter lab
```

Open either notebook and run its cells in order. On the first run, the notebooks download model weights and data from Hugging Face and cache features in `cache/` (Part 1) or `cache_part2/` (Part 2). Keep the [ImageNet label map](id_to_label.json) alongside Part 1. Both notebooks select CUDA, Apple MPS or CPU according to availability. The saved outputs can also be read directly on GitHub.

## Course materials and contributions

The [project brief](IT5005_Project_August_2026.pdf) describes the assignment and evaluation requirements. For group contributions, use a separate branch and open a pull request. Coordinate edits to shared notebook cells, record generative AI assistance in the notebook’s AI Usage section, and keep datasets, weights and local caches out of Git.
