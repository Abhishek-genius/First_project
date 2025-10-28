This project implements a robust face recognition system that detects, aligns, and matches faces using deep learning-based embeddings.
It also integrates synthetic data generation pipelines using Pix2Pix and Gemini/CLIP models to enhance dataset diversity and improve recognition performance.

Features
Face detection and alignment using InsightFace (RetinaFace)
Face embedding extraction via ArcFace model
Fast similarity search using FAISS and ChromaDB
Robust against rotations, lighting changes, and occlusions
Synthetic dataset generation using:

Pix2Pix (Image-to-Image Translation)

Gemini/CLIP (Text-to-Image Generation)
Cache-based storage for faster startup
Configurable thresholds and search parameters

System Architecture
[Input Image]
     ↓
[Face Detection → Alignment → Embedding Extraction]
     ↓
[Embedding Storage (ChromaDB + Cache)]
     ↓
[Similarity Search (FAISS Index)]
     ↓
[Top-k Matches + Confidence Scores]


Dataset
Real dataset: LFW (Labeled Faces in the Wild) or custom dataset.

Synthetic data:

Pix2Pix-generated images: structured → realistic transformations.

Gemini/CLIP-generated images: text-guided, semantically rich samples.

Combined datasets improve both realism and semantic diversity.


