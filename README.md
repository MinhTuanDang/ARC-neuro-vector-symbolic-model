# ARC-neuro-vector-symbolic-model
This repository contains a Google Colab–runnable PyTorch implementation of a modified LARS‑VSA model for training on ARC (Abstraction and Reasoning Corpus) tasks. The model is inspired by the vector symbolic architecture described in recent literature and is adapted for sequence-to-sequence grid transformation tasks.

Overview
The project demonstrates how to:

Load ARC tasks in JSON format (compatible with the ARC‑AGI dataset format).
Preprocess each ARC grid by padding it to a fixed size (30×30) and flattening it.
Convert grid cells (tokens) into embeddings.
Train a multi-head self-attention model (LARS‑VSA–inspired) that outputs a prediction for each cell.
Evaluate the model and visualize the predicted output grid.
Directory Structure
javascript
Copy
Edit
.
├── README.md
└── sample_data
    └── train
        ├── task1.json
        ├── task2.json
        └── ... (other ARC JSON files)
Note: Place your ARC JSON files (each following the ARC‑AGI format) into the sample_data/train folder.

ARC JSON Format
Each JSON file should follow the ARC-AGI format:

train: A list of demonstration input/output pairs.
test: A list of test input/output pairs.
Each pair is a dictionary with:

"input": A grid (list of lists) of integers (0–9).
"output": A grid (list of lists) of integers (0–9).
Model Architecture
The model consists of:

An embedding layer converting token indices (colors 0–9) into vectors.
A multi-head HDSymbolicAttention module that projects embedded tokens into a high-dimensional (bipolar) space using a learnable projection.
A custom binary activation and bundling operation to simulate hyperdimensional symbolic processing.
A token-wise classifier that predicts a color for each cell in the flattened grid.
Requirements
Python 3.x
PyTorch
Google Colab (or any environment that supports GPU acceleration)
How to Run in Google Colab
Upload the Data:

Place your ARC JSON files into a folder named train inside the sample_data directory. In Colab, you can either upload these files manually or mount your Google Drive.
Run the Notebook:

Copy the provided code into a Colab notebook cell and run it.
The code automatically detects if a GPU is available.
Training and evaluation logs will be printed, and an example predicted output grid will be displayed.
Code Explanation
Data Loading:
The ARCDataset class reads each JSON file from ./sample_data/train, pads each grid to 30×30, flattens it, and stores input/output pairs as tensors.

Model Definition:
The LARS‑VSA model uses a multi-head attention mechanism with custom binary activation to simulate hyperdimensional symbolic processing. The final token-wise classifier outputs a prediction for each of the 900 (30×30) tokens.

Training and Evaluation:
The training loop uses CrossEntropyLoss computed over all tokens in each example. The evaluation loop reports loss and an example prediction is reshaped back into a grid format for visualization.
