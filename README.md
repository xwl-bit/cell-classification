# cell-classification


## Overview

This repository implements a deep learning-based model for gene expression analysis using a Transformer architecture known as **Performer**. The accurate identification of human cell subtypes and understanding their interactions have long been central to the field of biology. Recent breakthroughs in single-cell RNA sequencing (scRNA-seq) technology have produced vast amounts of high-dimensional molecular data, offering new possibilities for classifying cell subtypes. Despite these advancements, the process of annotating cell types using scRNA-seq data remains largely manual. The challenge lies in determining which cell types can be accurately inferred from these complex molecular features.

## Features

- **Performer Architecture**: A Transformer model that uses linearized attention mechanisms to handle large-scale genomic data.
- **Data Masking and Tokenization**: Preprocessing includes gene tokenization and masking, similar to BERT's MLM (Masked Language Modeling).
- **Cross-validation**: The model supports training on split datasets for validation and testing.
- **Fine-tuning**: The model can be fine-tuned on pre-trained weights, enabling transfer learning for specific tasks like cell type classification.

## Installation

Ensure that you have the necessary dependencies installed. You can install the required packages using:

```bash
pip install -r requirements.txt
```

The necessary libraries include:

- **PyTorch**: For building and training the neural network.
- **Scanpy**: For working with single-cell RNA-seq data.
- **NumPy**, **SciPy**: For numerical operations.
- **pandas**: For data manipulation.
- **scikit-learn**: For dataset splitting and performance metrics.
- **Apex**: For mixed-precision training (optional).

### Requirements

- Python 3.6+
- CUDA-enabled GPU for faster training
- `torch`, `torchvision`, `scikit-learn`, `scanpy`, and `apex` (optional for mixed-precision support)

## Model Description

The Performer model uses **linear attention**, which is an approximation to the regular attention mechanism of Transformers. This approach uses random features to compute the attention weights, reducing memory and computational complexity from quadratic to linear in the sequence length. The code contains various layers like **FeedForward**, **SelfAttention**, **ReZero**, and **LayerNorm** that are applied to the input data.

The project contains two main models:

- **PerformerLM**: A model for sequence-based tasks like gene expression analysis.
- **Identity Model**: A classifier head added on top of the **Performer** architecture for downstream classification tasks.

## Dataset

The dataset used in this implementation is **single-cell RNA-seq** data. The data is split into training and validation sets for model evaluation.

### Preprocessing

1. **Gene Tokenization**: The dataset is tokenized such that each gene expression is represented as a token.
2. **Data Masking**: Genes are randomly masked during training, and the model learns to predict the missing genes using **Masked Language Modeling (MLM)**.
3. **Gene2Vec Encoding**: Positional embeddings are provided using a Gene2Vec encoding, which maps each gene into a high-dimensional vector.

## Training

To train the model, use the following command:

```bash
python train.py --data_path <path_to_data> --epoch <number_of_epochs> --batch_size <batch_size> --learning_rate <learning_rate>
```

### Command Line Arguments

- `--bin_num`: Number of bins for classification.
- `--gene_num`: Number of genes in the dataset.
- `--epoch`: Number of epochs for training.
- `--batch_size`: The batch size for training.
- `--learning_rate`: The learning rate for the optimizer.
- `--seed`: Random seed for reproducibility.
- `--data_path`: Path to the input dataset.
- `--mask_prob`: Probability of masking tokens.
- `--replace_prob`: Probability of replacing tokens with `[MASK]`.
- `--pos_embed`: Whether to use Gene2Vec positional embedding.

## Evaluation

After training, you can evaluate the model on a test set. Use the following command to perform evaluation:

```bash
python evaluate.py --model_path <path_to_trained_model> --data_path <path_to_test_data>
```

This will output the accuracy and F1 score of the model on the test dataset.

## File Structure

```plaintext
.
├── model.py              # Model architecture (Performer, Attention, FeedForward, etc.)
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── utils.py              # Utility functions for data processing and metrics
├── data/
│   └── train.h5ad       # Input dataset
├── ckpts/               # Directory to save model checkpoints
├── README.md            # Project README
└── requirements.txt      # Python dependencies
```



## License

This project is licensed under the MIT License. 

