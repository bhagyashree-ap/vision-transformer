## Vision Transformer (ViT) Image Classification

* To classify clothing images automatically.
* Dataset: FashionMNIST
* Input: Grayscale images, 28 × 28 pixels
* Output: One of 10 clothing classes
* Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
* Model: Vision Transformer (ViT)
* Framework: PyTorch
* Main goal: Compare 30 model configurations and select the best.
* Core idea: Split images into patches and use self-attention to learn visual patterns.

## Dataset and Preprocessing

* Total images: 70,000
* Training: 54,000 images
* Validation: 6,000 images
* Testing: 10,000 images
* Image size: 28 × 28 pixels
* Channels: 1 (grayscale)
* Classes: 10
* Normalization: Mean = 0.5, standard deviation = 0.5
* Batch size: 256
* Random seed: 42
* Purpose of training data: Learn model parameters.
* Purpose of validation data: Select the best model.
* Purpose of test data: Measure final performance.
* Data loading: PyTorch DataLoader.
* Reproducibility: Fixed random seed and saved dataset split indices.

## Model Architecture and Training

* Patch sizes: 4 × 4 and 7 × 7
* Patches per image: 49 or 16
* Patch embedding: Flatten patches and project them into 96-dimensional vectors.
* Embedding dimension: 96
* Class token: One learnable token per image.
* Positional embeddings: Preserve patch position information.
* Token sequence length: 50 or 17, including the class token.
* Transformer layers: 4, 5, 6, 7, 8
* Attention heads: 2, 3, 4
* Feed-forward dimension: 384
* Activation function: ReLU
* Layer normalization: Applied before attention and feed-forward blocks.
* Residual connections: Preserve information across Transformer blocks.
* Dropout: 0.1
* Classification head: LayerNorm followed by a linear layer (96 → 10).
* Classification outputs: 10 class scores.
* Optimizer: Adam
* Learning rate: 0.001
* Weight decay: 0.0001
* Loss function: Cross-entropy
* Training epochs: 20 per configuration
* Training process: Forward pass, loss calculation, backpropagation and parameter updates.
* Total configurations: 2 × 5 × 3 = 30
* Maximum total training epochs: 600 across all configurations.

## Workflow

1. Load the FashionMNIST dataset.
2. Normalize images and split the dataset.
3. Input a grayscale image.
4. Divide the image into non-overlapping patches.
5. Convert patches into embeddings.
6. Add positional embeddings and the class token.
7. Apply Transformer encoder layers.
8. Extract the class token representation.
9. Generate 10 class scores.
10. Calculate cross entropy loss and update model parameters.
11. Evaluate validation loss and accuracy after each epoch.
12. Save the best checkpoint for each configuration.
13. Compare all configurations and select the best model.
14. Evaluate the selected model on the test dataset.

## Evaluation, Results and Outputs

* Training metrics: Loss and accuracy.
* Validation metrics: Loss and accuracy.
* Test metrics: Loss and accuracy.
* Model comparison: Evaluate 30 configurations.
* Best model: Highest validation accuracy.
* Tie-breaker: Lowest validation loss.
* Checkpoints: Save trained model states.
* Training history: Record per-epoch metrics.
* CSV files: Store experiment results and selected-model information.
* Comparison table: Compare patch sizes, Transformer depths and attention heads.
* Accuracy plot: Show accuracy against Transformer depth.
* Training curves: Show training and validation loss and accuracy.
* Overfitting analysis: Compare training and validation performance.
* Final evaluation: Use test performance to report the selected model’s generalization.
* Libraries: PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Einops.
* Hardware: NVIDIA T4 GPU (Google Colab).
