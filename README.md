## Vision Transformer (ViT) Image Classification

* To classify clothing images automatically
* Dataset: FashionMNIST
* Input: Grayscale images, 28 × 28 pixels
* Output: 1 of 10 clothing classes
* Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
* Model: Vision Transformer (ViT)
* Framework: PyTorch
* Main goal: Compare 30 model configurations and select the best.
* Core idea: Split images into patches and use attention to learn visual patterns.

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

## Model Architecture and Training

* Patch sizes: 4 × 4 and 7 × 7
* Patches per image: 49 or 16
* Embedding dimension: 96
* Class token: 1 per image
* Positional embeddings: Preserve patch position information.
* Transformer layers: 4, 5, 6, 7, 8
* Attention heads: 2, 3, 4
* Attention head dimension: 64
* Feed forward dimension: 384
* Dropout: 0.1
* Classification outputs: 10
* Optimizer: Adam
* Learning rate: 0.001
* Weight decay: 0.0001
* Loss function: Cross entropy
* Training epochs: 20 per configuration
* Total configurations: 2 × 5 × 3 = 30
* Maximum total training epochs: 600 across all configurations.

## Workflow

1. Input image.
2. Divide into patches.
3. Convert patches into embeddings.
4. Add positional information and class token.
5. Apply Transformer attention layers.
6. Generate 10 class scores.
7. Predict the clothing category.

## Evaluation, Results and Outputs

* Training metrics: Loss and accuracy.
* Validation metrics: Loss and accuracy.
* Test metrics: Loss and accuracy.
* Model comparison: Evaluate 30 configurations.
* Best model: Highest validation accuracy.
* Tie breaker: Lowest validation loss.
* Checkpoints: Save trained model states.
* CSV files: Store experiment results.
* Comparison table: Compare patch sizes, depths, and attention heads.
* Accuracy plot: Show accuracy against Transformer depth.
* Training curves: Show training and validation loss and accuracy.
* Libraries: PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Einops.
* Hardware: NVIDIA T4 GPU (Google Colab)
* Final objective: Find the best ViT configuration for FashionMNIST classification.
