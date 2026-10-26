## Vision Transformer (ViT) Image Classification

* Objective: Classify clothing images automatically
* Dataset: FashionMNIST
* Input: Grayscale images, 28 × 28 pixels
* Output: One of 10 clothing classes
* Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
* Model: Vision Transformer (ViT)
* Main goal: Compare 30 model configurations and select the best using validation performance
* Core idea: Divide images into patches and use self attention to learn visual patterns

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
* Data loading: PyTorch DataLoader
* Reproducibility: Fixed random seed and saved dataset split indices
* Training data: Learn model parameters
* Validation data: Select the best model
* Test data: Measure final classification performance

## Hardware and Software

* Programming language: Python
* Framework: PyTorch
* Dataset utilities: Torchvision
* Tensor manipulation: Einops
* Numerical computation: NumPy
* Results processing: Pandas
* Visualization: Matplotlib
* Environment: Google Colab
* Device used: CUDA
* GPU: NVIDIA Tesla T4

## Model Architecture and Training

* Patch sizes: 4 × 4 and 7 × 7
* Patches per image: 49 or 16
* Patch embedding: Flatten patches and project them into 96-dimensional vectors
* Embedding dimension: 96
* Class token: One learnable token per image
* Positional embeddings: Preserve patch position information
* Token sequence length: 50 for 4 × 4 patches and 17 for 7 × 7 patches, including the class token
* Transformer layers: 4, 5, 6, 7, 8
* Attention heads: 2, 3, 4
* Feed-forward dimension: 384
* Activation function: ReLU
* Layer normalization: Applied before attention and feed forward blocks
* Residual connections: Preserve information across Transformer blocks
* Dropout: 0.1
* Classification head: LayerNorm followed by a linear layer (96 -> 10)
* Classification outputs: 10 class scores
* Optimizer: Adam
* Learning rate: 0.001
* Weight decay: 0.0001
* Loss function: Cross-entropy
* Training epochs: Up to 20 per configuration
* Total configurations: 2 × 5 × 3 = 30
* Maximum training workload: 600 model training epochs across all configurations

### ViT Pipeline Flowchart:
<img width="429" height="705" alt="image" src="https://github.com/user-attachments/assets/d3ff84cd-d1e7-4bf4-8d58-d2453ffd4862" />

### Transformer Encoder Block:
<img width="430" height="562" alt="image" src="https://github.com/user-attachments/assets/d112e8ae-0c52-41dd-9ecf-4835ea13ae8d" />

## Workflow

1. Load the FashionMNIST dataset.
2. Normalize image data and partition the dataset into training, validation, and test sets.
3. Represent each grayscale image as a model input tensor.
4. Partition each image into non-overlapping patches.
5. Project the flattened patches into embedding vectors.
6. Prepend a learnable class token and add learnable positional embeddings.
7. Process the token sequence through Transformer encoder layers.
8. Extract the encoded class token representation.
9. Apply the classification head to generate logits for 10 classes.
10. Compute cross-entropy loss and update model parameters through backpropagation and optimization.
11. Evaluate validation loss and accuracy at the end of each epoch.
12. Save model checkpoints and per-epoch training metrics.
13. Execute the training procedure for all 30 model configurations.
14. Compare configurations and select the optimal model based on validation accuracy, using validation loss as the tie-breaker.
15. Evaluate the selected model on the test set and record its performance metrics.
16. Export experimental results to CSV files and generate comparison tables and performance plots.

## Evaluation, Results and Outputs

* Training metrics: Loss and accuracy
* Validation metrics: Loss and accuracy
* Test metrics: Loss and accuracy
* Model comparison: Evaluate 30 configurations
* Model selection: Highest validation accuracy; lowest validation loss breaks ties
* Checkpoints: Save trained model states
* Training history: Record per-epoch metrics
* CSV outputs: Store experiment results and selected-model details
* Comparison table: Summarize patch sizes, Transformer depths, and attention heads
* Accuracy plot: Compare test accuracy across Transformer depths
* Training curves: Plot training and validation loss and accuracy
* Overfitting analysis: Compare training and validation performance
* Final evaluation: Report the selected model’s test performance

### Training and Evaluation Architecture:
<img width="449" height="361" alt="image" src="https://github.com/user-attachments/assets/ffb27ad4-daf6-4a40-978a-b6984e90baf6" />

### Per-Configuration Training Loop:
<img width="539" height="434" alt="image" src="https://github.com/user-attachments/assets/6c388356-55cb-4987-8d5d-2900d389c31c" />

### Data and Artifact Flow:
<img width="539" height="434" alt="image" src="https://github.com/user-attachments/assets/c8957c88-4ee5-4674-bf20-3e2b9ab56490" />

## Experimental Results

### Best Model Selected by Validation Performance

* Patch size: 7 × 7
* Transformer layers: 8
* Attention heads: 4
* Selected epoch: 19
* Validation accuracy: 90.68%
* Test accuracy: 90.11%

### Overall Test Results

* Configurations evaluated: 30
* Highest test accuracy: 90.31%
* Best test configuration: 4 × 4 patches, 8 layers, 3 heads.
* Selected model test accuracy: 90.11%.

## Selected Model Training Behaviour

* Training loss decreases steadily.
* Validation loss decreases and stabilizes.
* Training accuracy increases consistently.
* Validation accuracy reaches 90.68%.
* The training-validation gap suggests some overfitting.
