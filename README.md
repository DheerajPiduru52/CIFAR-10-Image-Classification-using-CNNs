# BUAN 6382 – Deep Learning  
## Project: CIFAR-10 Image Classification using CNNs

### **Team Members**
- Dheeraj Rahul Reddy Piduru
- Yoshinee Lingampalli 
- Jaahnavi Pothukanuri

---

## Step 1: Data Exploration and Preprocessing

### Dataset Overview
- **Dataset**: CIFAR-10  
- **Images**: 60,000 (32x32 RGB), 10 classes  
- **Train/Test Split**: 50,000 / 10,000  
- **Format**: 5 training `.py` batches + 1 test batch  
- **Labels**:  
  `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`
  ![images](assets/10images.jpg)
  ![images1](assets/distribution.jpg)

### Tasks Performed
- Custom `unpickle` function to load CIFAR-10 batches
- Loaded and combined training data
- Extracted human-readable class names
- Visualized one image per class  
- Displayed class distribution via seaborn

### Preprocessing
- **Reshape**: `(num_samples, 3072)` → `(num_samples, 3, 32, 32)`  
- **Normalization**: Divided pixel values by 255  
- **Split**:
  - 90% Train → 45,000  
  - 10% Validation → 5,000  
  - Test: 10,000  

---

## Step 2: Building a Basic CNN

### Architecture
- Conv2D (32 filters) → MaxPool  
- Conv2D (64 filters) → MaxPool  
- FC Layer (128 units) → Output Layer (10 classes)

### Details
- ReLU activation
- Flattening for Dense layers
- Used PyTorch `nn.Module`

### Training Setup
- Loss Function: `CrossEntropyLoss`  
- Optimizer: `Adam` (lr = 0.001)  
- Epochs: 10  
- Batch Size: 64

### Results Summary

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 10    | 82.73%    | 70.90%  | 0.4938     | 0.8863   |

![bm](assets/bm.jpg)
![bg](assets/bg.jpg)
  
- **Saved as**: [`basic_cnn_cifar10.pth`](./assets/basic_cnn_cifar10.pth)

- Validation plateaued after epoch 5 → potential overfitting

---

## Step 3: Model Evaluation (BasicCNN)

- **Test Accuracy**: 70.47%  
- **Macro Precision**: 0.7124  
- **Macro Recall**: 0.7047  
- **Macro F1-Score**: 0.7036  
- **Confusion Matrix**: Visualized with `ConfusionMatrixDisplay`

![bcm](assets/bcm.jpg)


### Observations
- Decent performance but underfitting on deeper patterns
- Lacks regularization and data variability

---

## Step 4: Model Improvement

### ImprovedCNN Architecture
- Added: 3rd Conv layer (128 filters)
- Dropout: 0.5 before FC
- FC Layer: Increased to 256 units

### Training Enhancements
- **L2 Regularization**: `weight_decay=1e-4`
- **Learning Rate**: Reduced to 0.0005
- **Epochs**: 20

### Results Summary

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 20    | 75.28%    | 77.42%  | 0.7042     | 0.6606   |

![im](assets/im.jpg)
![ig](assets/ig.jpg)

- **Saved as**: [`improved_cnn_cifar10.pth`](./assets/improved_cnn_cifar10.pth)

### Test Performance
- **Test Accuracy**: 76.74%  
- **Macro F1-Score**: 0.7642  
- Confusion matrix showed improved balance across classes
![icm](assets/icm.jpg)
---

## Final Model: DeeperCNN

### Key Enhancements
- 3 convolutional **blocks** (2 Conv layers + BN + MaxPool + Dropout)
- **Dropout Rates**: 0.2 → 0.3 → 0.4 (progressive)
- **Global Average Pooling** before final FC
- **Scheduler**: CosineAnnealingLR (`T_max=30`)
- **L2 Reg**: `weight_decay=5e-4`

### Configuration

| Parameter         | Value     |
|------------------|-----------|
| Learning Rate     | 0.001     |
| Epochs            | 25        |
| Dropout           | Progressive |
| Batch Size        | 64        |
| Optimizer         | Adam      |

### Results Summary (Trained to Epoch 20)

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 20    | 91.74%    | 91.42%  | 0.2383     | 0.2604   |

![dm](assets/dm.jpg)
![dg](assets/dg.jpg)

- **Saved as**: [`deeper_model_epoch20.pth`](./assets/deeper_model_epoch20.pth)

### Test Performance
- **Test Accuracy**: 90.17%  
- **Macro F1-Score**: 0.9009  
- Confusion matrix confirms strong performance across all classes
![dcm](assets/dcm.jpg)

---

## Comparison Summary

| Model       | Val Acc | Test Acc | Macro F1 | Notes                         |
|-------------|---------|----------|-----------|-------------------------------|
| BasicCNN    | ~71%    | 70.5%    | 0.7036    | Simple 2-layer CNN            |
| ImprovedCNN | ~77%    | 76.7%    | 0.7642    | Dropout + Augmentation        |
| DeeperCNN   | ~91%    | 90.2%    | 0.9009    | Deep blocks + Regularization  |

---

## Final Conclusion

- **BasicCNN** provided a strong start.
- **ImprovedCNN** used architectural tweaks to boost generalization.
- **DeeperCNN** introduced advanced design: deeper structure, batch norm, dropout, global pooling, and cosine scheduling.
- **Result**: DeeperCNN achieved ~91% validation accuracy and ~90% test accuracy with minimal overfitting.

