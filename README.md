# Pokemon Classifier
![example.png](/example.png)[image_source](https://colley.kr/item/1959806542)  
application that can classify 150 types of pokemon.

## Key Features
* **Multi-Model Experiment**: Fine-tuning and performance comparison of four different CNN architectures (ResNet34, VGG16, GoogLeNet, EfficientNet-B0)
* **Web Demo (GUI)**: Supports file uploads and image URL input using Streamlit, and provides a multi-model viewer that allows users to compare the prediction results of the four models at a glance

### Model Lineup and Features
1. **ResNet-34 (Baseline):** The most standard and well-balanced model using residual connections
2. **VGG-16:** A classic model built with deep layers using only 3x3 convolutional filters; it has the highest number of parameters
3. **GoogLeNet (Inception v1):** A model that maximizes computational efficiency through 1x1 convolutions
4. **EfficientNet-B0:** A state-of-the-art, high-performance model that automatically optimizes depth, width, and resolution

### Evaluation Metrics (Classification Report)
*Training Settings: 50 Epochs, Adam Optimizer (lr=0.001), CrossEntropyLoss*

| Model | Test Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet-34** | **0.96** | 0.96 | 0.96 | 0.96 |
| **EfficientNet-B0**| **0.95** | 0.96 | 0.95 | 0.95 |
| **VGG-16** | 0.92 | 0.93 | 0.93 | 0.92 |
| **GoogLeNet** | 0.54 | 0.74 | 0.54 | 0.57 |

> **Model Performance Analysis:** 
> * **ResNet-34 & EfficientNet-B0:** Both models achieved an outstanding accuracy of over 95%. EfficientNet, in particular, demonstrated excellent performance and efficiency comparable to the deeper and heavier ResNet, despite its lightweight architecture.
> * **VGG-16:** Achieved a respectable accuracy of 92%, but its massive number of parameters made it relatively heavy and slow during both training and inference.
> * **GoogLeNet:** Recorded the lowest performance at 54%. This suggests that its complex Inception module structure (utilizing 1x1 convolutions) struggled to sufficiently learn the distinct features of the Pokémon dataset within a short fine-tuning period of only 3 epochs.

## 💻 Installation & Execution

This project uses [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver.

**1. Setup Virtual Environment & Install Dependencies**

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux: source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# Install the required libraries at lightning speed
uv pip install torch torchvision streamlit pillow requests kagglehub scikit-learn tqdm
```

**2. Train the Models (Generate .pth weights)**
```bash
# Run the training scripts to generate the model weights
uv run resnet34.py
uv run vgg16.py
uv run googlenet.py
uv run efficientnet.py
```

**3. Run the Streamlit Web App**
```bash
uv run streamlit run app.py
```
