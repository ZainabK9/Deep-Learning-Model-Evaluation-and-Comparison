# CIFAR-10 Inference & Evaluation Pipeline

This repository provides scripts to train, fine-tune, and evaluate three deep learning models—ResNet-18, EfficientNet-B0, and Vision Transformer (ViT)—on the CIFAR-10 dataset for image classification. The pipeline includes training scripts for each model, a unified CPU-only inference script, and tools to compute evaluation metrics such as accuracy, inference time, and confusion matrices.

**Version**: 1.0  
**Last Updated**: May 14, 2025  
**Authors**: Zainab 

## Contents

- **Training Scripts**:
  - `ResNet18_Model_.ipynb`: Trains  and Tests a ResNet-18 model on CIFAR-10 from scratch or using pre-trained weights.
  - `EfficientNet_Model.ipynb`: Fine-tunes an EfficientNet-B0 model on CIFAR-10 using Hugging Face Transformers.
  - `Vit_Model.ipynb`: Fine-tunes a Vision Transformer (ViT) model on CIFAR-10 using Hugging Face Transformers.

- **Inference Script**:
  - `inference.py`: A unified CPU-only script that runs inference on all three models (ResNet-18, EfficientNet-B0, ViT) sequentially, computes metrics (accuracy, inference time, parameter count), generates confusion matrices, and saves results to the `results/` directory.

- **Sample Results**:
  - `resnet18_test_results.csv`: Example CSV file containing true vs. predicted labels for ResNet-18 inference.

- **Checkpoint Directories**:
  - `checkpoints/checkpoint-resnet/`: Contains the trained ResNet-18 model checkpoint (`pytorch_model.bin`).
  - `checkpoints/checkpoint-efficientnet/`: Contains the fine-tuned EfficientNet-B0 model checkpoint (Hugging Face format).
  - `checkpoints/checkpoint-vit/`: Contains the fine-tuned ViT model checkpoint (Hugging Face format).

- **Results Directory**:
  - `results/`: Generated after running `inference.py`. Contains subdirectories (`vit/`, `resnet/`, `efficientnet/`) with:
    - `metrics.txt`: Test accuracy, inference time, parameter count, and class-wise accuracies.
    - `confusion_matrix.png`: Confusion matrix visualization.
    - `class_wise_accuracy_table.tex`: Class-wise accuracy table in LaTeX format.
    - `wrong_*.png`: Misclassified example images.

## Prerequisites

- Python 3.8 or higher
- `pip` package manager
- A CUDA-enabled GPU is optional for training (inference runs on CPU by default)

**Create virtual environment**
python -m venv ds
source ds/bin/activate

Prerequisites: 
Python 3.8+
pip package manager
CUDA-enabled GPU


Install required libraries:
!pip install torch torchvision transformers datasets numpy matplotlib scikit-learn
!pip install transformers[torch]

Dataset Setup: The scripts will automatically download CIFAR-10 via torchvision. No manual data download is needed. By default, data will be stored under ./data/.


Inference (CPU)
Run the unified pipeline on CPU:

Training
To train or fine-tune the models on CIFAR-10, use the provided Jupyter notebooks:
ResNet-18
Open ResNet18_Train_Model.ipynb in Jupyter Notebook.

Run all cells to train ResNet-18 from scratch (or with pre-trained weights by setting pretrained=True).

The trained checkpoint will be saved to checkpoints/checkpoint-resnet/.

EfficientNet-B0
Open EfficientNet_Model.ipynb in Jupyter Notebook.

Run all cells to fine-tune EfficientNet-B0 using Hugging Face Transformers.

The fine-tuned checkpoint will be saved to checkpoints/checkpoint-efficientnet/.

Vision Transformer (ViT)
Open Vit_Model.ipynb in Jupyter Notebook.

Run all cells to fine-tune ViT using Hugging Face Transformers.

The fine-tuned checkpoint will be saved to checkpoints/checkpoint-vit/.

Training Notes
Training requires a CUDA-enabled GPU for optimal performance, but CPU training is also supported (slower).

By default, training uses a subset of 5,000 training samples from CIFAR-10. Modify the dataset split in the notebooks to use the full training set (50,000 images).

Ensure sufficient disk space for saving checkpoints, especially for ViT and EfficientNet-B0 (Hugging Face format).

Inference (CPU)
Run the unified inference pipeline to evaluate all three models (ResNet-18, EfficientNet-B0, ViT) on the CIFAR-10 test set:
bash

python inference.py

This script will:
Load each model from its respective checkpoint directory (checkpoints/checkpoint-resnet/, checkpoints/checkpoint-efficientnet/, checkpoints/checkpoint-vit/).

Run inference on a 1,000-image subset of the CIFAR-10 test set (100 images per class).

Compute and save the following metrics for each model:
Top-1 accuracy

Total inference time (CPU)

Average inference time per image

Model parameter count

Class-wise accuracies

Generate and save:
Confusion matrix visualizations (confusion_matrix.png)

Class-wise accuracy tables in LaTeX format (class_wise_accuracy_table.tex)

Misclassified example images (wrong_*.png)

Save all results to the results/ directory, with subdirectories for each model (results/vit/, results/resnet/, results/efficientnet/).

Notes
Full Test Set Evaluation: To evaluate on the full CIFAR-10 test set (10,000 images), modify inference.py by updating the testds line in the run_inference function to testds = load_dataset("cifar10", split="test") (remove the [:1000] subset logic).

Checkpoint Paths: Ensure the checkpoint directories (checkpoint-resnet, checkpoint-efficientnet, checkpoint-vit) are in the same directory as inference.py and contain the necessary files (e.g., pytorch_model.bin for ResNet-18, Hugging Face checkpoint files for ViT and EfficientNet-B0).

Windows Users: If you encounter a symlink warning from Hugging Face (e.g., during EfficientNet-B0 inference), this is due to Windows’ lack of default symlink support. It does not affect results but may increase disk usage. Enable Developer Mode or run Python as an administrator to resolve (see Hugging Face documentation).

Troubleshooting
Checkpoint Not Found: Ensure the checkpoints/ directory contains the correct subdirectories (checkpoint-resnet, checkpoint-efficientnet, checkpoint-vit) with the required files (pytorch_model.bin for ResNet-18, Hugging Face checkpoint files for ViT and EfficientNet-B0).

Low Accuracy for ResNet-18: If ResNet-18 performs poorly (e.g., ~10% accuracy), the model may not have been trained sufficiently. Retrain using ResNet18_Train_Model.ipynb with more epochs or use pre-trained weights (pretrained=True).

Symlink Warning on Windows: If you see a symlink warning during EfficientNet-B0 or ViT inference, this is a Hugging Face caching issue on Windows. It does not affect results but may increase disk usage. Enable Developer Mode or run Python as an administrator to resolve.

Module Not Found Errors: Ensure all required libraries are installed (see Installation section). If issues persist, check your Python version and virtual environment.

Contributing
Contributions are welcome! To contribute:
Fork the repository.

Create a new branch (git checkout -b feature/your-feature).

Make your changes and commit (git commit -m "Add your feature").

Push to your branch (git push origin feature/your-feature).

Open a Pull Request.

Please ensure your code follows the existing style and includes appropriate documentation.
License
© 2025 Zainab

---


