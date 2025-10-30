# Import required libraries
import torch
import argparse
import os
from transformers import ViTImageProcessor, ViTForImageClassification, AutoImageProcessor, AutoModelForImageClassification
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets import load_dataset
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_model(checkpoint_dir, model_type='vit'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, checkpoint_dir)
    if model_type == 'resnet':
        model = resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location='cpu'))
        processor = None
    elif model_type == 'vit':
        model_name = "google/vit-base-patch16-224"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            checkpoint_path,
            num_labels=10,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )
    elif model_type == 'efficientnet':
        model_name = "google/efficientnet-b0"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            checkpoint_path,
            num_labels=10,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.to('cpu')
    model.eval()
    return model, processor

def inference(image, model, processor, model_type='vit'):
    if model_type == 'vit':
        transform = Compose([
            Resize(224),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std)
        ])

    elif model_type == 'resnet': 
        transform = Compose([ 
            ToTensor(), 
            Normalize( 
                mean=(0.4914, 0.4822, 0.4465), 
                std =(0.2470, 0.2435, 0.2616)), 
        ])
    elif model_type == 'efficientnet':
        transform = Compose([
            Resize(224),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std)
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    pixel_values = transform(image.convert('RGB')).unsqueeze(0).to('cpu')
    
    with torch.no_grad():
        outputs = model(pixel_values)
    
    return outputs.logits.argmax(-1).item() if model_type in ['vit', 'efficientnet'] else outputs.argmax(-1).item()

def run_inference(model_type, checkpoint_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRunning inference for {model_type.upper()}...")
    model, processor = load_model(checkpoint_dir, model_type)

    testds = load_dataset("cifar10", split="test[:1000]")
    itos = dict((k, v) for k, v in enumerate(testds.features['label'].names))

    start_time = time.time()
    predictions = []
    true_labels = []
    for example in testds:
        image = example['img']
        label = example['label']
        pred = inference(image, model, processor, model_type)
        predictions.append(pred)
        true_labels.append(label)

    # Tahmin dağılımını kontrol etme
    print("Prediction distribution:", np.bincount(predictions))

    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    inference_time = time.time() - start_time
    param_count = sum(p.numel() for p in model.parameters())

    class_accuracies = {}
    for class_id in range(10):
        class_mask = np.array(true_labels) == class_id
        if class_mask.sum() > 0:
            class_preds = np.array(predictions)[class_mask]
            class_true = np.array(true_labels)[class_mask]
            class_acc = np.mean(class_preds == class_true)
            class_accuracies[itos[class_id]] = class_acc

    cm = confusion_matrix(true_labels, predictions)
    labels = list(itos.values())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(xticks_rotation=45, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Total Inference Time (CPU): {inference_time:.2f} seconds\n")
        f.write(f"Average Inference Time per Image: {inference_time / len(testds):.4f} seconds\n")
        f.write(f"Model Parameters: {param_count:,}\n")
        f.write("\nClass-wise Accuracy:\n")
        for class_name, class_acc in class_accuracies.items():
            f.write(f"{class_name}: {class_acc:.4f} ({np.sum(np.array(true_labels) == list(itos.keys())[list(itos.values()).index(class_name)])} samples)\n")

    with open(os.path.join(results_dir, "class_wise_accuracy_table.tex"), "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("    \\centering\n")
        f.write("    \\begin{tabular}{|l|c|}\n")
        f.write("        \\hline\n")
        f.write("        \\textbf{Class} & \\textbf{Accuracy (\\%)} \\\\\n")
        f.write("        \\hline\n")
        for class_name, class_acc in class_accuracies.items():
            f.write(f"        {class_name} & {class_acc*100:.2f} \\\\\n")
        f.write("        \\hline\n")
        f.write(f"        \\textbf{{Average}} & \\textbf{{{accuracy*100:.1f}}} \\\\\n")
        f.write("        \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write(f"    \\caption{{Class-wise accuracy (\\%) of {model_type.upper()} on CIFAR-10}}\n")
        f.write(f"    \\label{{tab:class_wise_accuracy_{model_type}}}\n")
        f.write("\\end{table}\n")

    wrong_indices = np.where(np.array(predictions) != np.array(true_labels))[0]
    if len(wrong_indices) == 0:
        print("No misclassified examples!")
    else:
        print(f"Number of misclassified examples: {len(wrong_indices)}")
        for i, idx in enumerate(wrong_indices[:5]):
            image = testds[int(idx)]['img']
            true_label = itos[true_labels[idx]]
            pred_label = itos[predictions[idx]]
            plt.imshow(image)
            plt.title(f"True: {true_label}, Predicted: {pred_label}")
            plt.axis('off')
            plt.savefig(os.path.join(results_dir, f"wrong_{i}.png"))
            plt.close()

def main():

    run_inference(
        model_type     = "model_name",  # Change to 'resnet' or 'efficientnet' as needed     
        checkpoint_dir = "checkpoint-model_name",  # Change to 'checkpoint-efficientnet' as needed
        results_dir    = "results/model_name"  # Change to 'results/efficientnet' as needed
    )

if __name__ == "__main__":
    main()