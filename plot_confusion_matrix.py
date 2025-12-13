import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from bcresnet import BCResNets
from utils import SpeechCommand, Padding, Preprocess, label_dict

def load_model(model_path, device):
    """Load the model and infer parameters."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint (dict) and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        tau = checkpoint.get('tau', 8.0) # Default to 8.0 if not found
        print(f"Checkpoint info: Tau={tau}, Epoch={checkpoint.get('epoch', '?')}, Acc={checkpoint.get('valid_acc', checkpoint.get('test_acc', '?'))}")
    else:
        state_dict = checkpoint
        tau = 8.0 # Assumption if loading raw weights
        print("Warning: Loading raw state_dict, assuming tau=8.0")

    # Infer base_c from weights if possible
    # cnn_head.0.weight shape is [base_c * 2, 1, 5, 5]
    if 'cnn_head.0.weight' in state_dict:
        head_shape = state_dict['cnn_head.0.weight'].shape[0]
        base_c = head_shape // 2
        print(f"Inferred base_c={base_c} from weights.")
        model = BCResNets(base_c=base_c)
    else:
        # Fallback to calculating from tau
        print(f"Could not infer base_c, calculating from tau={tau}")
        model = BCResNets(base_c=int(tau * 8))
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tau

def evaluate(model, loader, device, preprocess_fn):
    """Run inference and return predictions and labels."""
    all_preds = []
    all_labels = []
    
    print(f"Evaluating on {len(loader.dataset)} samples...")
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Apply preprocessing (spectrogram)
            inputs = preprocess_fn(inputs, labels=labels, is_train=False, augment=False)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    return all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred, output_file, title="Confusion Matrix"):
    """Plot and save normalized confusion matrix."""
    # Get class names and indices in correct order from label_dict
    sorted_items = sorted(label_dict.items(), key=lambda x: x[1])
    class_names = [name for name, idx in sorted_items]
    class_indices = [idx for name, idx in sorted_items]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)
    
    # Normalize
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1 # Prevent division by zero
    cm_normalized = cm.astype('float') / row_sums
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Confusion matrix saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Rebuild confusion matrix from saved BCResNet model")
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'valid', 'test'], help='Dataset split to evaluate on')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--download', action='store_true', help='Download dataset if missing')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Check for data
    if args.download or (not os.path.exists("./data/speech_commands_v0.02_split") and not os.path.exists("./data/speech_commands_v0.01_split")):
        print("Dataset not found or download requested. Downloading...")
        # Use utils to download
        from utils import DownloadDataset, DownloadDonutClass, SplitDataset
        
        if not os.path.isdir("./data"):
            os.mkdir("./data")
            
        # Default to v2
        base_dir = "./data/speech_commands_v0.02"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
        test_dir = base_dir.replace("commands", "commands_test_set")
        
        # Clean old?
        if os.path.exists(base_dir):
            import shutil
            shutil.rmtree(base_dir)
            
        os.mkdir(test_dir)
        DownloadDataset(test_dir, url_test)
        os.mkdir(base_dir)
        DownloadDataset(base_dir, url)
        DownloadDonutClass(base_dir)
        SplitDataset(base_dir)
        print("Download and split complete.")

    # Load Data
    # Assuming standard GSC data structure in ./data/speech_commands_v0.02
    # We need to detect version v1 or v2?
    # utils.py defaults to v2 structure usually if ScanAudioFiles is used with ver=2?
    # Let's try to detect based on directory existence
    
    data_dir_v2 = "./data/speech_commands_v0.02"
    data_dir_v1 = "./data/speech_commands_v0.01"
    
    if os.path.isdir(data_dir_v2):
        base_dir = data_dir_v2
        ver = 2
    elif os.path.isdir(data_dir_v1):
        base_dir = data_dir_v1
        ver = 1
    else:
        print("Error: Could not find dataset directory (checked v0.02_split and v0.01_split in ./data)")
        print("Please run main.py --download first or ensure data structure.")
        return

    data_dir = os.path.join(base_dir, f"{args.dataset}_13class")
    print(f"Using dataset directory: {data_dir} (v{ver})")
    
    # Setup Preprocessing (must match training parameters roughly)
    # Preprocess class needs noise_dir to initialize backing noise, even if we don't augment testing
    noise_dir = os.path.join(base_dir.replace("_split", ""), "_background_noise_")
    
    # Transform
    transform = transforms.Compose([Padding()])
    dataset = SpeechCommand(data_dir, ver, transform=transform)
    loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    
    # Preprocessor
    preprocess = Preprocess(noise_dir, device) # augment=False will be passed in call
    
    # Load Model
    model, tau = load_model(args.model, device)
    
    # Evaluate
    y_true, y_pred = evaluate(model, loader, device, preprocess)
    
    # Calculate Accuracy
    acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    print(f"Accuracy on {args.dataset} set: {acc:.2f}%")
    
    # Plot
    output_filename = f"rebuilt_confusion_matrix_{os.path.basename(args.model)}.png"
    plot_confusion_matrix(y_true, y_pred, output_filename, 
                          title=f"Confusion Matrix (Acc={acc:.2f}%)")

if __name__ == "__main__":
    main()
