# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from argparse import ArgumentParser
import shutil
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from bcresnet import BCResNets
from utils import DownloadDataset, DownloadDonutClass, Padding, Preprocess, SpeechCommand, SplitDataset, label_dict


class BinaryDataset(Dataset):
    """Wrapper dataset that converts multi-class to binary (donut=1, other=0)."""
    
    def __init__(self, base_dataset, donut_label=12):
        self.base_dataset = base_dataset
        self.donut_label = donut_label
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample, label = self.base_dataset[idx]
        # Convert to binary: donut=1, everything else=0
        binary_label = 1 if label == self.donut_label else 0
        return sample, binary_label
    
    def count_classes(self):
        """Count samples per class for weight calculation."""
        donut_count = 0
        other_count = 0
        for i in range(len(self.base_dataset)):
            _, label = self.base_dataset[i]
            if label == self.donut_label:
                donut_count += 1
            else:
                other_count += 1
        return other_count, donut_count  # [class 0, class 1]


class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="google speech command set version 1 or 2", type=int
        )
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        parser.add_argument("--binary", help="binary classification: donut vs all", action="store_true")
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self.num_classes = 2 if self.binary else 13
        self.class_weights = None  # Will be set for binary mode
        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # train hyperparameters
        total_epoch = 200
        warmup_epoch = 5
        init_lr = 1e-1
        lr_lower_limit = 0

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        # train
        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                
                # Use weighted loss for binary mode to handle class imbalance
                if self.class_weights is not None:
                    loss = F.cross_entropy(outputs, labels, weight=self.class_weights)
                else:
                    loss = F.cross_entropy(outputs, labels)
                    
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print("valid acc: %.3f" % (valid_acc))
            
            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                print("\n" + "="*50)
                print("CHECKPOINT at epoch %d" % (epoch + 1))
                print("="*50)
                
                with torch.no_grad():
                    self.model.eval()
                    # Test on validation set with predictions for confusion matrix
                    valid_acc_ckpt, valid_preds, valid_labels = self.Test(
                        self.valid_dataset, self.valid_loader, augment=False, return_preds=True
                    )
                    print("Validation accuracy: %.3f%%" % valid_acc_ckpt)
                    
                    # Plot and save confusion matrix
                    self._plot_confusion_matrix(
                        valid_labels, valid_preds, 
                        suffix="_epoch%d" % (epoch + 1)
                    )
                
                # Save checkpoint model
                mode_str = "_binary" if self.binary else ""
                ckpt_path = "bcresnet_tau%.1f_v%d%s_epoch%d_acc%.2f.pth" % (
                    self.tau, self.ver, mode_str, epoch + 1, valid_acc_ckpt
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tau': self.tau,
                    'ver': self.ver,
                    'binary': self.binary,
                    'num_classes': self.num_classes,
                    'valid_acc': valid_acc_ckpt,
                    'lr': lr,
                }, ckpt_path)
                print("Checkpoint saved to %s" % ckpt_path)
                print("="*50 + "\n")

        test_acc, all_preds, all_labels = self.Test(self.test_dataset, self.test_loader, augment=False, return_preds=True)  # official testset
        print("test acc: %.3f" % (test_acc))
        
        # Compute and display normalized confusion matrix
        self._plot_confusion_matrix(all_labels, all_preds, suffix="_final")
        
        # Save the trained model
        mode_str = "_binary" if self.binary else ""
        model_path = "bcresnet_tau%.1f_v%d%s_acc%.2f.pth" % (self.tau, self.ver, mode_str, test_acc)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tau': self.tau,
            'ver': self.ver,
            'binary': self.binary,
            'num_classes': self.num_classes,
            'test_acc': test_acc,
        }, model_path)
        print("Model saved to %s" % model_path)
        
        print("End.")

    def Test(self, dataset, loader, augment, return_preds=False):
        """
        Tests the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.
            return_preds (bool): If True, also return all predictions and labels.

        Returns:
            float: The accuracy of the model on the given dataset.
            If return_preds is True, also returns (all_preds, all_labels).
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        all_preds = []
        all_labels = []
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
            if return_preds:
                all_preds.extend(prediction.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
        acc = true_count / num_testdata * 100.0  # percentage
        if return_preds:
            return acc, all_preds, all_labels
        return acc

    def _plot_confusion_matrix(self, all_labels, all_preds, suffix=""):
        """
        Plots a normalized confusion matrix and saves it to a file.
        
        Parameters:
            all_labels: List of true labels.
            all_preds: List of predicted labels.
            suffix: Optional suffix for the filename (e.g., "_epoch10").
        """
        # Get class names + fixed label order (so missing classes still appear)
        if self.binary:
            class_names = ["other", "donut"]
        else:
            class_names = [name for name, idx in sorted(label_dict.items(), key=lambda x: x[1])]

        labels = list(range(self.num_classes))

        # Compute confusion matrix with fixed labels (prevents "missing class" shrinking)
        cm = confusion_matrix(all_labels, all_preds, labels=labels)

        # Normalize by row (true labels) safely (handles rows with 0 samples)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        
        # Plot
        figsize = (8, 6) if self.binary else (12, 10)
        plt.figure(figsize=figsize)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        mode_str = "_binary" if self.binary else ""
        plt.title('Normalized Confusion Matrix (tau=%.1f, v%d%s%s)' % (self.tau, self.ver, mode_str, suffix))
        plt.tight_layout()
        
        # Save figure
        cm_path = "confusion_matrix_tau%.1f_v%d%s%s.png" % (self.tau, self.ver, mode_str, suffix)
        plt.savefig(cm_path, dpi=150)
        print("Confusion matrix saved to %s" % cm_path)
        plt.close()

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        print("Check google speech commands dataset v1 or v2 ...")
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")
        official_test_dir = base_dir.replace("commands", "commands_test_set")
        if self.download:
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(official_test_dir)
            DownloadDataset(official_test_dir, url_test)
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url)
            # Download custom donut class
            DownloadDonutClass(base_dir)
            SplitDataset(base_dir)
            print("Done...")

        # Define data loaders
        train_dir = "%s/train_13class" % base_dir
        valid_dir = "%s/valid_13class" % base_dir
        # IMPORTANT: use the split test set so that custom class (donut) is present in test
        test_dir = "%s/test_13class" % base_dir
        noise_dir = "%s/_background_noise_" % base_dir

        transform = transforms.Compose([Padding()])
        train_dataset_base = SpeechCommand(train_dir, self.ver, transform=transform)
        valid_dataset_base = SpeechCommand(valid_dir, self.ver, transform=transform)
        test_dataset_base = SpeechCommand(test_dir, self.ver, transform=transform)
        
        # Wrap datasets for binary mode if needed
        if self.binary:
            print("\n" + "="*50)
            print("BINARY MODE: donut vs all")
            print("="*50)
            
            donut_label = label_dict.get('donut', 12)
            self.train_dataset = BinaryDataset(train_dataset_base, donut_label)
            self.valid_dataset = BinaryDataset(valid_dataset_base, donut_label)
            self.test_dataset = BinaryDataset(test_dataset_base, donut_label)
            
            # Calculate class weights for imbalanced data
            print("Counting samples per class for weight calculation...")
            other_count, donut_count = self.train_dataset.count_classes()
            total = other_count + donut_count
            
            # Inverse frequency weighting
            weight_other = total / (2.0 * other_count)
            weight_donut = total / (2.0 * donut_count)
            
            self.class_weights = torch.tensor([weight_other, weight_donut], dtype=torch.float32).to(self.device)
            
            print(f"  Class 0 (other): {other_count} samples, weight: {weight_other:.4f}")
            print(f"  Class 1 (donut): {donut_count} samples, weight: {weight_donut:.4f}")
            print(f"  Imbalance ratio: {other_count/donut_count:.1f}:1")
            print("="*50 + "\n")
        else:
            self.train_dataset = train_dataset_base
            self.valid_dataset = valid_dataset_base
            self.test_dataset = test_dataset_base
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=100, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        mode_str = " (BINARY: donut vs all)" if self.binary else ""
        print("model: BC-ResNet-%.1f on data v0.0%d%s" % (self.tau, self.ver, mode_str))
        self.model = BCResNets(int(self.tau * 8), num_classes=self.num_classes).to(self.device)


if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()
