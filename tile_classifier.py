# tile_classifier.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import config  # Use our config for class definitions
import argparse
from pathlib import Path

# --- Configuration ---
# You might want to move these to config.py or keep them here
MODEL_SAVE_PATH = "tile_classifier_model.pth"
NUM_EPOCHS = 20  # Adjust as needed
BATCH_SIZE = 32  # Adjust based on your GPU memory
LEARNING_RATE = 0.001
RESIZE_SIZE = 64  # Resize tiles to this dimension for the CNN


# --- Custom Dataset ---
class TileDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.content_classes = config.CONTENT_CLASSES
        self.background_classes = config.BACKGROUND_CLASSES
        self.map_fg = config.MAP_FG
        self.map_bg = config.MAP_BG

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            # Extract labels from folder name
            folder_name = Path(img_path).parent.name
            fg_str, bg_str = folder_name.split("__")

            # Convert labels to integer indices
            content_label = self.map_fg[fg_str]
            background_label = self.map_bg[bg_str]

            # Load image
            image = Image.open(img_path).convert("RGB")  # Ensure 3 channels

            if self.transform:
                image = self.transform(image)

            return image, content_label, background_label
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            # Return a dummy item or skip - choosing dummy here
            # Be cautious with this in production, might skew training
            dummy_image = torch.zeros((3, RESIZE_SIZE, RESIZE_SIZE))
            return dummy_image, 0, 0  # empty, none


# --- Model Definition (2-Headed ResNet18) ---
class TileClassifier(nn.Module):
    def __init__(self, num_content_classes, num_background_classes):
        super().__init__()
        # Use a pre-trained ResNet18 as the base
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final fully connected layer of ResNet
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the original classification layer

        # Head 1: Content Classification
        self.content_head = nn.Linear(num_ftrs, num_content_classes)

        # Head 2: Background Classification
        self.background_head = nn.Linear(num_ftrs, num_background_classes)

    def forward(self, x):
        features = self.base_model(x)
        content_logits = self.content_head(features)
        background_logits = self.background_head(features)
        return content_logits, background_logits


# --- Training Function ---
def train_model(
    model,
    train_loader,
    val_loader,
    criterion_content,
    criterion_background,
    optimizer,
    device,
    num_epochs=NUM_EPOCHS,
):
    model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_content = 0
        correct_background = 0
        total = 0

        for i, (inputs, content_labels, background_labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            content_labels = content_labels.to(device)
            background_labels = background_labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            content_logits, background_logits = model(inputs)

            # Calculate loss
            loss_content = criterion_content(content_logits, content_labels)
            loss_background = criterion_background(background_logits, background_labels)
            loss = loss_content + loss_background  # Combine losses

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted_content = torch.max(content_logits.data, 1)
            _, predicted_background = torch.max(background_logits.data, 1)
            total += content_labels.size(0)  # or background_labels.size(0)
            correct_content += (predicted_content == content_labels).sum().item()
            correct_background += (
                (predicted_background == background_labels).sum().item()
            )

            if (i + 1) % 5 == 0:  # Print progress every 5 batches
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_content = 100 * correct_content / total
        epoch_acc_background = 100 * correct_background / total

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {epoch_loss:.4f} | Content Acc: {epoch_acc_content:.2f}% | BG Acc: {epoch_acc_background:.2f}%"
        )

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct_content = 0
        val_correct_background = 0
        val_total = 0
        with torch.no_grad():
            for inputs, content_labels, background_labels in val_loader:
                inputs = inputs.to(device)
                content_labels = content_labels.to(device)
                background_labels = background_labels.to(device)

                content_logits, background_logits = model(inputs)
                loss_content = criterion_content(content_logits, content_labels)
                loss_background = criterion_background(
                    background_logits, background_labels
                )
                loss = loss_content + loss_background

                val_loss += loss.item() * inputs.size(0)
                _, predicted_content = torch.max(content_logits.data, 1)
                _, predicted_background = torch.max(background_logits.data, 1)
                val_total += content_labels.size(0)
                val_correct_content += (
                    (predicted_content == content_labels).sum().item()
                )
                val_correct_background += (
                    (predicted_background == background_labels).sum().item()
                )

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc_content = 100 * val_correct_content / val_total
        val_epoch_acc_background = 100 * val_correct_background / val_total
        # Use average accuracy for simplicity in saving best model
        val_epoch_acc_avg = (val_epoch_acc_content + val_epoch_acc_background) / 2

        print(
            f"Val Loss: {val_epoch_loss:.4f} | Content Acc: {val_epoch_acc_content:.2f}% | BG Acc: {val_epoch_acc_background:.2f}%"
        )
        print("-" * 30)

        # Save the best model based on validation accuracy
        if val_epoch_acc_avg > best_val_acc:
            best_val_acc = val_epoch_acc_avg
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(
                f"*** Best model saved to {MODEL_SAVE_PATH} (Avg Acc: {best_val_acc:.2f}%) ***"
            )

    print("Training Finished.")


# --- Inference Function ---
def classify_tile(model, image_path_or_pil, device):
    model.eval()
    model.to(device)

    # Define the same transforms used during training (except augmentation)
    infer_transform = transforms.Compose(
        [
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet stats
        ]
    )

    if isinstance(image_path_or_pil, str) or isinstance(image_path_or_pil, Path):
        image = Image.open(image_path_or_pil).convert("RGB")
    elif isinstance(image_path_or_pil, Image.Image):
        image = image_path_or_pil.convert("RGB")
    else:
        raise ValueError("Input must be a file path or PIL image")

    input_tensor = infer_transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        content_logits, background_logits = model(input_tensor)

    content_prob = torch.softmax(content_logits, dim=1)
    background_prob = torch.softmax(background_logits, dim=1)

    content_pred_idx = content_prob.argmax(1).item()
    background_pred_idx = background_prob.argmax(1).item()

    content_label = config.CONTENT_CLASSES[content_pred_idx]
    background_label = config.BACKGROUND_CLASSES[background_pred_idx]

    content_confidence = content_prob.max().item()
    background_confidence = background_prob.max().item()

    # Don't predict background if content is empty
    if content_label == "empty":
        background_label = "none"  # Force background to 'none'

    return content_label, background_label, content_confidence, background_confidence


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the tile classifier.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run in training or testing mode.",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image to classify (requires --mode test).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="unique_labeled",
        help="Directory containing labeled images (e.g., unique_labeled/gem_0__stone/*.png)",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help="Learning rate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size."
    )

    args = parser.parse_args()

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Model ---
    num_content = len(config.CONTENT_CLASSES)
    num_background = len(config.BACKGROUND_CLASSES)
    model = TileClassifier(num_content, num_background)

    if args.mode == "train":
        print("Starting training...")
        # --- Prepare Data ---
        all_files = glob.glob(f"{args.data_dir}/**/*.png", recursive=True)
        if not all_files:
            raise FileNotFoundError(
                f"No labeled images found in {args.data_dir}. Run labeling first."
            )

        # Split data (consider stratification if classes are imbalanced)
        train_files, val_files = train_test_split(
            all_files, test_size=0.2, random_state=42
        )  # 80% train, 20% val

        # Define transforms (add augmentation for training)
        train_transform = transforms.Compose(
            [
                transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
                transforms.RandomHorizontalFlip(),  # Simple augmentation
                transforms.RandomRotation(10),  # Simple augmentation
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1
                ),  # Simple augmentation
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet stats
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = TileDataset(train_files, transform=train_transform)
        val_dataset = TileDataset(val_files, transform=val_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # --- Setup Optimizer and Loss ---
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion_content = nn.CrossEntropyLoss()
        criterion_background = nn.CrossEntropyLoss()

        # --- Start Training ---
        train_model(
            model,
            train_loader,
            val_loader,
            criterion_content,
            criterion_background,
            optimizer,
            device,
            num_epochs=args.epochs,
        )
        print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

    elif args.mode == "test":
        if not args.image:
            print("Error: --image argument is required for test mode.")
            exit()
        if not os.path.exists(MODEL_SAVE_PATH):
            print(
                f"Error: Model file not found at {MODEL_SAVE_PATH}. Train the model first."
            )
            exit()

        print(f"Loading model from {MODEL_SAVE_PATH} for inference...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Classifying image: {args.image}")

        content, background, conf_fg, conf_bg = classify_tile(model, args.image, device)

        print("\n--- Classification Result ---")
        print(f"Content:    {content} (Confidence: {conf_fg:.2f})")
        print(f"Background: {background} (Confidence: {conf_bg:.2f})")
