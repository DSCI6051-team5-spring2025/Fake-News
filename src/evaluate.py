from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, processor, data_loader, device):
    """
    Evaluate the model and print confusion matrix and metrics.

    Args:
    - model: Trained CLIP model
    - processor: CLIPProcessor for preprocessing
    - data_loader: DataLoader for validation/test dataset
    - device: Device (CPU or GPU)

    Returns:
    - Confusion matrix and classification report
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch.pop('labels')

            # Perform inference
            logits, _ = model(**batch, return_loss=False)

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print accuracy and classification report
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc:.4f}\n")

    class_report = classification_report(all_labels, all_preds, target_names=["Fake", "Real"])
    print("\n Classification Report:\n")
    print(class_report)

    # Generate and display confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return conf_matrix, class_report

  # Evaluate the model on the validation set
conf_matrix, class_report = evaluate_model(model, processor, val_loader, device)

