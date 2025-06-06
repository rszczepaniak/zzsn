import matplotlib.pyplot as plt


def plot_metrics_on_same_plot(training_results, file_name):
    history = training_results["train_history"]
    if "num_pseudo_epochs" in training_results["hyperparameters"]:
        epochs = list(range(training_results["hyperparameters"]["num_epochs"] * training_results["hyperparameters"]["num_pseudo_epochs"]))
    else:
        epochs = list(range(training_results["hyperparameters"]["num_epochs"]))
    val_f1 = [entry["val_f1_score"] for entry in history]
    val_iou = [entry["val_iou"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, val_f1, label="Validation F1 Score")
    plt.plot(epochs, val_iou, label="Validation IoU")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Validation Metrics over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}.png")
