
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns




def plot_training_metrics(train_losses, f1_scores, recall_scores, validation_losses, validation_f1_scores, filename, learning_rate, batch_size):
    min_len = min(len(train_losses), len(validation_losses), len(f1_scores), len(recall_scores), len(validation_f1_scores))
    epochs = range(1, min_len + 1)

    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses[:min_len], label='Train Loss', marker='o', color='red')
    plt.plot(epochs, validation_losses[:min_len], label='Validation Loss', marker='x', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Plot F1 Score and Recall
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores[:min_len], label='Train F1 Score', marker='o', color='blue')
    plt.plot(epochs, validation_f1_scores[:min_len], label='Validation F1 Score', marker='s', color='purple')
    plt.plot(epochs, recall_scores[:min_len], label='Train Recall', marker='x', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('F1 Score and Recall per Epoch')
    plt.legend()
    plt.grid(True)

    # Annotate Learning Rate and Batch Size
    plt.figtext(0.15, 0.05, f'Learning Rate: {learning_rate}', fontsize=10, color='black')
    plt.figtext(0.15, 0.01, f'Batch Size: {batch_size}', fontsize=10, color='black')

    plt.tight_layout()
    save_path = filename + ".png"
    plt.savefig(save_path)
    print(f"[INFO] Training plot saved to: {save_path}")

    if matplotlib.is_interactive():
        plt.show()


def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'/user/HS401/kk01579/APML_PROJECT_KAMALS/src/visuals/{name.lower()}_confusion_matrix_classweights_main_1.png')
    print(f"[INFO] Confusion matrix saved to: {name.lower()}_confusion_matrix_main_1.png")
    plt.close()
