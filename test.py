import torch
import matplotlib.pyplot as plt

def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            x_reconst, _, _ = model(images, labels)

            # Show original and reconstructed images
            original = images[0].view(28, 28).cpu().numpy()
            reconstructed = x_reconst[0].view(28, 28).cpu().numpy()

            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(original, cmap='gray')

            plt.subplot(1, 2, 2)
            plt.title("Reconstructed")
            plt.imshow(reconstructed, cmap='gray')

            plt.show()
            break
