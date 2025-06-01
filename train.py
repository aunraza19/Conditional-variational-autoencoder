import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def train_model(model, train_loader, optimizer, device, num_epochs=10):
    model.train()
    losses = []

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            x_reconst, mu, logvar = model(images, labels)

            # Compute losses
            reconst_loss = F.binary_cross_entropy(
                x_reconst, images.view(-1, 28 * 28), reduction='sum'
            )
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconst_loss + kl_div

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Record average loss per image
        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save generated samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            z = torch.randn(10, model.latent_dim).to(device)
            sample_labels = torch.arange(10).to(device)
            with torch.no_grad():
                samples = model.decode(z, sample_labels)

            samples = samples.view(-1, 28, 28).cpu().numpy()
            fig, axes = plt.subplots(1, 10, figsize=(12, 2))
            for i, ax in enumerate(axes):
                ax.imshow(samples[i], cmap="gray")
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"outputs/sample_epoch_{epoch + 1}.png")
            plt.close()
            model.train()

    # Plot loss curve
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.title("Reconstruction Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("outputs/loss_plot.png")
    plt.close()
