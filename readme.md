# Conditional Variational Autoencoder (CVAE) on MNIST

This project implements a Conditional Variational Autoencoder (CVAE) trained on the MNIST dataset. It allows the generation of handwritten digits conditioned on their class labels (0–9), and demonstrates digit-to-digit transformation by manipulating labels during decoding.

##  Project Summary
- **Base Model:** Variational Autoencoder (VAE)
- **Innovation:** Conditional VAE — label is input to both encoder and decoder
- **Dataset:** MNIST (handwritten digits 0–9)
- **Framework:** PyTorch
- **Creative Twist:** Change the label of a digit to reconstruct it as a different digit (e.g., 7 ➝ 2)


##  Features
- Conditional image generation using digit labels (0–9)
- Modular code (train, test, model, dataloader)
- CVAE with reparameterization trick
- PyTorch-based implementation

##  How to Run


### 1. Clone the repository
```bash
git clone https://github.com/your-username/CVAE-MNIST.git
cd CVAE-MNIST


### 2. Install dependencies:
```bash
   pip install -r requirements.txt

### 3. Train the model
```bash
   python main.py

### 4. Outputs
Reconstructed digits vs. original

New digits generated from random latent vectors

Digit 7 decoded as digit 2 (creative transformation)

Loss curve saved as image

###5. Evaluation
# Visual Samples:
outputs/sample_epoch_5.png

outputs/sample_epoch_10.png

# Reconstruction Loss:
outputs/loss_plot.png

# modifying label:
outputs/digit_transform_7_to_2.png: Original "7" reconstructed as "2" by modifying the label

