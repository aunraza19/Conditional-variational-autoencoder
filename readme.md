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

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/aunraza19/Conditional-variational-autoencoder.git
```
### 2. Change directory into the project
```bash
cd CVAE-MNIST
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Train the model
```bash
python main.py
```
## Outputs:
outputs/sample_epoch_5.png — Generated digits at epoch 5

outputs/sample_epoch_10.png — Generated digits at epoch 10

outputs/loss_plot.png — Training loss over epochs

outputs/digit_transform_7_to_2.png — Creative twist result: 7 transformed into 2

 ## Evaluation
 Visual Samples:
Synthetic digits generated from latent space and labels

 Reconstruction Loss:
Binary cross-entropy + KL divergence

 ## Creative Twist
Use of the decoder with altered labels to transform one digit into another

## License
This project is released for academic purposes. Feel free to fork and build upon it.

