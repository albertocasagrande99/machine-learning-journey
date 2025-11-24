# ✨ 05-generative-models: Deep Dive into Modern Generative AI

This folder contains a comprehensive sequence of notebooks exploring the most important architectures in generative artificial intelligence. We move from foundational methods (Autoencoders) to modern state-of-the-art techniques (Diffusion Models), focusing on intuition, PyTorch implementation, and key training concepts.

## 🎯 Notebooks

| # | Notebook | Topic | Key concepts & highlights |
| --- | --- | --- | --- |
| 1 | `01-autoencoders.ipynb` | Autoencoders (AE) | Introduction to the Encoder-Decoder architecture and the concept of a compressed **latent space**. Focuses on self-supervised learning, reconstruction loss (MSE), and practical applications like **dimensionality reduction** and **denoising**. |
| 2 | `02-variational-autoencoders.ipynb` | Variational Autoencoders (VAE) | Addresses the "unstructured" latent space problem of AEs by introducing **probabilistic encoding**. Deep dive into the VAE loss function: **Reconstruction Loss** + **KL Divergence** (regularization). Explains the **reparameterization trick** and sampling from a structured Gaussian latent space.  |
| 3 | `03-generative-adversarial-networks.ipynb` | Generative Adversarial Networks (GANs) | Covers the foundational concept of the **adversarial game** between a Generator (forger) and a Discriminator (critic). Implements a simple MLP-based GAN and transitions to a **Deep Convolutional GAN (DCGAN)**, explaining the role of transpose convolutions for image generation. |
| 4 | `04-conditional-gans.ipynb` | Conditional Generative Adversarial Networks (cGANs) | Extends GANs to enable control over the generated output by introducing a **condition (c)** (e.g., a class label). Shows how the condition is integrated into **both** the Generator and Discriminator to model the conditional distribution P(x\|c). |
| 5 | `05-diffusion-models.ipynb` | Denoising Diffusion Probabilistic Models (DDPM) | Explores the modern state-of-the-art generative approach. Details the two-stage process: the fixed **Forward Diffusion** (destruction) process and the learned **Reverse Diffusion** (creation/denoising) process. Implements the core architecture, which uses a **U-Net** to predict the noise at each step.  |