
# 📚 02-deep-learning-architectures: Modern Neural Network Architectures

This folder contains hands-on notebooks that dive into important deep learning architectures. Each notebook mixes theory, worked examples, and runnable PyTorch (and optional PyG/torchtext) code. The notebooks use MNIST for many image examples and smaller synthetic or demo datasets for faster experimentation.

## 🎯 Notebooks

| \# | Notebook | Topic | Key concepts & highlights |
| --- | --- | --- | --- |
| 1 | `01-multi-layer-perceptrons.ipynb` | Multi-Layer Perceptrons (MLPs) | Perceptron to MLPs, activation functions (sigmoid, tanh, ReLU, LeakyReLU, softmax), forward/backward math, implement an MLP from scratch (NumPy) and practical usage with scikit-learn's `MLPClassifier` and a small PyTorch example. Includes XOR demo, visualization and hyperparameter tuning. |
| 2 | `02-convolutional-neural-networks.ipynb` | Convolutional Neural Networks (CNNs) — MNIST | Intuition and formulas for 2D convolutions, padding/stride, channels and receptive field. Full PyTorch MNIST tutorial: DataLoader setup, simple CNN (Conv-BN-ReLU blocks), training loop, checkpointing, visualization of filters & feature maps, and tips for computing spatial dimensions. |
| 3 | `03-recurrent-neural-networks.ipynb` | Recurrent Neural Networks (RNNs) / LSTMs | Lightweight NLP pipeline: tokenization, Vocab, padding/collate, and `TextDataset`. Builds an Embedding + LSTM classifier (supports pack/padded sequences, bidirectional LSTM), training/evaluation utilities, and shows saving/loading for inference. Useful as a compact text-classification template. |
| 4 | `04-graph-neural-networks.ipynb` | Graph Neural Networks (GNNs) — Cora | Message-passing intuition, GCN formula, PyTorch Geometric (PyG) usage (optional install), loading Planetoid datasets (Cora), building a 2-layer GCN, full-batch training with masks, and inspecting node-level predictions. Good for node-classification experiments. |
| 5 | `05-autoencoders.ipynb` | Autoencoders & VAEs | From deterministic Autoencoders (AE) to Variational Autoencoders (VAE): encoder/decoder design, reconstruction vs KL losses, reparameterization trick, training on MNIST, visualizing reconstructions, and generating samples from the VAE latent space. Includes practical notes on loss choices and convolutional variants. |
| 6 | `06-GANs.ipynb` | Generative Adversarial Networks (GANs) — Vanilla & DCGAN | End-to-end GAN tutorial: adversarial objective, MLP-based (vanilla) GAN on MNIST, training loop details and common tricks (non-saturating loss, label handling), then a DCGAN implementation (conv/transposed-conv, weight init, batchnorm) with training and side-by-side comparisons. Includes visualization of generated samples and diagnostic tips. |
