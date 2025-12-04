# ⚙️ 06-advanced-techniques: Advanced Deep Learning Techniques

This folder collects practical notebooks that teach advanced, high-impact techniques used in modern deep learning workflows: transfer learning, fine-tuning, self-supervised & contrastive learning, multi-task learning, few-shot learning and model compression via knowledge distillation.

## 📚 Notebooks

| # | Notebook | Topic | Key concepts & highlights |
|---:|---|---|---|
| 1 | `01-transfer-learning.ipynb` | Transfer Learning (theory & strategies) | Explains Feature Extraction vs Fine-Tuning, layer freezing/unfreezing, differential learning rates, ImageNet normalization, and a decision matrix for choosing strategies. Includes PyTorch examples for replacing heads and partially unfreezing ResNet blocks. |
| 2 | `02-resnet-transfer-learning.ipynb` | ResNet18 → CIFAR-10 (Practical TL) | Hands-on: use a pre-trained ResNet18 as a fixed feature extractor for CIFAR-10, freeze backbone, replace head, train only head, then unfreeze `layer4` for fine-tuning. Shows transforms, training/eval loops, and accuracy visualization. |
| 3 | `03-transformers-fine-tuning.ipynb` | Transformers Fine-Tuning (DistilBERT) | NLP-focused TL: load DistilBERT with Hugging Face, prepare IMDb dataset with `datasets`, tokenize, use `Trainer` API for fine-tuning and `pipeline` for inference. Demonstrates low LR fine-tuning and evaluation metrics. |
| 4 | `04-self-supervised-learning.ipynb` | Self-Supervised Learning (RotNet) | Implements a pretext task (rotation prediction) on CIFAR-10 to learn representations from unlabeled data. Builds an SSL CNN backbone, trains the pretext head, then re-uses learned features for a downstream CIFAR-10 classifier (fixed feature extractor strategy). |
| 5 | `05-contrastive-learning.ipynb` | Contrastive Learning (SimCLR) | Implements SimCLR: strong augmentations that produce two views, a ResNet-18 encoder adapted to small images, projection head, NT-Xent loss, and t-SNE visualization of learned features. Discusses temperature, batch size, and why contrastive objectives produce high-quality embeddings. |
| 6 | `06-multi-task-learning.ipynb` | Multi-Task Learning (Hard Parameter Sharing) | Example MTL setup on MNIST where a shared backbone predicts both digit identity and parity (even/odd). Shows dataset wrapping, two task-specific heads, combined loss (weighted sum), and training/inference for both tasks. Covers loss balancing considerations. |
| 7 | `07-few-shot-learning.ipynb` | Few-Shot Learning (Prototypical Networks) | Episodic training using EasyFSL and CIFAR-100: implements ProtoNets (N-way K-shot), ResNet backbone modifications for small images, episodic samplers, training loop and evaluation on unseen classes. Includes visualization of episodes and support vs query sets. |
| 8 | `08-knowledge-distillation.ipynb` | Knowledge Distillation (Teacher → Student) | Demonstrates KD on CIFAR-10: train a strong Teacher (ResNet-18), define a small Student CNN, implement combined KD loss (KL on softened logits + CE on hard labels), and compare distilled student vs baseline student trained from scratch. Includes temperature and alpha hyperparameters. |
