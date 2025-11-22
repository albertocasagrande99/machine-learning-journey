# 📷 04-computer-vision: Practical Computer Vision Tutorials

This folder collects hands-on notebooks that walk through core computer vision tasks: image classification, object detection, semantic segmentation, and modern Transformer-based vision models. Each notebook mixes intuition, runnable PyTorch code, and small experiments.

## 🧭 Notebooks

| # | Notebook | Topic | Key concepts & highlights |
| --- | --- | --- | --- |
| 1 | `01-image-classification-architectures.ipynb` | Image Classification Architectures | Deep dive into ResNet (skip connections) and EfficientNet (compound scaling). Includes code using `torchvision.models.resnet50` and `timm`'s `efficientnet_b0`, demonstration of model-specific preprocessing, Top-5 inference on ImageNet labels, and practical tips for transfer learning and choosing backbones. |
| 2 | `02-object-detection.ipynb` | Object Detection | Covers two-stage (Faster R-CNN) and one-stage (YOLOv8) approaches. Demonstrates `torchvision`'s `fasterrcnn_resnet50_fpn_v2` (with COCO classes) and `ultralytics` YOLOv8 (yolov8n). Shows preprocessing, running inference, drawing boxes, reading raw outputs, and discusses anchors, RPN, FPN, NMS, speed vs accuracy trade-offs. |
| 3 | `03-semantic-segmentation.ipynb` | Semantic Segmentation | Builds a U-Net from scratch (DoubleConv, encoder–decoder, skip connections), trains it on the Oxford-IIIT Pet segmentation masks, and demonstrates DeepLabV3 (pretrained deeplabv3_resnet101) with Atrous convolutions and ASPP. Includes data pipelines, loss (CrossEntropy), training loop, visualization, and evaluation tips (pixel accuracy, mIoU). |
| 4 | `04-vision-transformers.ipynb` | Vision Transformers (ViT) | Explains patch embedding, CLS token, positional embeddings, and Transformer encoders. Implements a PatchEmbedding layer from scratch and uses Hugging Face `transformers` (google/vit-base-patch16-224) for inference and transfer learning. Discusses inductive bias differences between CNNs and ViTs and when to prefer each. |
