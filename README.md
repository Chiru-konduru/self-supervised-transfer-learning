# Self-Supervised Learning and Object Detection with PyTorch

This repository consists of two substantial disjoint parts aimed at gaining hands-on experience with PyTorch, utilizing pre-trained models from the deep learning community, and adapting these models to new tasks and losses. The first part involves a self-supervised learning task for pre-training a feature representation on the CIFAR10 dataset, followed by fine-tuning for CIFAR10 classification. The second part entails the implementation and training of a YOLO-based object detector on the PASCAL VOC 2007 dataset.

## Table of Contents

- [Description](#Description)
- [Implementation](#Implementation)
- [Extra Credit](#Extra-Credit)
- [Acknowledgements](#Acknowledgements)

## Description

### Part 1: Self-Supervised Learning and Fine-Tuning

In this part, a model is trained on a self-supervised task of image rotation prediction without using the CIFAR10 class labels, followed by fine-tuning a subset of the model's weights and fully supervised training. The model used for this part is ResNet18, and the PyTorch implementation of the same is used for training.

### Part 2: YOLO Object Detection on PASCAL VOC 2007

In this part, a YOLO-like object detector is implemented and trained on the PASCAL VOC 2007 dataset. The focus is on implementing the loss function of YOLO, with a pre-trained network structure provided for the model.

## Implementation

### Part 1: Self-Supervised Learning and Fine-Tuning

The top-level notebook, `a3_part1_rotation.ipynb`, guides through the process of training a ResNet for the rotation task and fine-tuning it for the classification task. Tasks include:

- Training a ResNet18 on the rotation task.
- Fine-tuning the weights of the final block of convolutional layers and linear layer on the supervised CIFAR10 classification task.
- Training the full network on the supervised CIFAR10 classification task.
- 
### Extra Credit

Used a more advanced model than ResNet18 to try to get higher accuracy on the rotation prediction task, as well as for transfer to supervised CIFAR10 classification.
For rotation prediction task:
We used resnet50 with the parameters : num_epochs=45, decay_epochs=5, init_lr=0.01, task='rotation' and got the accuracy of 84.89% which is higher than resnet18 (81.51%).

For Classification task:
We used resnet50 with the parameters : num_epochs=20, decay_epochs=4, init_lr=0.1,
task='classification' and got the accuracy of 68.70% which is higher than resnet18(57.43%).

### Part 2: YOLO Object Detection on PASCAL VOC 2007

The top-level notebook, `MP3_P2.ipynb`, provides a guide to implement a YOLO-like object detector. The focus is mainly on implementing the loss function of YOLO.

### Extra Credit

For extra credit, we have replaced the provided pre-trained network with a different one (maskrcnn_resnet50_fpn_v2, Resnet152, and fasterrcnn)and trained with the YOLO loss on top to attempt to get better accuracy.


## Acknowledgements

This project has been done as a part of a course assignment. We would like to thank the course instructors and TAs for their guidance and support throughout the project.
Please note that this project is meant for educational purposes and should be used responsibly.

---

Please refer to the respective Jupyter notebooks for more details on the project implementation, results, and discussions. If you have any questions or suggestions, feel free to open an issue.
