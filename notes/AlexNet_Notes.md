# AlexNet Architecture Notes

AlexNet won the ImageNet ILSVRC challenge in 2012 by a huge margin, proving the viability of Deep Learning.

## Key Innovations
- **ReLU activation**: Instead of Tanh/Sigmoid, speeded up training significantly.
- **Dropout**: Addressed overfitting in the dense layers.
- **GPU training**: Parallelizing convolutions across 2 GPUs (grouped convolutions).
- **Data Augmentation**: Cropping, translation, color jittering.
- **Local Response Normalization (LRN)**: Replaced today by Batch Normalization.
