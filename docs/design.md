# Design note

Target: ValueLabs, Computer Vision Engineer. The posting asks for CNNs trained from scratch, architecture selection and transfer learning, hyperparameter optimisation, model versioning and a registry, reproducible training, explainability, and monitoring of model performance.

## Question

Can one small pipeline cover that lifecycle on a public image dataset with honest measurements?

## Method

- Data: CIFAR-10. Train on a fixed 20,000-image subset, validate on 5,000 other training images, test on the official 10,000 test images.
- Compare three architectures with the same schedule: a small CNN from scratch, ResNet-18 from scratch, and ResNet-18 with ImageNet weights fine-tuned (input upsampled to 64 px).
- Search: random search over learning rate, weight decay and batch size on the small CNN, 8 trials, 5 epochs each. Compare the best trial with the default configuration.
- Registry: files plus an index. Each version stores the configuration hash, a checkpoint SHA-256 and metrics. A new version becomes production only if validation accuracy beats the incumbent by 0.5 points. Checkpoint tampering is detected.
- Explainability: Grad-CAM, scored with a deletion test. Mask the top 20% of CAM pixels and compare the drop in true-class probability with masking a random square of the same area.
- Monitoring: Kolmogorov-Smirnov distance between the reference confidence distribution and each 200-image batch, with a threshold from clean calibration batches (95th percentile). Test with four corruptions.

## Scope limits

Single GPU, no distributed training, no cloud training service. One training run per setting.

## Change after the first run

The first deletion test compared the CAM mask with a mask of randomly scattered pixels. Scattered pixels act like noise and hurt any model, so the baseline was unfair. The baseline is now a random square patch of the same area.
