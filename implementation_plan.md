# Kế hoạch thực nghiệm lại Paper "Deep Ensembles: A Loss Landscape Perspective" trên Kaggle

## Tổng quan về paper

Paper **"Deep Ensembles: A Loss Landscape Perspective"** (Fort, Hu, Lakshminarayanan, 2019, arXiv:1912.02757) nghiên cứu câu hỏi: **Tại sao deep ensembles hoạt động tốt hơn single deep neural networks?**

### Phát hiện chính:
1. Các **snapshots** (mô hình từ epoch 1, epoch 2,...) của **cùng một mô hình** thể hiện **functional similarity** → ensemble của chúng **ít khả năng** khám phá các modes khác nhau của local minima
2. Các **solutions khác nhau** (trained với random initializations khác nhau) thể hiện **functional dissimilarity** → ensemble của chúng **nhiều khả năng** khám phá các modes khác nhau

---

## Thông tin phần cứng và nền tảng

> [!IMPORTANT]
> Vì không có GPU rời trên máy local, toàn bộ thí nghiệm sẽ được thực hiện trên **Kaggle Notebooks** (miễn phí GPU P100/T4, 30 giờ/tuần).

### Kaggle GPU Resources (2026):
- **GPU**: NVIDIA Tesla P100 (16GB VRAM) hoặc T4 (16GB VRAM)
- **RAM**: 13GB
- **Disk**: 73GB (+ 20GB output)
- **Session timeout**: 12 giờ liên tục
- **Quota**: 30 giờ GPU/tuần

---

## Kiến trúc mô hình cần thực nghiệm

| Mô hình | Số params | Mô tả |
|---------|-----------|-------|
| **SmallCNN** | ~15,722 | 3 Conv2D (16→32→32) + MaxPool + GAP + Dense(32) + Dropout(0.1) |
| **MediumCNN** | ~89,226 | 4 Conv2D (32→64→128→128) + MaxPool + Dropout(0.1) + GAP + Dense(32) |
| **ResNet20v1** | ~272,474 | ResNet20 từ keras-idiomatic-programmer |

---

## Kế hoạch thực nghiệm chi tiết

### Phase 1: Chuẩn bị môi trường (Notebook 0)
- [ ] Tạo Kaggle account (nếu chưa có)
- [ ] Verify phone number để unlock GPU
- [ ] Tạo notebook mới, chọn **GPU P100/T4** làm accelerator
- [ ] Cài đặt thư viện: `wandb`, `tensorflow`, `tensorflow_addons`, `scikit-learn`
- [ ] Đăng nhập W&B (Weights & Biases) để tracking experiments:
  ```python
  !pip install wandb
  import wandb
  wandb.login()
  ```
- [ ] Tải dataset CIFAR-10:
  ```python
  from tensorflow.keras.datasets import cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  ```

> [!NOTE]
> **Lưu ý về W&B**: W&B là công cụ tracking experiment được sử dụng trong repository gốc. Bạn cần tạo tài khoản tại [wandb.ai](https://wandb.ai) (miễn phí). Nếu không muốn dùng W&B, có thể thay thế bằng TensorBoard hoặc log thủ công.

---

### Phase 2: Training các mô hình (Notebooks 1-3)

#### 2.1 Training SmallCNN — Notebook 1
- [ ] Implement kiến trúc SmallCNN:
  - Input: (32, 32, 3)
  - Conv2D(16, 3x3, same) → ReLU → MaxPool(2)
  - Conv2D(32, 3x3, same) → ReLU → MaxPool(2)
  - Conv2D(32, 3x3, same) → ReLU → MaxPool(2)
  - GlobalAveragePooling2D
  - Dense(32, relu) → Dropout(0.1)
  - Dense(10, softmax)
- [ ] Cấu hình training:
  - **Optimizer**: Adam
  - **Loss**: sparse_categorical_crossentropy
  - **Batch size**: 128
  - **Epochs**: 40
  - **LR Schedule** (theo paper):
    - Epoch 0-8: lr = 1.6e-3
    - Epoch 9-18: lr = 8e-4
    - Epoch 19-28: lr = 4e-4
    - Epoch 29+: lr = 2e-4
- [ ] Callbacks:
  - LearningRateScheduler
  - Model checkpoint (lưu model mỗi epoch - **quan trọng** cho phân tích snapshot)
  - Confusion matrix logging (qua W&B hoặc local)
- [ ] **Train 5 lần** với random initializations khác nhau (set random seed khác nhau)
- [ ] Lưu tất cả model weights vào Kaggle Output

#### 2.2 Training MediumCNN — Notebook 2
- [ ] Implement kiến trúc MediumCNN:
  - Input: (32, 32, 3)
  - Conv2D(32, 3x3, same) → ReLU → Dropout(0.1) → MaxPool(2)
  - Conv2D(64, 3x3, valid) → ReLU → Dropout(0.1) → MaxPool(2)
  - Conv2D(128, 3x3, same) → ReLU → Dropout(0.1) → MaxPool(2)
  - Conv2D(128, 3x3, same) → ReLU → Dropout(0.1) → MaxPool(2)
  - GlobalAveragePooling2D
  - Dense(32, relu) → Dropout(0.1)
  - Dense(10, softmax)
- [ ] Cùng cấu hình training như SmallCNN
- [ ] **Train 5 lần** với random initializations khác nhau
- [ ] Lưu tất cả model weights

#### 2.3 Training ResNet20v1 — Notebook 3
- [ ] Download `resnet_cifar10.py` từ [keras-idiomatic-programmer](https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer):
  ```python
  !wget https://raw.githubusercontent.com/GoogleCloudPlatform/keras-idiomatic-programmer/master/zoo/resnet/resnet_cifar10.py
  ```
- [ ] Implement ResNet20v1 (n=2, depth=20):
  ```python
  def get_training_model():
      n = 2
      depth = n * 9 + 2
      n_blocks = ((depth - 2) // 9) - 1
      inputs = Input(shape=(32, 32, 3))
      x = resnet_cifar10.stem(inputs)
      x = resnet_cifar10.learner(x, n_blocks)
      outputs = resnet_cifar10.classifier(x, 10)
      model = Model(inputs, outputs)
      return model
  ```
- [ ] Train **không có data augmentation** (5 lần, random init)
- [ ] Train **có data augmentation** (5 lần, random init):
  ```python
  def augment(image, label):
      image = tf.image.resize_with_crop_or_pad(image, 40, 40)
      image = tf.image.random_crop(image, size=[32, 32, 3])
      image = tf.image.random_brightness(image, max_delta=0.5)
      image = tf.clip_by_value(image, 0., 1.)
      return image, label
  ```
- [ ] Lưu tất cả model weights

> [!TIP]
> **Mẹo tiết kiệm GPU quota trên Kaggle:**
> - Mỗi notebook chỉ train 1 mô hình, 1 lần
> - Sử dụng Kaggle Datasets để lưu weights giữa các sessions
> - Batch training: train nhiều initializations trong 1 notebook nếu đủ thời gian
> - SmallCNN và MediumCNN train rất nhanh (~5-10 phút/run), ResNet20 chậm hơn (~20-30 phút/run)

---

### Phase 3: Phân tích Ensemble Accuracy (Notebooks 4-6)

Thí nghiệm này kiểm tra **accuracy dưới dạng hàm của ensemble size**.

#### 3.1 SmallCNN Val Acc Ensembles — Notebook 4
- [ ] Load tất cả model weights đã train (5 runs)
- [ ] Với mỗi epoch, tính ensemble predictions bằng **averaging logits/probabilities**
- [ ] So sánh accuracy của:
  - Single model
  - Ensemble 2 models
  - Ensemble 3 models
  - Ensemble 4 models
  - Ensemble 5 models
- [ ] Vẽ biểu đồ **Val Accuracy vs Ensemble Size**

#### 3.2 MediumCNN Val Acc Ensembles — Notebook 5
- [ ] Tương tự như SmallCNN

#### 3.3 ResNet20v1 Val Acc Ensembles — Notebook 6
- [ ] Tương tự, cho cả phiên bản có và không data augmentation

---

### Phase 4: Phân tích Function Space Similarity (Notebooks 7-9)

Đây là thí nghiệm **quan trọng nhất** của paper, bao gồm 3 phân tích:

#### 4.1 Cosine Similarity Analysis
- [ ] Tính **cosine similarity** giữa weight vectors:
  - Giữa các **snapshots** (epoch khác nhau, cùng init) → kỳ vọng: **high similarity**
  - Giữa các **trajectories** (cùng epoch, khác init) → kỳ vọng: **lower similarity**
- [ ] Visualize bằng heatmap

#### 4.2 Prediction Disagreement Analysis
- [ ] Tính **prediction disagreement rate** giữa các models:
  - Giữa snapshots → kỳ vọng: **low disagreement**
  - Giữa trajectories → kỳ vọng: **higher disagreement**
- [ ] Visualize bằng heatmap

#### 4.3 tSNE Visualization
- [ ] Flatten weight vectors thành 1D
- [ ] Áp dụng **tSNE** để visualize:
  - Trajectory của mỗi training run (qua các epochs)
  - So sánh vị trí của các solutions khác nhau trong weight space
- [ ] Kỳ vọng: Các snapshots cluster gần nhau, các trajectories khác nhau tách xa

> [!WARNING]
> **tSNE với ResNet20v1**: Repository gốc **không** có tSNE visualization cho ResNet20v1 vì vector weight quá lớn. Trên Kaggle với 13GB RAM, cần cẩn thận với memory. Có thể:
> - Chỉ sử dụng subset weights (ví dụ: chỉ last few layers)
> - Hoặc sử dụng UMAP thay vì tSNE (nhanh hơn, ít tốn memory hơn)

---

### Phase 5: Tổng hợp và so sánh kết quả (Notebook 10)

- [ ] So sánh kết quả thực nghiệm với paper gốc
- [ ] Tạo bảng tổng hợp metrics cho tất cả mô hình
- [ ] Viết nhận xét và kết luận
- [ ] Tạo report final (có thể dùng W&B Reports hoặc Kaggle Discussion)

---

## Cấu trúc thư mục trên Kaggle

```
📁 Kaggle Datasets (lưu model weights)
├── smallcnn_run1/ (40 checkpoint files)
├── smallcnn_run2/
├── smallcnn_run3/
├── smallcnn_run4/
├── smallcnn_run5/
├── mediumcnn_run1/
├── mediumcnn_run2/
├── ...
├── resnet20_noaug_run1/
├── resnet20_noaug_run2/
├── ...
├── resnet20_aug_run1/
├── resnet20_aug_run2/
└── ...
```

---

## Timeline dự kiến

| Tuần | Công việc | GPU Time ước tính |
|------|-----------|-------------------|
| **Tuần 1** | Setup + Train SmallCNN (5 runs) + MediumCNN (5 runs) | ~3-4 giờ |
| **Tuần 2** | Train ResNet20v1 no aug (5 runs) + aug (5 runs) | ~8-10 giờ |
| **Tuần 3** | Ensemble accuracy analysis (3 notebooks) | ~2-3 giờ |
| **Tuần 4** | Function Space Similarity analysis (3 notebooks) | ~4-6 giờ |
| **Tuần 5** | Tổng hợp, so sánh, viết report | ~1-2 giờ |

> [!IMPORTANT]
> **Tổng GPU time ước tính**: ~18-25 giờ, nằm trong quota miễn phí Kaggle (30 giờ/tuần). Nếu cần nhiều hơn, có thể spread qua 2 tuần hoặc giảm số runs xuống 3 thay vì 5.

---

## Verification Plan

### Automated Tests
- So sánh training accuracy/loss curves với kết quả từ repository gốc (report trên W&B)
- Verify ensemble accuracy tăng theo ensemble size
- Verify cosine similarity patterns (snapshots > trajectories)
- Verify prediction disagreement patterns (snapshots < trajectories)

### Manual Verification
- So sánh visual tSNE plots với hình trong paper gốc
- Kiểm tra training time phù hợp với hardware
- Review confusion matrices để đảm bảo model train đúng

---

## Tài liệu tham khảo

- **Paper gốc**: [Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/abs/1912.02757) (Fort, Hu, Lakshminarayanan, 2019)
- **Repository gốc**: [ayulockin/LossLandscape](https://github.com/ayulockin/LossLandscape)
- **Report gốc**: [Understanding the Effectivity of Ensembles in Deep Learning](https://app.wandb.ai/authors/loss-landscape/reports/Understanding-the-effectivity-of-ensembles-in-deep-learning-(tentative)--VmlldzoxODAxNjA)
- **Yannic Kilcher's explanation**: [YouTube video](https://www.youtube.com/watch?v=5IRlUVrEVL8)
- **Model weights**: [GitHub Releases v0.1.0](https://github.com/ayulockin/LossLandscape/releases/tag/v0.1.0)
