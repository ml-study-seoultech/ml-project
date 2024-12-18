# 21101234 최동주 프로젝트

# Problem

## Loss image recovering

- Loss(masked)된 부분이 있는 흑백 이미지로부터 원본 컬러 이미지 복원
- Specific Task : Computer Vision - **MIM : Masked Image Modeling / Coloring / Inpainting**

Contest link : [https://dacon.io/competitions/official/236420/overview/description](https://dacon.io/competitions/official/236420/overview/description)

# Dataset

### Structure

```markdown
open/
│
├── train_input/         # Train data 
│   ├── TRAIN_00000.png  # 29603 512x512 masked grayscale image
│   ├── TRAIN_00001.png
│   ├── TRAIN_00002.png
│   └── ...
│
├── train_gt/            # Train_target data (Ground Truth)
│   ├── TRAIN_00000.png  # 29603 512x512 original colored image
│   ├── TRAIN_00001.png
│   ├── TRAIN_00002.png
│   └── ...
│
├── test_input/          # Test data
│   ├── TEST_00000.png   # 100 512x512 masked grayscale image
│   ├── TEST_00001.png
│   ├── TEST_00002.png
│   └── ...
│
├── train.csv             # triain_input / train_gt image path
└── test.csv             # test_input image path
```

## Sample Data

![스크린샷 2024-12-18 17.33.55.png](./images/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-18_17.33.55.png)

# First approach : GAN

생성적 적대 생성망 GAN(Generative Adversarial Network)

![스크린샷 2024-12-18 17.42.15.png](./images/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-18_17.42.15.png)

GAN은 이미지를 만들어내는 Generator와, 이를 판별하는 Discriminator로 구별됨.

Generator는 실제(now train_gt)와 같은 이미지를 생성하려 하고, Discriminator는 실제 이미지와 생성된 이미지를 구분하려고 한다. 이러한 적대적 학습을 통해 Generator는 점점 더 실제와 유사한 이미지를 생성할 수 있게 된다.

첫 번째 시도는 GAN이 mask 부분 복구와 coloring을 동시에 수행하는 것을 목표로 함.

## Process overview

1. 입력이 train_input(masked grayscale image)인 Generator 정의.
2. Generator가 mask 영역 복원 및 coloring을 동시 수행
3. Discriminator가 train_gt(원본 이미지)와 비교해서 판별
4. Loss 바탕으로 Generator는 더 train_gt와 같은 이미지를 생성하게 됨

### Generator

```python
# Generator
class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet", # pre-trained weight 사용
            in_channels=1, # train_input은 흑백 이미지
            classes=3, # 생성할 이미지는 컬러(3채널)
        )

    def forward(self, x):
        return self.model(x)

```

### U-net

![Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation*. In *International Conference on Medical Image Computing and Computer-Assisted Intervention* (pp. 234–241). Springer, Cham.](./images/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-17_13.19.43.png)

Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation*. In *International Conference on Medical Image Computing and Computer-Assisted Intervention* (pp. 234–241). Springer, Cham.

U-net은 CNN을 이용한 Encoder-decoder 모델로, 이미지를 압축된 형태로 encoding 후 다시 원래 크기로 복원하는 구조를 가짐.  Encoder→Decoder의 layer를 직접 연결하는 skip connection을 사용하여 세부 정보의 손실을 방지하고, gradient vanishing 문제를 완화하는 특징이 있어 GAN에 자주 사용되는 것으로 알려져 있음.

인코더로 pre-trained된 resnet을 사용함.

또한 중요한 이유로, dataset image 크기가 512 x 512인데, pre-trained resnet34의 기본 입력 크기는 224 x 224 ****이지만 U-net은 FCN(Fully Convolution Network)이므로 따로 resize하지 않아도 알아서 입출력이 크기에 맞게 잘 동작하므로, U-net을 사용함.

 

### Discriminator

```python
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() # Discriminator는 binary classifier이다
        )

    def forward(self, x):
        return self.model(x)

```

Disciminator는 Generator가 생성한 이미지가 원본과 동일한지 판별한다.

`nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),`

`nn.LeakyReLU(0.2, inplace=True)`

4x4 filter, stride 2, zero-padding(keras에서는 ‘same’) 으로 다운샘플링.

LeakyReLU로 gradient vanish 완화, α=0.2(x<0에서), inplace=True는 pytorch에서 텐서를 직접 수정할 수 있게 해서(복사 후 수정된 텐서 반환이 아닌) 메로리를 절약할 수 있음.

`nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
nn.BatchNorm2d(128),
nn.LeakyReLU(0.2, inplace=True),`

중간에 Batch Nomalization으로 training 안정화

`nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
nn.Sigmoid()` 

출력층에서, Discriminator는 binary classifier이므로 sigmoid를 사용한다.

### Loss

`Generator loss = adversarial_loss + (100*pixelwise_loss)`  (scale 맞추기 위해 x100)

`Discriminator loss = (loss_D_real + loss_D_fake)/2`  

| **모델** | **손실 구성 요소** | **목표** | **손실 함수** |
| --- | --- | --- | --- |
| **Generator** | Adversarial Loss  | Discriminator가 생성된 이미지를 **1**로 판별 | Binary Cross Entropy (BCE) |
|  | Pixelwise Loss  | 생성 이미지와 실제 이미지(train_gt)와 픽셀 값이 유사하게 | L1 Loss (절댓값 오차) |
| **Discriminator** | Real Image Loss (Loss_D_read) | 진짜 이미지(train_gt)를 **1**로 판별 | Binary Cross Entropy (BCE) |
|  | Fake Image Loss (Loss_D_fake) | 가짜 이미지(생성된 이미지)를 **0**으로 판별 | Binary Cross Entropy (BCE) |

**label(train_gt) = 1, label(generator_generated_image) = 0으로 설정**

---

### Other setting

- 이미지 증강을 사용하려고 했었으나 기존 이미지만 해도 학습 시간이 너무 오래 걸리는 바람에 사용하지 못함
- 본래 AgementedDataset이라는 이름으로, train_gt에서 랜덤 위치를 masking한 후 흑백으로 변경해서, 이것 또한 학습에 사용하려고 함
- GAN은 원래 학습이 불안정하기 때문에 early stopping을 사용하지 않고 꽤 많은 epoch를 돌린 후, tensorboard에 기록된 로그를 보고 최적의 모델을 찾고자 했음. 3epoch마다 checkpoint를 저장함.
- 

## Entire Training code

```python
# 코랩 환경에서 google drive 연동
from google.colab import drive
drive.mount('/content/drive')

### Unzip

# 이미 압축 풀었으면 삭제하기 위함 (/content/dataset은 코랩 instance 실행 중에만 접근 가능함)
!rm -rf /content/dataset

# 구글 드라이브에 압축 풀린 파일 전체를 올려서 하면 로드 속도가 너무 느려서, instance 실행 중에 생성되는 저장공간에 압축 풀어서 로드하면 훨씬 빠르다
!unzip -qq /content/drive/MyDrive/MLProject/open.zip -d /content/dataset/
# -qq : verbose=False

# 라이브러리 import
import random
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
!pip install segmentation_models_pytorch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings(action='ignore')
!pip install scikit-image
# 대회에서 평가지표가 ssim이다
from skimage.metrics import structural_similarity as ssim

# 디바이스 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Hyperparameter setting

CFG = {
    'IMG_SIZE':512, # 원래 이미지 크기도 512*512
    'EPOCHS':30,
    'BATCH_SIZE':32,
    'SEED':41
}

### Fix seed

# 시드 고정
def seed_everything(seed):
  # Python의 기본 난수 생성기 시드 설정
  random.seed(seed)  # random 모듈에서 생성되는 난수들을 고정시키기 위해 시드 설정
  # Python 해시 시드 설정 (파이썬의 해시 기반 객체 비교 등에 영향을 미친다고 함)
  os.environ['PYTHONHASHSEED'] = str(seed)  # 환경 변수로 설정된 해시 시드 값 고정
  # NumPy의 난수 생성기 시드 설정
  np.random.seed(seed)
  # PyTorch의 CPU 난수 생성기 시드 설정
  torch.manual_seed(seed)
  # PyTorch의 GPU 난수 생성기 시드 설정 (CUDA)
  torch.cuda.manual_seed(seed)
  # cudnn 연산 결과가 결정적(항상 같게)
  torch.backends.cudnn.deterministic = True  # True로 설정하면 cudnn 연산이 결정적으로 동작
  # cudnn에서 연산 속도가 빨라진다고 함
  torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

# train_input, train_gt 이미지 경로가 매핑되어있는 csv 파일 로드
df = pd.read_csv('/content/dataset/train.csv')
df # 한번 보기

### Dataset Split

# 데이터셋 분할
df = df.sample(frac=1).reset_index(drop=True)
data_size = len(df)
train_ratio = 0.8 # train set : 80% / validation set : 20%로 분할 (배포용이 아니므로 test set은 안함)

# train set과 validation set크기 계산
train_size = int(data_size * train_ratio)
val_size = data_size - train_size

# 데이터셋 분할
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# 확인용 출력
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Total size: {len(train_df) + len(val_df)}")

## Dataset Define

class OriginalDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    # pytorch dataset class의 필수 메서드들

    def __len__(self): # custom 객체에 len(object) 했을 때 반환할 값
        return len(self.df) # dataset instance 개수 반환

    # pytorch에서 instance를 로드할 때 이 메서드를 사용함, 객체를 인덱스로 접근 가능하게
    def __getitem__(self, idx):
        base_path = '/content/dataset'
        # train.csv에는 ./train_input/TRAIN_00000.png 이런 식으로 있다
        # 이미지 경로 얻기 위한 전처리
        input_image_rel_path = self.df.loc[idx, 'input_image_path'].lstrip('./')
        gt_image_rel_path = self.df.loc[idx, 'gt_image_path'].lstrip('./')

        input_image_path = os.path.join(base_path, input_image_rel_path)
        gt_image_path = os.path.join(base_path, gt_image_rel_path)

        # 이미지 로드
        input_image = Image.open(input_image_path).convert('L')  # 흑백 이미지
        gt_image = Image.open(gt_image_path).convert('RGB')      # 컬러 이미지

        # 이미지 전처리
        if self.transforms:
            input_image = self.transforms['input'](input_image)
            gt_image = self.transforms['gt'](gt_image)

        return input_image, gt_image

### Transform

# 이미지 전처리 (범위를 [-1,1]로 만들면 0 기준 대칭이기 때문에 활성화함수를 쓸 때 안정적이다)
original_input_transform = transforms.Compose([
    transforms.ToTensor(), # ToTensor 수행 시 [0,255] -> [0,1]로 바뀜
    transforms.Normalize(mean=[0.5], std=[0.5]) # 이러면 [0,1] -> [-1,1]로 바뀜
])

original_gt_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

original_transforms = {
    'input': original_input_transform,
    'gt': original_gt_transform
}

## Create Dataset

# 데이터셋 생성
original_dataset = OriginalDataset(train_df, transforms=original_transforms)

print(f"Original dataset size: {len(original_dataset)}")

## Dataloader

# train dataloader
train_loader = DataLoader(original_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

# validation Dataloader
val_dataset = OriginalDataset(val_df, transforms=original_transforms)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

### GAN Model define

# Generator
class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet34", # 인코더로 resnet34 사용
            encoder_weights="imagenet", # pre-trained 가중치 로드
            in_channels=1, # train_input은 흑백 이미지이므로
            classes=3, # generator가 생성할 이미지는 컬러 이미지이므로
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            # binary classifier이다
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

## Loss Function & Optimizer

# 손실 함수 및 옵티마이저 설정
adversarial_loss = nn.BCELoss() # binary crossentropy
pixelwise_loss = nn.L1Loss() # abs(실제값 - 예측값)

generator = GeneratorUNet().to(device) # GPU로 모델 옮기기
discriminator = Discriminator().to(device)

# Generator는 Discriminator 피드백을 기반으로 학습하기 때문에 Dicriminator가 학습이 더 빨리 되는 것이 좋다고 한다
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999)) # Adam optimizer는 b1, b2가 있음(Momentum + RMSProp), Adam이 가장 보편적
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

## Train

!pip install pytorch-msssim

from pytorch_msssim import ssim  # GPU에서 SSIM 계산을 위한 라이브러리
from tqdm import tqdm  # progress bar 표시 tqdm

# SSIM 계산을 위한 메서드 ssim 계산하려면 denormalize 해야함
def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) # (x-mean)/std를 반대로
    return tensor

# validation
def validate(generator, dataloader):
    # validation에는 generator만 있으면 된다
    generator.eval()
    total_ssim = 0.0
    with torch.no_grad(): # validate, inference에는 그래디언트가 필요없음
        for input_image, gt_image in tqdm(dataloader, desc='Validation', unit='batch'):
            input_image = input_image.to(device)
            gt_image = gt_image.to(device)

            gen_output = generator(input_image)

            # Denormalize , [0,1] 범위 만들기
            gen_output_denorm = (gen_output * 0.5) + 0.5
            gt_image_denorm = (gt_image * 0.5) + 0.5

            # Denormalize 했을 때 [0,1] 범위를 벗어나지 않게 하기
            gen_output_denorm = torch.clamp(gen_output_denorm, 0, 1)
            gt_image_denorm = torch.clamp(gt_image_denorm, 0, 1)

            # SSIM 계산 (GPU 사용)
            ssim_value = ssim(gen_output_denorm, gt_image_denorm, data_range=1.0, size_average=True)
            total_ssim += ssim_value.item() * input_image.size(0)  # 배치 크기를 곱하여 총 SSIM 누적

    # 평균 SSIM 계산
    avg_ssim = total_ssim / len(dataloader.dataset)
    return avg_ssim

## Train & Validation

# 체크포인트 불러오기
checkpoint = torch.load('/content/drive/MyDrive/MLProject/saved/GAN/checkpoints/checkpoint_epoch30.pth')

#모델 불러오기
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

#optimizer 불러오기
optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

# epoch 정보 불러오기
#start_epoch = checkpoint['epoch'] + 1  # 다음 에포크부터 시작

#print(f"Start training epoch : {start_epoch}")

from torch.utils.tensorboard import SummaryWriter

# 로그 저장 디렉토리 설정
log_dir = '/content/drive/MyDrive/MLProject/saved/GAN/logs'
writer = SummaryWriter(log_dir=log_dir)

# GPU 메모리 비우기
import gc
gc.collect()
torch.cuda.empty_cache()

# Train
# Generator가 생성한 것은 가짜 이미지, train_gt는 진짜 이미지
real_label = 1.0
fake_label = 0.0

for epoch in range(CFG['EPOCHS']):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CFG['EPOCHS']}]", unit="batch")

    for batch_idx, (input_image, gt_image) in enumerate(train_loader_tqdm):
        input_image = input_image.to(device)
        gt_image = gt_image.to(device)

        batch_size = input_image.size(0)

        # 연산 시 차원 맞추기 위해서 크기 늘려주는것
        real_labels = torch.full((batch_size, 1, 1, 1), real_label, device=device)
        fake_labels = torch.full((batch_size, 1, 1, 1), fake_label, device=device)

        ####### Train Generator #######
        optimizer_G.zero_grad() #optimizer step 후 그래디언트 리셋

        gen_output = generator(input_image)

        pred_fake = discriminator(gen_output)

        # 레이블 크기를 pred_fake와 동일하게 생성
        real_labels = torch.ones_like(pred_fake, device=device)

        # Generator는 Discriminator가 진짜 이미지로 분류하게 하는 이미지를 생성하는 것이 목표.
        # Generator Loss는, Discriminator가 판별한 값 <->  1(실제 이미지는 레이블이 1이다) 사이의 loss(binary crossentropy)이다
        loss_G_adv = adversarial_loss(pred_fake, real_labels)

        # 픽셀 수준 loss로, 생성된 이미지와 ground truth 간의 픽셀값 차이에 대한 loss()
        loss_G_pixel = pixelwise_loss(gen_output, gt_image)

        loss_G = loss_G_adv + 100 * loss_G_pixel # 총 Generator loss는 adv_loss와 pixel loss의 합으로(scale 맞추기 위해서 *100)

        loss_G.backward() # 역전파
        optimizer_G.step() # optization

        ####### Train Discriminator #######
        optimizer_D.zero_grad()

        pred_real = discriminator(gt_image) # Discriminator가 train_gt를 진짜로 판별하도록 결과값 저장

        # 레이블 크기를 pred_real과 동일하게 생성
        real_labels = torch.ones_like(pred_real, device=device)

        loss_D_real = adversarial_loss(pred_real, real_labels) # Discriminator가 평가한 train_gt <-> 1 사이의 loss, Discriminator는 train_gt를 1(진짜)로 판별하도록 학습되어야 한다

        pred_fake = discriminator(gen_output.detach()) #discriminator(gen_output.detach()): gen_output.detach()는 Generator의 출력을 계산에서 분리하여, Discriminator가 Generator의 영향을 받지 않도록 합니다. 가짜 이미지에 대한 예측을 평가합니다.

        # 레이블 크기를 pred_fake와 동일하게 생성
        fake_labels = torch.zeros_like(pred_fake, device=device)

        loss_D_fake = adversarial_loss(pred_fake, fake_labels) #  Discriminator가 평가한 generator가 생성한 이미지 <-> 0 사이의 loss, generator가 생성한 이미지는 0으로 판별하도록 학습되어야 한다.

        loss_D = (loss_D_real + loss_D_fake) / 2 # 두 loss 평균을 총 Discriminator Loss로

        loss_D.backward()
        optimizer_D.step()

        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()

        # loss_G=2.345 이런 식으로 progress bar에 표시
        train_loader_tqdm.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        # TensorBoard에 batch당 loss 기록
        step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Loss/Generator', loss_G.item(), step)
        writer.add_scalar('Loss/Discriminator', loss_D.item(), step)

        # 100개의 batch 처리마다 출력으로 간단히 보여주기
        if (batch_idx + 1) % 100 == 0:
            input_img = denormalize(input_image[0].cpu(), mean=[0.5], std=[0.5]) # 각 batch 중 첫 번째 이미지 선택
            gt_img = denormalize(gt_image[0].cpu(), mean=[0.5]*3, std=[0.5]*3)
            output_img = denormalize(gen_output[0].detach().cpu(), mean=[0.5]*3, std=[0.5]*3) #[0.5]*3은 [0.5, 0.5, 0.5]랑 같음

            input_img = input_img.squeeze()
            input_img = torch.clamp(input_img, 0, 1)
            gt_img = torch.clamp(gt_img.permute(1, 2, 0), 0, 1) # PLT에서 쓰기 위해 (channel, height, width)를 (height, width, channel)로 바꿔줌
            output_img = torch.clamp(output_img.permute(1, 2, 0), 0, 1)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(input_img, cmap='gray')
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(gt_img)
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(output_img)
            plt.title('Generated Image')
            plt.axis('off')

            plt.show()
    # checkpoint 저장 (3 epoch당)
    if epoch % 3 == 0:
      checkpoint = {
          'epoch': epoch,
          'generator_state_dict': generator.state_dict(),
          'discriminator_state_dict': discriminator.state_dict(),
          'optimizer_G_state_dict': optimizer_G.state_dict(),
          'optimizer_D_state_dict': optimizer_D.state_dict(),
      }
      checkpoint_save_path = f'/content/drive/MyDrive/MLProject/saved/GAN/checkpoints/checkpoint_epoch{epoch+1}.pth'
      torch.save(checkpoint, checkpoint_save_path)

    epoch_loss_G = running_loss_G / len(train_loader)
    epoch_loss_D = running_loss_D / len(train_loader)
    print(f"Epoch [{epoch+1}/{CFG['EPOCHS']}], Generator Loss: {epoch_loss_G:.4f}, Discriminator Loss: {epoch_loss_D:.4f}")

    # TensorBoard에 에포크별 손실 기록
    writer.add_scalar('Epoch Loss/Generator', epoch_loss_G, epoch)
    writer.add_scalar('Epoch Loss/Discriminator', epoch_loss_D, epoch)

    # 검증 및 SSIM 계산
    avg_ssim = validate(generator, val_loader)
    print(f"Validation SSIM: {avg_ssim:.4f}")

# 학습 종료 후 SummaryWriter 닫기
writer.close()

## Inference
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import zipfile
import pandas as pd

# 디바이스 설정 (GPU 사용 가능하면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 체크포인트 불러오기
checkpoint = torch.load('/content/drive/MyDrive/MLProject/saved/GAN/checkpoints/checkpoint_epoch30.pth')

#모델 상태 불러오기
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

# 테스트 데이터프레임 로드
test_df = pd.read_csv('/content/drive/MyDrive/MLProject/test.csv')

# 테스트 이미지 경로와 결과 저장 경로 설정
base_path = '/content/drive/MyDrive/MLProject/MAT'  # 이미지 파일이 저장된 기본 경로
output_dir = '/content/output_images'
os.makedirs(output_dir, exist_ok=True)

# 입력 이미지에 대한 변환 정의 (학습 시 사용한 것과 동일하게)
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 출력 이미지를 저장하기 위한 함수
def tensor_to_pil(tensor):
    # Denormalize
    tensor = tensor.squeeze(0).cpu().clone()
    tensor = tensor * 0.5 + 0.5  # [-1,1] -> [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    # Tensor에서 PIL 이미지로 변환
    #tensor = tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    print(tensor.shape)
    image = transforms.ToPILImage()(tensor)
    return image

# Inference
generator.eval()
i=0
with torch.no_grad():
    for idx in tqdm(range(len(test_df)), desc='Inference'):
        # 테스트 데이터프레임에서 이미지 경로 가져오기
        input_image_rel_path = test_df.loc[idx, 'input_image_path'].lstrip('./')
        input_image_path = os.path.join(base_path, input_image_rel_path)
        filename = os.path.basename(input_image_path)

        # 입력 이미지 로드 및 전처리
        input_image = Image.open(input_image_path).convert('L')
        # 이미지 크기를 512x512로 조정 (필요한 경우)
        # 변환 적용
        input_tensor = input_transform(input_image).unsqueeze(0).to(device)

        # Generator를 통해 출력 이미지 생성
        gen_output = generator(input_tensor)
        # 출력 텐서를 PIL 이미지로 변환
        output_image = tensor_to_pil(gen_output)
        # 결과 이미지 저장
        output_path = os.path.join(output_dir, filename)
        output_image.save(output_path)
        if i < 10:
          plt.figure(figsize=(12, 4))
          plt.subplot(1, 2, 1)
          plt.imshow(input_image, cmap='gray')
          plt.title('Input Image')
          plt.axis('off')

          plt.subplot(1, 2, 2)
          plt.imshow(output_image)
          plt.title('Generated Image')
          plt.axis('off')

          plt.show()
          i = i + 1
# 결과 이미지들을 ZIP 파일로 압축 (제출용)
zip_filename = '/content/drive/MyDrive/MLProject/saved/GAN/GAN_30epoch_output.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.png'):
            file_path = os.path.join(output_dir, filename)
            zipf.write(file_path, arcname=filename)

# TensorBoard 로드 및 실행
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/MLProject/saved/GAN/logs

torch.save(checkpoint, '/content/drive/MyDrive/MLProject/saved/GAN/checkpoints/checkpoint_epoch30.pth')
```

### Training Review

![image.png](./images/image.png)

![image.png](./images/image%201.png)

![image.png](./images/image%202.png)

![image.png](./images/image%203.png)

Training 하면서 보니, 색칠은 그럭저럭 잘 되는 것 같은데 mask recovering이 잘 되지 않는 것으로 보였다. Tensorboard를 보니 학습은 매우 불안정했지만 Generator loss는 계속 감소했기 때문에, 이미지 증강을 통한 더 많은 데이터, 더 많은 반복 학습을 통하면 더 개선될 수 있지 않았을까 생각한다.

# Second approach : MAT(MIM) + GAN(coloring)

Papers With Code에서, 채우기만 전문으로 학습된 모델이 없을까 찾아보았는데, 

![[https://paperswithcode.com/task/image-inpainting#datasets](https://paperswithcode.com/task/image-inpainting#datasets)](./images/image%204.png)

[https://paperswithcode.com/task/image-inpainting#datasets](https://paperswithcode.com/task/image-inpainting#datasets)

그중 공개되어 있으며 모델 크기가 학습에 적당하고 성능도 CelebA-HQ Dataset에서 1등으로 괜찮다고 생각되고, 무엇보다 학습 데이터가 512*512로 이번 데이터셋과 동일한,  MAT: Mask-Aware Transformer for Large Hole Image Inpainting 모델을 사용하기로 함. 또한 이번에는 흑백 사진에서 mask된 영역만 복구한 후에, 이미 학습한 GAN으로 coloring하는 방법을 시도함.

### MAT: Mask-Aware Transformer for Large Hole Image Inpainting

![스크린샷 2024-12-18 17.46.59.png](./images/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-18_17.46.59.png)

![스크린샷 2024-12-18 17.41.18.png](./images/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-18_17.41.18.png)

기존 접근법(CNN 기반 모델, 트랜스포머 기반 모델)은 한계를 가진다고 함.

1. **CNN 기반 모델**: 근거리 정보에 집중하며, 복잡한 구조를 복원하는 데 한계가 있음.
2. **트랜스포머 기반 모델**: 계산 비용 문제로 인해 저해상도에서만 작동, 세부 정보 복원이 부족함.

위 논문에서 위와 같은 새로운 모델 구조와, Transfomer Block(TB)를 제안하는데, 차이점은 다음과 같음.

1. **Layer Normalization 제거** : large scale mark에서 gradient exploding 문제와, 정규화 과정에서 쓸모없는 토큰들임에도 확대(정규화니까)하는 문제가 있었음
    
    *we observe unstable optimization using the general block when handling large-scale
    masks, sometimes incurring gradient exploding. We attribute this training issue to the large ratio of invalid tokens (their values are nearly zero). In this circumstance, layer normalization may magnify useless tokens overwhelmingly, leading to unstable training(원문)*
    
2. **Residual connection → Feature concatenation 변경**
    
    초기에 masked 부분이 많은 상태에서 base 없이 residual connection을 써봤자 의미가 없다는 것
    
    *residual learning generally encourages the model to learn high-frequency contents. However, considering most tokens are invalid at the beginning, it is difficult to directly learn high-frequency details without proper low-frequency basis in GAN training, which makes the optimization harder(원문)*
    
    - 또한 transformer block 내에서만 사용했던 residual connection을, Conv layer와 연결되는 Global Residual connection을 사용했다고 함.
    
3.  **Style Manipulation Module :** CNN weight에 ****noise 주입으로 다양한 표현을 생성할 수 있게 함
    
    
4. **Multi-Head-Attention → Multi-Head Contextual Attention** 
- Multi-Head Attention (기존 transformer):
    - 전체 입력 토큰에 대해 Attention 계산
- **Multi-Head Contextual Attention (MCA):**
    
    ![image.png](./images/image%205.png)
    

- 입력 토큰을 고정된 크기의 window로 나눈 뒤, 각 window 내에서만 attention 계산 수행, attention 수행 후 window 이동시키기 (w*w size에서  $(⌊
w/
2
⌋, ⌊
w/
2
⌋) pixel$ 이동한다고 함)
- 이는 계산 효율성을 높이고 (일부에 대해서만 계산하면 되기 떄문에), 지역적(local) 및 장거리(global) 관계를 점진적으로 모델링할 수 있게 함.

- **Mask Updating Strategy :  valid token끼리만 attention 계산하기 위함**
    - *default attention strategy not only fails to borrow visible
    information to inpaint the holes, but also undermines the effective valid pixels* (원문)
    - 기존 attention 계산을 사용하면 Missing area가 많을수록 missing area의 정보도 계산에 사용하기 때문에 좋지 않다고 함

![image.png](./images/image%206.png)

$τ$는 매우 큰 자연수. Foward step에서 Mask는 업데이트되는데, 규칙은 다음과 같음.

- window 내에 적어도 하나의 valid token이 존재하면, window 내 모든 token이 valid로 업데이트
- window 내 모든 토큰이 invalid인 경우, 그대로 invalid로 남음.

Transformer block 차이점 정리

| **특징** | **기존 Transformer Block** | **Adjusted Transformer Block** |
| --- | --- | --- |
| **Layer Normalization** | 사용 | 제거 (학습 안정성 확보) |
| **Residual Learning** | **Residual connection** | Feature Concatenation으로 변경 (저주파수 기반 학습) |
| **Positional Embedding** | 사용 | 생략 (3×3  ConV layer로 대체, 경험적으로 위치정보를 제공하는데 충분했다고 함) |
| **Attention Method** | Multi-Head-Attention | Multi-Head Contextual Attention  |

---

## MAT + GAN Training

### MAT을 이용한 Inpainting (grayscale → grayscale)

- mask된 흑백 이미지에서 손실 영역만 복구하는 과정.
- 논문에서 공개된, https://github.com/fenglinglwb/mat([https://github.com/fenglinglwb/mat](https://github.com/fenglinglwb/mat)) 모델 사용

1. **TRAIN_input에 대한 mask 생성 → mask를 생성하는 모델 만들기**
- MAT 모델은 검은색으로 마스킹된 부분에 대해서만 inpainting작업을 수행한다.  따라서 train_input에서 어떤 부분이 mask되었는지 판별하여, 그 부분에 대해서만 검은색으로 색칠된  mask image가 필요

![image.png](./images/image%207.png)

- Train 이미지의 masking 영역은 완전한 검은색 또는 회색 등 일정한 색이 아니다.
    
    → YOLO 등 기존 Image segementaion model 사용 불가, 비슷한 색이면 mask로 인식해버린다
    
     
    
- 어떻게 train_input에서 masked area를 파악할 것인가? → train_input과 train_gt(흑백으로 변환)를 비교해서, 다른 부분을 $y_{target}$으로 설정. 이를 기반으로 image segmentation 모델 만들기
- train_input이  train_X, mask(단순 픽셀비교를 통해 다른 부분이 검은색으로 칠해진 이미지)가 정답값이 되는 것

- Image Sementaion 모델은 U-net 사용, target_y는 Dataset class에서 직접 생성
- Loss는`criterion = nn.BCEWithLogitsLoss()` 를 사용했는데, U-net의 마지막 출력층에 sigmoid가 적용되지 않아서 출력이 0~1의 확률값이 아니어도 BCELoss를 계산할 수 있게 해준다.
- 또한 이 Image Segmentation Model은 Pixel 단위 이진 분류(mask인지 아닌지)를 수행하기 때문에, BCE Loss를 사용하는 것이 적절하다.

```python
# Dataset
class SegmentationDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transforms=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transforms = transforms

        # 이미지 파일 목록 가져오기
        self.input_images = sorted(glob(os.path.join(self.input_dir, '*')))
        self.gt_images = sorted(glob(os.path.join(self.gt_dir, '*')))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # 이미지 경로
        input_image_path = self.input_images[idx]
        gt_image_path = self.gt_images[idx]

        # 이미지 로드 및 흑백으로 변환
        input_image = Image.open(input_image_path).convert('L')
        gt_image = Image.open(gt_image_path).convert('L')  # 컬러 이미지를 흑백으로 변환

        # numpy 배열로 변환
        input_array = np.array(input_image)
        gt_array = np.array(gt_image)

        # 세그멘테이션 마스크 생성
        # 마스킹된 영역: 0, 나머지 영역: 1 (train_input과 train_gt가 같으면 검은색(0) 다르면 흰색(1)
        mask = (input_array == gt_array).astype(np.float32)

        # 변환 적용
        if self.transforms:
            input_image = self.transforms['input'](input_image)
            mask = self.transforms['mask'](mask)
				
				#input_image가 train_X, mask가 target_y(정답값)이 되는 것
        return input_image, mask|
```

## Entire Training Code

```python
from google.colab import drive
drive.mount('/content/drive')

#!unzip -qq /content/drive/MyDrive/MLProject/train_dataset.zip -d /content/dataset/
!mkdir -p /content/dataset/
!unzip -qq /content/drive/MyDrive/MLProject/open.zip -d /content/dataset/train_dataset/

import os
print(len(os.listdir('/content/drive/MyDrive/MLProject/MAT/masks')))
print(len(os.listdir('/content/dataset/train_dataset/train_input')))

## MASK 생성

import os
import numpy as np
from PIL import Image
from glob import glob
from tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Colab에서 TensorBoard를 사용하기 위한 설정
%load_ext tensorboard

# Pre-trained 되지 않은 U-net
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1): # 흑백 이미지에서 흑백 마스크만 생성하는 것이므로 입출력 채널은 1 
        super(UNet, self).__init__()
        # Convolution-BatchNormalization-ReLU
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        # 디코더
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        # 출력 레이어
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 인코더
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        bottleneck = self.bottleneck(pool4)

        # 디코더
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(up4)

        up3 = self.upconv3(dec4)
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(up3)

        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(up2)

        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(up1)

        # 출력
        out = self.conv_last(dec1)
        return out

# 7. 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화
model = UNet(in_channels=1, out_channels=1)
model = model.to(device)

# 데이터 경로 설정
train_x_path = '/content/dataset/train_dataset/train_input'
label_y_path = '/content/dataset/train_dataset/train_gt'

# Dataset
class SegmentationDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transforms=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transforms = transforms

        # 이미지 파일 목록 가져오기
        self.input_images = sorted(glob(os.path.join(self.input_dir, '*')))
        self.gt_images = sorted(glob(os.path.join(self.gt_dir, '*')))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # 이미지 경로
        input_image_path = self.input_images[idx]
        gt_image_path = self.gt_images[idx]

        # 이미지 로드 및 흑백으로 변환
        input_image = Image.open(input_image_path).convert('L')
        gt_image = Image.open(gt_image_path).convert('L')  # 컬러 이미지를 흑백으로 변환

        # numpy 배열로 변환
        input_array = np.array(input_image)
        gt_array = np.array(gt_image)

        # 세그멘테이션 마스크 생성
        # 마스킹된 영역: 0, 나머지 영역: 1 (train_input과 train_gt가 같으면 검은색(0) 다르면 흰색(1)
        mask = (input_array == gt_array).astype(np.float32)

        # 변환 적용
        if self.transforms:
            input_image = self.transforms['input'](input_image)
            mask = self.transforms['mask'](mask)

        return input_image, mask

# 데이터 변환 설정
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

mask_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x).unsqueeze(0))
])

# 데이터셋 및 데이터로더 생성
# 전체 데이터셋 로드
full_dataset = SegmentationDataset(
    input_dir=train_x_path,
    gt_dir=label_y_path,
    transforms={'input': input_transform, 'mask': mask_transform}
)

# 데이터셋을 8:2 비율로 분할
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# 데이터로더 생성
batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

import gc
gc.collect()
torch.cuda.empty_cache()

# TensorBoard 로그를 저장할 디렉토리 설정
log_dir = '/content/drive/MyDrive/MLProject/MAT/tensorboard_logs'

# SummaryWriter 생성
writer = SummaryWriter(log_dir=log_dir)

# 손실 함수 및 옵티마이저 설정
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

    for batch_idx, (input_image, mask) in enumerate(train_loader_tqdm):
        input_image = input_image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        outputs = model(input_image)

        loss = criterion(outputs, mask)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

    # TensorBoard에 기록
    writer.add_scalar('Loss/train', epoch_loss, epoch)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_image, mask in tqdm(val_loader):
            input_image = input_image.to(device)
            mask = mask.to(device)

            outputs = model(input_image)
            loss = criterion(outputs, mask)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    # TensorBoard에 검증 손실 기록
    writer.add_scalar('Loss/val', val_loss, epoch)

    # 모델 저장 (선택 사항)
    torch.save(model.state_dict(), f'/content/drive/MyDrive/MLProject/MAT/maskGen/unet_epoch_{epoch+1}.pth')

# TensorBoard Writer 닫기
writer.close()

import os
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('/content/drive/MyDrive/MLProject/MAT/maskGen/unet_epoch_3.pth'))
model = model.to(device)

# Train_image에 대해 mask를 생성
def inference(model, image_path, device):
    model.eval()
    with torch.no_grad():
        input_image = Image.open(image_path).convert('L') # 이미 흑백이므로 안 해도 됨

        input_tensor = transforms.ToTensor()(input_image)
        input_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(input_tensor)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

        # 모델 출력값이 0.5 이상이면 흰색으로 칠하기 위해
        threshold = 0.5
        binary_mask = (output > threshold).astype(np.uint8)

        # 마스크 이미지 생성
        mask_image = Image.fromarray(binary_mask * 255)

        return mask_image

from tqdm import tqdm
# 마스크 이미지를 저장할 디렉토리 설정
#output_dir = '/content/drive/MyDrive/MLProject/MAT/masks'
output_dir = '/content/drive/MyDrive/MLProject/tast_output_masks'
# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 입력 이미지 디렉토리 설정
#input_dir = '/content/dataset/train_dataset/train_input'
input_dir = '/content/drive/MyDrive/MLProject/test_input'
i=0
# 추론 및 결과 시각화 및 저장
for img_name in tqdm(os.listdir(input_dir)):
    #print(f"Processing: {img_name}")
    test_image_path = os.path.join(input_dir, img_name)
    predicted_mask = inference(model, test_image_path, device)

    # 마스크 이미지 저장 경로 설정
    mask_save_path = os.path.join(output_dir, f"mask_{img_name}")

    # 마스크 이미지 저장
    predicted_mask.save(mask_save_path)

    # 원본 이미지 로드
    original_image = Image.open(test_image_path).convert('L')

    # 결과 시각화
    if i < 5:
      plt.figure(figsize=(10, 5))

      plt.subplot(1, 2, 1)
      plt.imshow(original_image, cmap='gray')
      plt.title('Input Image')
      plt.axis('off')

      plt.subplot(1, 2, 2)
      plt.imshow(predicted_mask, cmap='gray')
      plt.title('Predicted Mask')
      plt.axis('off')

      plt.show()
      i = i + 1
```

![image.png](./images/image%208.png)

3번의 epoch로 꽤 정확히 mask 영역이 검은색으로 칠해진 mask 이미지를 생성할 수 있었다. 하지만 완벽히 정확하지는 않다.

![image.png](./images/image%209.png)

1. **Train_input 전체에 대해 mask 생성**

![image.png](./images/image%2010.png)

위 모델을 통해 모든 이미지에 대해 mask를 생성하였다.

1. **MAT를 이용하여 Inpainting 수행**

```python
!git  clone https://github.com/fenglinglwb/MAT.git /content/drive/MyDrive/MLProject/MAT
!pip install -r /content/drive/MyDrive/MLProject/MAT/requirements.txt
!python /content/drive/MyDrive/MLProject/MAT/generate_image.py --network /content/drive/MyDrive/MLProject/MAT/CelebA-HQ_512.pkl --dpath /content/dataset/train_dataset/train_input --outdir /content/drive/MyDrive/MLProject/MAT/sample_ouput --mpath /content/drive/MyDrive/MLProject/MAT/masks
```

논문에서 공개한 모델을 이용하여 Inpainting을 수행함. 해당 github 링크에서 자세한 사용법이 나와있음.

![스크린샷 2024-12-18 17.34.44.png](./images/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-18_17.34.44.png)

1. **이전에 학습한 GAN을 이용하여 coloring**

```python
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import zipfile
import pandas as pd

# 디바이스 설정 (GPU 사용 가능하면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 체크포인트 불러오기
checkpoint = torch.load('/content/drive/MyDrive/MLProject/saved/GAN/checkpoints/checkpoint_epoch30.pth')

#모델 상태 불러오기
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

# 테스트 데이터프레임 로드
test_df = pd.read_csv('/content/drive/MyDrive/MLProject/test.csv')

# 테스트 이미지 경로와 결과 저장 경로 설정
base_path = '/content/drive/MyDrive/MLProject/MAT'  # 이미지 파일이 저장된 기본 경로
output_dir = '/content/output_images'
os.makedirs(output_dir, exist_ok=True)

# 입력 이미지에 대한 변환 정의 (학습 시 사용한 것과 동일하게)
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 출력 이미지를 저장하기 위한 함수
def tensor_to_pil(tensor):
    # Denormalize
    tensor = tensor.squeeze(0).cpu().clone()
    tensor = tensor * 0.5 + 0.5  # [-1,1] -> [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    # Tensor에서 PIL 이미지로 변환
    #tensor = tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    print(tensor.shape)
    image = transforms.ToPILImage()(tensor)
    return image

# Inference
generator.eval()
i=0
with torch.no_grad():
    for idx in tqdm(range(len(test_df)), desc='Inference'):
        # 테스트 데이터프레임에서 이미지 경로 가져오기
        input_image_rel_path = test_df.loc[idx, 'input_image_path'].lstrip('./')
        input_image_path = os.path.join(base_path, input_image_rel_path)
        filename = os.path.basename(input_image_path)

        # 입력 이미지 로드 및 전처리
        input_image = Image.open(input_image_path).convert('L')
        # 이미지 크기를 512x512로 조정 (필요한 경우)
        # 변환 적용
        input_tensor = input_transform(input_image).unsqueeze(0).to(device)

        # Generator를 통해 출력 이미지 생성
        gen_output = generator(input_tensor)
        # 출력 텐서를 PIL 이미지로 변환
        output_image = tensor_to_pil(gen_output)
        # 결과 이미지 저장
        output_path = os.path.join(output_dir, filename)
        output_image.save(output_path)
        if i < 10:
          plt.figure(figsize=(12, 4))
          plt.subplot(1, 2, 1)
          plt.imshow(input_image, cmap='gray')
          plt.title('Input Image')
          plt.axis('off')

          plt.subplot(1, 2, 2)
          plt.imshow(output_image)
          plt.title('Generated Image')
          plt.axis('off')

          plt.show()
          i = i + 1
# 결과 이미지들을 ZIP 파일로 압축 (제출용)
zip_filename = '/content/drive/MyDrive/MLProject/saved/GAN/MAT_GAN.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.png'):
            file_path = os.path.join(output_dir, filename)
            zipf.write(file_path, arcname=filename)

```

![image.png](./images/image%2011.png)

![image.png](./images/image%2012.png)

눈으로 보기에는 mask 영역에 대해서는 어느 부분이 inpainting되었는지 찾아볼 수 없는 정도였지만, coloring도 완벽하지는 못한 모습을 보임.

# Review

![image.png](./images/image%2013.png)

- 아쉽게도 기존 GAN과 별 차이없는 성적을 보임. MAT 결과물을 기반으로 Coloring하는 GAN을 따로 학습했으면 더 좋았을 것임.
- Competition 종료 후, 1등 참가자가 자신의 코드를 공유했는데, 공교롭게도 **Inpainting에 동일한 모델(MAT)을** 사용해서 매우 놀라웠음. Coloring을 더 잘 했더라면 좋은 성적이 나왔을 것 같음.

![스크린샷 2024-12-18 17.35.24.png](./images/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-12-18_17.35.24.png)

위 사진은 대회 1등의 결과물이다. ([https://dacon.io/competitions/official/236420/codeshare/12131?page=1&dtype=recent](https://dacon.io/competitions/official/236420/codeshare/12131?page=1&dtype=recent))

- 최종 결과

![image.png](./images/image%2014.png)

## 프로젝트를 진행하며 느낀 점

- 시간, 비용적 문제로 Image Augmentation 등 더 많은 데이터를 학습에 사용하지 못한 점이 아쉽다.
- Coloring 전문 모델을 더 찾아봤으면 좋았을 것이다.
- 생각보다 아주 근소한 점수 차이로 순위가 갈리는 것이 신기했다.
- Mask 생성 모델을 직접 구현하였는데, openCV 기반 모델 등 더 나은 모델을 찾아봤으면 더 좋은 결과가 있었을 것같다.
- 수업 시간에 배운 지식들(CNN, Tranformer, Normalization 등)을 기반으로 논문을 찾아보며 단순 Pre-trained model 사용에 그치는 것이 아닌 논문에 대한 대략적인 이해(수학적으로는 아니지만) 할 수 있었다.
- 프로젝트를 진행하면서 Pytorch, colab 등에 대한 사용법을 익힐 수 있었다.