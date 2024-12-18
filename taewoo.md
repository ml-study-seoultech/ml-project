# 기계학습 팀 프로젝트

---

## 문제 상황 설명
손상된 이미지를 복구하고 흑백 이미지를 컬러로 변환하는 Vision AI 알고리즘을 개발하는 것을 목표로 합니다. 
이미지 복원 기술은 역사적 사진 복원, 영상 편집, 의료 영상 분석 등 다양한 분야에서 활용됩니다. 이번 대회에서는 손실된 이미지의 특정 영역을 복구하고 흑백 이미지를 원본과 유사한 컬러 이미지로 변환하는 알고리즘을 만들어야 합니다.

---

## 문제 해결 전략

### 1. 마스킹 부분 감지
- 이미지에서 손상된 부분(마스킹된 영역)을 감지하는 모델을 개발합니다.

### 2. 복원 (Inpainting)
- 마스킹된 부분을 주변 이미지와 자연스럽게 연결되도록 복원합니다.

### 3. 컬러화 (Colorization)
- 복원된 흑백 이미지를 컬러 이미지로 변환하여 원본에 가까운 이미지를 생성합니다.

---

## 첫 번째 시도: GAN(생성적 적대 신경망)을 활용한 접근

### 코드 설명
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

# Hyperparameters
batch_size = 4  # 큰 이미지를 처리하므로 배치 크기를 줄였습니다.
lr = 0.0002
num_epochs = 5
img_size = 512  # 이미지 크기를 512로 설정
channels = 1  # 흑백 이미지일 경우 1, 컬러 이미지일 경우 3
img_shape = (channels, img_size, img_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),  # 이미지 크기를 512x512로 조정
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(
            os.path.join(self.input_dir, self.input_images[idx])
        ).convert("L")
        gt_image = Image.open(os.path.join(self.gt_dir, self.gt_images[idx])).convert(
            "L"
        )

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image


# Directories
input_dir = "./test_train_input"
gt_dir = "./test_train_gt"

# Data loading
custom_dataset = CustomDataset(input_dir, gt_dir, transform=transform)
custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def down_block(
            in_channels, out_channels, kernel_size, stride, padding, normalize=True
        ):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def up_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            return layers

        self.encoder = nn.Sequential(
            *down_block(
                channels, 64, 4, 2, 1, normalize=False
            ),  # [batch, 64, 256, 256]
            *down_block(64, 128, 4, 2, 1),  # [batch, 128, 128, 128]
            *down_block(128, 256, 4, 2, 1),  # [batch, 256, 64, 64]
            *down_block(256, 512, 4, 2, 1),  # [batch, 512, 32, 32]
            *down_block(512, 512, 4, 2, 1),  # [batch, 512, 16, 16]
            *down_block(512, 512, 4, 2, 1),  # [batch, 512, 8, 8]
            *down_block(512, 512, 4, 2, 1),  # [batch, 512, 4, 4]
            *down_block(512, 512, 4, 2, 1),  # [batch, 512, 2, 2]
            *down_block(512, 512, 4, 2, 1),  # [batch, 512, 1, 1]
        )

        self.decoder = nn.Sequential(
            *up_block(512, 512, 4, 2, 1),  # [batch, 512, 2, 2]
            *up_block(512, 512, 4, 2, 1),  # [batch, 512, 4, 4]
            *up_block(512, 512, 4, 2, 1),  # [batch, 512, 8, 8]
            *up_block(512, 512, 4, 2, 1),  # [batch, 512, 16, 16]
            *up_block(512, 512, 4, 2, 1),  # [batch, 512, 32, 32]
            *up_block(512, 256, 4, 2, 1),  # [batch, 256, 64, 64]
            *up_block(256, 128, 4, 2, 1),  # [batch, 128, 128, 128]
            *up_block(128, 64, 4, 2, 1),  # [batch, 64, 256, 256]
            nn.ConvTranspose2d(64, channels, 4, 2, 1),  # [batch, channels, 512, 512]
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(
            in_channels, out_channels, kernel_size, stride, padding, normalize=True
        ):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(channels, 64, 4, 2, 1, normalize=False),  # [batch, 64, 256, 256]
            *block(64, 128, 4, 2, 1),  # [batch, 128, 128, 128]
            *block(128, 256, 4, 2, 1),  # [batch, 256, 64, 64]
            *block(256, 512, 4, 2, 1),  # [batch, 512, 32, 32]
            *block(512, 512, 4, 2, 1),  # [batch, 512, 16, 16]
            *block(512, 512, 4, 2, 1),  # [batch, 512, 8, 8]
            *block(512, 512, 4, 2, 1),  # [batch, 512, 4, 4]
            nn.Conv2d(512, 1, 4, 1, 0),  # [batch, 1, 1, 1]
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


# 모델 초기화
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (input_imgs, gt_imgs) in enumerate(custom_dataloader):

        # Adversarial ground truths
        valid = torch.ones(input_imgs.size(0), 1, requires_grad=False).to(device)
        fake = torch.zeros(input_imgs.size(0), 1, requires_grad=False).to(device)

        # Configure input
        real_imgs = gt_imgs.to(device)
        input_imgs = input_imgs.to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(input_imgs)

        # Loss measures generator's ability to fool the discriminator
        pred_fake = discriminator(gen_imgs)
        g_loss = criterion(pred_fake, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real images
        pred_real = discriminator(real_imgs)
        real_loss = criterion(pred_real, valid)

        # Fake images
        pred_fake = discriminator(gen_imgs.detach())
        fake_loss = criterion(pred_fake, fake)

        # Total loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(custom_dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
        )
```

GAN(생성적 적대 신경망)을 사용하여 이미지 복원 문제를 해결하려 했습니다. Generator는 손상된 이미지를 입력받아 복원 및 컬러화를 수행하며, Discriminator는 복원된 이미지를 평가하여 학습을 돕습니다.

1. **Generator**:
   - U-Net 아키텍처를 기반으로 흑백 이미지를 복원.
   - Encoder-Decoder 구조를 통해 고해상도 특징을 복원합니다.

2. **Discriminator**:
   - PatchGAN 구조로, 복원된 이미지의 현실성을 평가합니다.
   - 손실 함수로 BCE(Binary Cross-Entropy)를 사용하여 진짜/가짜 이미지를 구분합니다.

3. **손실 함수**:
   - Generator는 Discriminator를 속이는 방향으로 학습하며, 손실 함수로 BCE를 사용했습니다.

4. **결과 저장**:
   - Generator를 통해 생성된 이미지를 `[0, 1]` 범위로 변환한 후 저장하였습니다.

### TEST
```python
import torch
from torchvision import transforms
from PIL import Image
import os

# Hyperparameters
img_size = 512  # 이미지 크기
channels = 1  # 흑백 이미지
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform 정의 (학습 시 사용한 것과 동일하게 설정)
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# 테스트 데이터 디렉토리
test_input_dir = "./test_test_input"
output_dir = "./test_output_images"
os.makedirs(output_dir, exist_ok=True)

# 모델 로드 (학습된 가중치를 로드했다고 가정)
generator = Generator().to(device)
generator.eval()  # 평가 모드로 전환

# 테스트 함수 정의
def generate_images(input_dir, output_dir):
    # 테스트 디렉토리의 모든 이미지 파일 로드
    test_images = sorted(os.listdir(input_dir))

    for idx, img_file in enumerate(test_images):
        input_path = os.path.join(input_dir, img_file)

        # 입력 이미지 로드 및 전처리
        input_image = Image.open(input_path).convert("L")
        input_tensor = transform(input_image).unsqueeze(0).to(device)  # 배치 차원 추가

        # 이미지 생성
        with torch.no_grad():
            gen_tensor = generator(input_tensor)

        # 생성된 이미지를 저장 ([-1, 1] -> [0, 1]로 변환)
        gen_tensor = (gen_tensor * 0.5 + 0.5).clamp(0, 1)  # 정규화 해제
        gen_image = gen_tensor.squeeze(0).squeeze(0).cpu().numpy()  # 배치, 채널 제거
        gen_image = (gen_image * 255).astype("uint8")  # [0, 1] -> [0, 255]

        # PIL 이미지로 변환하여 저장
        output_path = os.path.join(output_dir, f"generated_{idx+1}.png")
        Image.fromarray(gen_image).save(output_path)
        print(f"Generated image saved to {output_path}")

# 테스트 실행
generate_images(test_input_dir, output_dir)
```

### 결과 및 아쉬운 점

<div style="display: flex; justify-content: space-around; align-items: center;">
<img src="./result_code1_1.png.png" alt="이미지 1" style="width: 100%; height: auto;">
</div>

- **결과**: GAN을 활용해 손상된 이미지를 복구하고 컬러화를 수행했으나, 특정 영역에서 자연스러운 복원이 이루어지지 않았습니다.
- **아쉬운 점**:
  - 마스킹된 영역이 복원된 이미지는 일부 픽셀에서 결함이 발견되었습니다.
  - 컬러화 부분에서도 색상 표현이 불완전했습니다.
- **개선 방향**:
  - 더 큰 데이터셋으로 학습하여 모델의 일반화 성능을 높여야 합니다.
  - 마스킹 처리 부분을 더욱 세밀하게 다룰 필요가 있습니다.

---

## 두 번째 시도: U-Net 기반 FullPipeline 활용

### 코드 설명
```python
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
import segmentation_models_pytorch as smp

# 1. 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, csv_file, input_dir, gt_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.data.iloc[idx]['input_image_path'])
        gt_path = os.path.join(self.gt_dir, self.data.iloc[idx]['gt_image_path'])

        # 이미지 로드
        input_image = Image.open(input_path).convert('L')  # 흑백
        gt_image = Image.open(gt_path).convert('RGB')  # 컬러

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image

# 2. 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1] 정규화
])

# 3. 데이터셋 및 데이터로더 생성
train_dataset = ImageDataset(
    csv_file='./train.csv',
    input_dir='',
    gt_dir='',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 4. 모델 정의
class FullPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights="imagenet",     
            in_channels=1,                  
            classes=1                       
        )
        self.inpaint_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=2,  # 이미지 + 마스크
            classes=1
        )
        self.colorize_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=3
        )

    def forward(self, x):
        # Step 1: Mask Detection
        mask = self.mask_model(x)
        
        # Step 2: Inpainting
        input_with_mask = torch.cat([x, mask], dim=1)  # 채널 합치기
        inpainted = self.inpaint_model(input_with_mask)
        
        # Step 3: Colorization
        colorized = self.colorize_model(inpainted)
        
        return mask, inpainted, colorized

# 5. 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullPipeline().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()

# 6. 학습 루프
from tqdm import tqdm

for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # 모델 예측
        mask, inpainted, colorized = model(inputs)

        # 손실 계산
        loss = criterion(inpainted, inputs) + criterion(colorized, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
```

U-Net 기반 **FullPipeline 모델**을 정의하여 마스킹, 복원, 컬러화를 단계적으로 처리했습니다.

1. **Mask Model**:
   - 입력 이미지에서 마스킹된 영역을 감지하는 역할.
   - `segmentation_models_pytorch` 라이브러리의 U-Net을 활용하여 손실된 영역을 정확히 감지하도록 학습.

2. **Inpaint Model**:
   - 마스킹된 영역을 주변 픽셀과 유사한 패턴으로 복원.
   - 입력 이미지와 마스크를 채널 차원에서 결합하여 복원합니다.

3. **Colorize Model**:
   - 복원된 흑백 이미지를 컬러 이미지로 변환.
   - RGB로 변환된 이미지를 출력하여 자연스러운 컬러화 수행.

4. **손실 함수**:
   - L1 Loss를 사용하여 출력 이미지와 원본 이미지의 차이를 최소화합니다.

5. **결과 저장**:
   - 각 테스트 이미지에 대해 `TEST_000`, `TEST_001` 형식으로 저장.

### TEST
```python
import os


def test_model(model, test_csv, input_dir, transform, save_dir):
    # 테스트 데이터셋 생성
    test_data = pd.read_csv(test_csv)
    input_paths = test_data["input_image_path"]

    model.eval()  # 모델 평가 모드
    os.makedirs(save_dir, exist_ok=True)  # 결과 저장 디렉토리 생성

    with torch.no_grad():
        for idx, input_path in enumerate(input_paths):
            # 입력 이미지 로드 및 전처리
            full_input_path = os.path.join(input_dir, input_path)
            input_image = Image.open(full_input_path).convert("L")  # 흑백
            input_tensor = (
                transform(input_image).unsqueeze(0).to(device)
            )  # 배치 차원 추가

            # 모델 예측
            _, _, colorized = model(input_tensor)

            # 결과를 numpy 형식으로 변환
            output_np = colorized[0].cpu().permute(1, 2, 0).numpy()  # 컬러화된 결과

            # 이미지 범위 복구 (Normalize 후 값 범위 복원)
            output_np = (
                (output_np * 255).clip(0, 255).astype("uint8")
            )  # 0-255로 스케일링

            # 파일명 설정 (TEST_000, TEST_001, ...)
            save_path = os.path.join(save_dir, f"TEST_{idx:03d}.png")
            Image.fromarray(output_np).save(save_path)

            print(f"Result saved to: {save_path}")


# 8. 테스트 실행
test_csv = "./test.csv"  # 테스트 CSV 파일 경로
input_dir = "./test_inputs"  # 테스트 입력 이미지 디렉토리
save_dir = "./sample_submission"  # 결과 저장 디렉토리

test_model(model, test_csv, input_dir, transform, save_dir)
```

### 결과 및 아쉬운 점

<div style="display: flex; justify-content: space-around; align-items: center;">
<img src="./result_code2_1.png.png" alt="이미지 1" style="width: 30%; height: auto;">
<img src="./result_code2_2.png.png" alt="이미지 2" style="width: 30%; height: auto;">
<img src="./result_code2_3.png.png" alt="이미지 3" style="width: 30%; height: auto;">
</div>

- **결과**:
  - 컬러화 성능이 이전보다 개선되었지만, 특정 영역에서 복원이 부자연스러운 부분이 확인되었습니다.
- **아쉬운 점**:
  - 데이터셋 크기의 제한으로 인해 모델이 일부 패턴에 과적합되거나, 특정 픽셀에서 복원이 잘 이루어지지 않음.
  - 복원된 흑백 이미지를 컬러화하는 과정에서 정확한 색상 매칭이 부족.
  - 기존 어두운 부분에 대해서 컬러화 과정에서 검은 색상의 비율이 많이 들어가게 되어, 흑백 조화가 이루어지지 않음.

---

## 개선 방향

### 1. 데이터셋 확장
- 다양한 유형의 이미지와 마스킹 데이터로 학습하여 모델의 일반화 성능을 향상.
- 특히, 복잡한 패턴과 색상이 포함된 이미지를 추가로 확보.

### 2. 마스킹 처리 강화
- Mask Model의 민감도를 개선하여 더욱 정확한 마스크 영역을 탐지.
- 마스킹 처리에서 False Positive/Negative 비율을 줄이기 위한 후처리 적용.

### 3. Inpaint 모델 개선
- GAN과 같은 생성 모델을 활용하여 복원된 영역이 주변과 자연스럽게 연결되도록 처리.
- 복원 과정에서 Texture Consistency를 보장하기 위한 새로운 손실 함수 도입.

### 4. 컬러화 성능 향상
- Colorize Model의 색상 표현 능력을 강화하기 위해 사전 학습된 모델을 활용.
- 원본 이미지의 색상 히스토그램을 참조하는 알고리즘 추가.

---

## 결론

첫 번째 시도는 GAN 기반 접근법으로, 전체적인 구조와 손실 함수에서 개선 여지가 있었으며, 두 번째 시도는 U-Net 기반 FullPipeline을 사용하여 단계적으로 문제를 해결했으나 복원 및 컬러화의 세부 성능에서 한계를 확인할 수 있었습니다.

향후 개선을 통해 복원 및 컬러화의 정확도를 높이고, 더 큰 데이터셋과 고성능 모델을 적용하여 실제 활용 가능한 이미지 복원 알고리즘으로 발전시킬 계획입니다.