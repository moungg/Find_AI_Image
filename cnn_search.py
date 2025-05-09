import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의 및 가중치 불러오기
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 클래스 수 2 (실제, AI)
model.load_state_dict(torch.load("ai_detection_model.pt", map_location=device))
model = model.to(device)
model.eval()

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 탐지 함수 정의
def predict_image(path):
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1)
    return "ai 이미지" if pred.item() == 0 else "실제 이미지"



# 테스트 예시
if __name__ == "__main__":
    print(predict_image("val/fake_1.png"))  # 파일 경로에 맞게 수정

