"""
5개 CNN 백본 모델
- ResNet18
- EfficientNet-B0
- MobileNetV3-Small
- ResNet50
- ConvNeXt-Tiny
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VideoClassifier(nn.Module):
    """비디오 분류 베이스 모델"""
    
    def __init__(self, backbone, num_classes=3, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.backbone = backbone
        self.num_classes = num_classes
        
        # 백본의 출력 차원 확인
        self.feature_dim = self._get_feature_dim()
        
        # Temporal pooling + Classifier
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def _get_feature_dim(self):
        """백본 출력 차원 자동 감지"""
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, torch.Tensor):
                return features.shape[1]
            return features
    
    def forward(self, x):
            """
            Args:
                x: (B, T, C, H, W) 비디오 텐서
            Returns:
                logits: (B, num_classes)
            """
            B, T, C, H, W = x.shape

            # (B, T, C, H, W) -> (B*T, C, H, W)
            x = x.view(B * T, C, H, W)

            # 백본 통과
            features = self.backbone(x)  # (B*T, D, H, W) or (B*T, D)

            # ✅ 4D feature map이면 Global Average Pooling
            if features.ndim == 4:
                features = torch.mean(features, dim=[2, 3])  # (B*T, D)

            # (B*T, D) -> (B, T, D)
            features = features.view(B, T, -1)

            # Temporal pooling: (B, T, D) -> (B, D)
            features = features.permute(0, 2, 1)
            features = self.temporal_pool(features).squeeze(-1)

            # Classifier
            logits = self.classifier(features)
            return logits   


def build_resnet18(num_classes=3, pretrained=True):
    """ResNet18 백본"""
    backbone = models.resnet18(pretrained=pretrained)
    # FC layer 제거
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.add_module('flatten', nn.Flatten())
    
    model = VideoClassifier(backbone, num_classes)
    return model


def build_efficientnet_b0(num_classes=3, pretrained=True):
    """EfficientNet-B0 백본"""
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
    else:
        backbone = models.efficientnet_b0(weights=None)
    
    # Classifier 제거
    backbone.classifier = nn.Identity()
    backbone.avgpool = nn.AdaptiveAvgPool2d(1)
    backbone.flatten = nn.Flatten()
    
    model = VideoClassifier(backbone, num_classes)
    return model


def build_mobilenet_v3_small(num_classes=3, pretrained=True):
    """MobileNetV3-Small 백본"""
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
    else:
        backbone = models.mobilenet_v3_small(weights=None)
    
    # Classifier 제거
    backbone.classifier = nn.Identity()
    backbone.avgpool = nn.AdaptiveAvgPool2d(1)
    backbone.flatten = nn.Flatten()
    
    model = VideoClassifier(backbone, num_classes)
    return model


def build_resnet50(num_classes=3, pretrained=True):
    """ResNet50 백본"""
    backbone = models.resnet50(pretrained=pretrained)
    # FC layer 제거
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.add_module('flatten', nn.Flatten())
    
    model = VideoClassifier(backbone, num_classes)
    return model


def build_convnext_tiny(num_classes=3, pretrained=True):
    """ConvNeXt-Tiny 백본"""
    if pretrained:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        backbone = models.convnext_tiny(weights=weights)
    else:
        backbone = models.convnext_tiny(weights=None)
    
    # Classifier 제거
    backbone.classifier = nn.Sequential(
        backbone.classifier[0],  # LayerNorm
        backbone.classifier[1],  # Flatten
    )
    
    model = VideoClassifier(backbone, num_classes)
    return model


def build_model(model_name, num_classes=3, pretrained=True):
    """
    모델 생성 통합 함수
    
    Args:
        model_name: 'resnet18', 'efficientnet_b0', 'mobilenet_v3', 'resnet50', 'convnext_tiny'
        num_classes: 클래스 수
        pretrained: 사전학습 가중치 사용
    
    Returns:
        model
    """
    
    models_dict = {
        'resnet18': build_resnet18,
        'efficientnet_b0': build_efficientnet_b0,
        'mobilenet_v3': build_mobilenet_v3_small,
        'resnet50': build_resnet50,
        'convnext_tiny': build_convnext_tiny,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")
    
    model = models_dict[model_name](num_classes, pretrained)
    
    print(f"✅ {model_name.upper()} 생성 (pretrained={pretrained})")
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   파라미터: {total_params:,} (학습 가능: {trainable_params:,})")
    
    return model


# 모델 정보
MODEL_INFO = {
    'resnet18': {
        'name': 'ResNet-18',
        'params': '11.7M',
        'description': '가볍고 빠른 베이스라인'
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'params': '5.3M',
        'description': '효율적인 아키텍처'
    },
    'mobilenet_v3': {
        'name': 'MobileNetV3-Small',
        'params': '2.5M',
        'description': '경량 모바일 모델'
    },
    'resnet50': {
        'name': 'ResNet-50',
        'params': '25.6M',
        'description': '강력한 성능'
    },
    'convnext_tiny': {
        'name': 'ConvNeXt-Tiny',
        'params': '28.6M',
        'description': '최신 아키텍처'
    }
}


if __name__ == "__main__":
    # 테스트
    print("="*80)
    print("🧪 모델 테스트")
    print("="*80)
    
    dummy_input = torch.randn(2, 16, 3, 224, 224)  # (B, T, C, H, W)
    
    for model_name in MODEL_INFO.keys():
        print(f"\n{'='*80}")
        print(f"모델: {MODEL_INFO[model_name]['name']}")
        print(f"{'='*80}")
        
        try:
            model = build_model(model_name, num_classes=3, pretrained=False)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"   입력: {dummy_input.shape}")
            print(f"   출력: {output.shape}")
            print(f"   예측: {output.argmax(dim=1).tolist()}")
            print("   ✅ 정상 작동")
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")