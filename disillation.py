import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# U-Net 구성 요소
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNetWithFeatures(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, features=[64, 128]):
        super(UNetWithFeatures, self).__init__()
        self.encoder_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_features = []  # 각 encoder block의 출력을 저장하기 위한 placeholder (forward에서 저장)

        # Encoder
        prev_channels = in_channels
        for feature in features:
            self.encoder_convs.append(DoubleConv(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.decoder_upconvs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.decoder_upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_convs.append(DoubleConv(feature * 2, feature))

        # Final output conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_features = []
        # Encoder 단계
        for enc in self.encoder_convs:
            x = enc(x)
            encoder_features.append(x)
            x = self.pool(x)

        # Bottleneck
        bottleneck_feature = self.bottleneck(x)
        x = bottleneck_feature

        # Decoder 단계: encoder의 skip connection 사용
        decoder_features = []
        encoder_features_rev = encoder_features[::-1]
        for idx in range(len(self.decoder_upconvs)):
            x = self.decoder_upconvs[idx](x)
            # 만약 사이즈가 다르면 보간(interpolate)으로 맞춤
            skip_feature = encoder_features_rev[idx]
            if x.shape[2:] != skip_feature.shape[2:]:
                x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip_feature, x], dim=1)
            x = self.decoder_convs[idx](x)
            decoder_features.append(x)

        out = self.final_conv(x)
        # encoder_features, bottleneck_feature, decoder_features 모두 반환
        return out, encoder_features, bottleneck_feature, decoder_features

# -------------------------
# Hybrid Distillation Loss (이미지 출력 및 중간 feature 전체 활용)
# -------------------------
class HybridDistillationLossFull(nn.Module):
    def __init__(self, alpha=0.5, beta_enc=0.25, beta_dec=0.25, beta_bn=0.5):
        """
        alpha: 최종 출력 이미지 간 pixel-level loss의 가중치
        beta_enc: encoder feature loss 가중치 (모든 encoder 단계 평균)
        beta_dec: decoder feature loss 가중치 (모든 decoder 단계 평균)
        beta_bn: bottleneck feature loss 가중치
        """
        super(HybridDistillationLossFull, self).__init__()
        self.alpha = alpha
        self.beta_enc = beta_enc
        self.beta_dec = beta_dec
        self.beta_bn = beta_bn
        self.mse_loss = nn.MSELoss()

    def forward(self, student_output, teacher_output, 
                student_enc_feats, teacher_enc_feats,
                student_bn_feat, teacher_bn_feat,
                student_dec_feats, teacher_dec_feats, target):
        # 최종 이미지 output에 대한 pixel-level loss
        pixel_loss = self.mse_loss(student_output, teacher_output)

        # Encoder feature loss: 각 encoder 단계별 feature를 MSE로 계산한 후 평균
        enc_losses = [self.mse_loss(s_feat, t_feat) 
                      for s_feat, t_feat in zip(student_enc_feats, teacher_enc_feats)]
        enc_loss = sum(enc_losses) / len(enc_losses) if enc_losses else 0

        # Decoder feature loss: 각 decoder 단계별 feature를 MSE로 계산한 후 평균
        dec_losses = [self.mse_loss(s_feat, t_feat) 
                      for s_feat, t_feat in zip(student_dec_feats, teacher_dec_feats)]
        dec_loss = sum(dec_losses) / len(dec_losses) if dec_losses else 0

        # Bottleneck feature loss
        bn_loss = self.mse_loss(student_bn_feat, teacher_bn_feat)

        # Task loss: student 최종 출력과 ground truth SCER event 간의 MSE loss
        task_loss = self.mse_loss(student_output, target)

        total_loss = task_loss \
                     + self.alpha * pixel_loss \
                     + self.beta_enc * enc_loss \
                     + self.beta_dec * dec_loss \
                     + self.beta_bn * bn_loss

        return total_loss, pixel_loss, enc_loss, bn_loss, dec_loss, task_loss

# -------------------------
# 사용 예시
# -------------------------
if __name__ == "__main__":
    # Teacher와 Student 모델 생성 (입력은 blur image, 출력은 SCER event 이미지)
    teacher_model = UNetWithFeatures(in_channels=3, out_channels=6, features=[64, 128])
    student_model = UNetWithFeatures(in_channels=3, out_channels=6, features=[64, 128])
    
    # 예시 입력 (batch_size=4, 채널=3, H=128, W=128)
    input_tensor = torch.randn(4, 3, 128, 128)
    # Ground truth SCER event 이미지 (batch_size=4, 채널=6, H=128, W=128)
    target = torch.randn(4, 6, 128, 128)
    
    # Teacher 모델 forward pass
    teacher_output, teacher_enc_feats, teacher_bn_feat, teacher_dec_feats = teacher_model(input_tensor)
    # Student 모델 forward pass
    student_output, student_enc_feats, student_bn_feat, student_dec_feats = student_model(input_tensor)
    
    # Hybrid distillation loss 계산 (모든 단계 활용)
    loss_fn = HybridDistillationLossFull(alpha=0.5, beta_enc=0.25, beta_dec=0.25, beta_bn=0.5)
    total_loss, pixel_loss, enc_loss, bn_loss, dec_loss, task_loss = loss_fn(
        student_output, teacher_output,
        student_enc_feats, teacher_enc_feats,
        student_bn_feat, teacher_bn_feat,
        student_dec_feats, teacher_dec_feats,
        target
    )
    
    print("Total Loss:", total_loss.item())
    print("Pixel-level Distillation Loss:", pixel_loss.item())
    print("Encoder Feature Loss:", enc_loss.item())
    print("Bottleneck Feature Loss:", bn_loss.item())
    print("Decoder Feature Loss:", dec_loss.item())
    print("Task Loss:", task_loss.item())
