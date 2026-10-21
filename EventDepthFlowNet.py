import torch
import torch.nn as nn
import torch.nn.functional as F

_BASE_CHANNELS = 64


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 4, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = F.relu(attention)
        attention = self.conv3(attention)
        attention = self.sigmoid(attention)
        return x * attention

class EventDepthFlowNet(nn.Module):
    def __init__(self, args):
        super(EventDepthFlowNet, self).__init__()
        self._args = args
        self.use_rgb = getattr(args, 'use_rgb', True)


        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )


        self.encoder1 = nn.Sequential(
            nn.Conv2d(4 + 32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.attn1 = AttentionBlock(64)
        self.se1 = SEBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.attn2 = AttentionBlock(128)
        self.se2 = SEBlock(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.attn3 = AttentionBlock(256)
        self.se3 = SEBlock(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.depth_decoder1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.depth_out_scale1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.depth_decoder2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.depth_out_scale2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.depth_decoder3 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.depth_out_scale3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.depth_output = nn.Conv2d(64 + 64, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()


        self.flow_decoder1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.flow_out_scale1 = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)

        self.flow_decoder2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.flow_out_scale2 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        self.flow_decoder3 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.flow_out_scale3 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.flow_output = nn.Sequential(
            nn.Conv2d(64 + 64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, left_event, right_event=None, left_rgb=None, right_rgb=None):
        skip_connections = {}


        if right_event is not None:
            event_features = torch.cat([left_event, right_event], dim=1)
        else:
            event_features = left_event


        if self.use_rgb and left_rgb is not None:
            rgb_features = self.rgb_encoder(left_rgb)
            x = torch.cat([event_features, rgb_features], dim=1)
        else:
            x = event_features


        x = self.encoder1(x)
        x = self.attn1(x)
        x = self.se1(x)
        skip_connections['skip0'] = x.clone()
        x = self.pool1(x)


        x = self.encoder2(x)
        x = self.attn2(x)
        x = self.se2(x)
        skip_connections['skip1'] = x.clone()
        x = self.pool2(x)


        x = self.encoder3(x)
        x = self.attn3(x)
        x = self.se3(x)
        skip_connections['skip2'] = x.clone()
        x = self.pool3(x)


        x = self.encoder4(x)


        depth_preds = []
        d = self.depth_decoder1(x)
        depth_preds.append(self.sigmoid(self.depth_out_scale1(d)))

        d = self.up1(d)
        d = torch.cat([d, skip_connections['skip2']], dim=1)
        d = self.depth_decoder2(d)
        depth_preds.append(self.sigmoid(self.depth_out_scale2(d)))

        d = self.up2(d)
        d = torch.cat([d, skip_connections['skip1']], dim=1)
        d = self.depth_decoder3(d)
        depth_preds.append(self.sigmoid(self.depth_out_scale3(d)))

        d = self.up3(d)
        d = torch.cat([d, skip_connections['skip0']], dim=1)
        depth_final = self.sigmoid(self.depth_output(d))
        depth_preds.append(depth_final)


        flow_preds = []
        f = self.flow_decoder1(x)
        flow_preds.append(self.flow_out_scale1(f))

        f = self.up1(f)
        f = torch.cat([f, skip_connections['skip2']], dim=1)
        f = self.flow_decoder2(f)
        flow_preds.append(self.flow_out_scale2(f))

        f = self.up2(f)
        f = torch.cat([f, skip_connections['skip1']], dim=1)
        f = self.flow_decoder3(f)
        flow_preds.append(self.flow_out_scale3(f))

        f = self.up3(f)
        f = torch.cat([f, skip_connections['skip0']], dim=1)
        flow_final = self.flow_output(f)
        flow_preds.append(flow_final)

        return {
            'depth': depth_final,
            'flow': flow_final,
            'depth_preds': depth_preds,
            'flow_preds': flow_preds
        }
