import torch
import torch.nn as nn
import torchvision.models as models
from .ResNet import ResNet50
from torch.nn import functional as F

class ChannelAttention_cbam(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_cbam, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention_cbam(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_cbam, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class InterFeatureFusion(nn.Module):
    def __init__(self, source_in_planes, target_in_planes, bn_eps=1e-5,
                 bn_momentum=0.1, inplace=True, norm_layer=nn.BatchNorm2d):
        super(InterFeatureFusion, self).__init__()
        
        self.convSource1 = ConvBnRelu(source_in_planes, target_in_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)

        self.convSource2 = ConvBnRelu(target_in_planes, target_in_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)

        self.convTarget = ConvBnRelu(target_in_planes, target_in_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, source_feature, target_feature):
        
        source_feature = self.upsample2(source_feature)
        source_feature = self.convSource1(source_feature)
        
        out = source_feature + target_feature + self.convSource2(source_feature * target_feature)
        out = self.convTarget(out)
        return out

class Conv2dGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(Conv2dGRUCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.in_conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                                 out_channels=2 * self.hidden_channels,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 dilation=1,
                                 padding=self.padding,
                                 bias=self.bias)

        self.out_conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  kernel_size=self.kernel_size,
                                  stride=1,
                                  dilation=1,
                                  padding=self.padding,
                                  bias=self.bias)

        self.conv_wg = nn.Conv1d(input_channels, input_channels, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(input_channels)
        self.softmax = nn.Softmax(dim=2)
        self.conv_wg_z = nn.Conv1d(input_channels, input_channels, kernel_size=1, bias=False)
        self.bn_wg_z = nn.BatchNorm1d(input_channels)

    def forward(self, input_tensor, hidden_state):
        h_cur = hidden_state
        combined = torch.cat((input_tensor, h_cur), dim=1)  # concatenate along channel axis

        N, C, H, W = input_tensor.size()
        combined_conv = self.in_conv(combined)
        cc_r, cc_z = torch.split(combined_conv, self.hidden_channels, dim=1)

        nodek = cc_r.view(N, 64, -1).permute(0, 2, 1)
        node_k = F.normalize(nodek, p=2, dim=2, eps=1e-12)
        nodeq = cc_r.view(N, 64, -1)
        node_q = F.normalize(nodeq, p=2, dim=1, eps=1e-12)
        node_v = cc_r.view(N, C, -1).permute(0, 2, 1)
        AV = torch.matmul(node_q, node_v)
        AV = torch.matmul(node_k, AV)
        AV = AV.permute(0, 2, 1).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        r = AVW.view(N, C, H, W).sigmoid()

        h_cur_bar = h_cur * r
        cc_h = self.out_conv(torch.cat((input_tensor, h_cur_bar), dim=1))
        h_bar = torch.tanh(cc_h)

        nodek_z = cc_z.view(N, 64, -1).permute(0, 2, 1)
        node_k_z = F.normalize(nodek_z, p=2, dim=2, eps=1e-12)
        nodeq_z = cc_z.view(N, 64, -1)
        node_q_z = F.normalize(nodeq_z, p=2, dim=1, eps=1e-12)
        node_v_z = cc_z.view(N, C, -1).permute(0, 2, 1)
        AV_z = torch.matmul(node_q_z, node_v_z)
        AV_z = torch.matmul(node_k_z, AV_z)
        AV_z = AV_z.permute(0, 2, 1).contiguous()
        AVW_z = self.conv_wg_z(AV_z)
        AVW_z = self.bn_wg_z(AVW_z)
        z = AVW_z.view(N, C, H, W).sigmoid()

        h_next = z * h_cur + (1 - z) * h_bar
        return h_next


class BGRDNet(nn.Module):
    def __init__(self, channel=32):
        super(BGRDNet, self).__init__()

        # Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')

        # GRU
        self.gru_1 = Conv2dGRUCell(input_channels=64, hidden_channels=64, kernel_size=3, bias=True)
        self.gru_2 = Conv2dGRUCell(input_channels=64, hidden_channels=64, kernel_size=3, bias=True)

        self.fuse2_1 = InterFeatureFusion(64, 64, bn_eps=1e-5,
                                          bn_momentum=0.1, norm_layer=nn.BatchNorm2d)
        self.fuse3_1 = InterFeatureFusion(64, 64, bn_eps=1e-5,
                                          bn_momentum=0.1, norm_layer=nn.BatchNorm2d)
        self.fuse3_2 = InterFeatureFusion(64, 64, bn_eps=1e-5,
                                          bn_momentum=0.1, norm_layer=nn.BatchNorm2d)
        self.fuse4_1 = InterFeatureFusion(64, 64, bn_eps=1e-5,
                                          bn_momentum=0.1, norm_layer=nn.BatchNorm2d)
        self.fuse4_2 = InterFeatureFusion(64, 64, bn_eps=1e-5,
                                          bn_momentum=0.1, norm_layer=nn.BatchNorm2d)
        self.fuse4_3 = InterFeatureFusion(64, 64, bn_eps=1e-5,
                                          bn_momentum=0.1, norm_layer=nn.BatchNorm2d)
        # upsample function
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Components of TEM module
        self.atten_depth_channel_0 = ChannelAttention_cbam(64)
        self.atten_depth_channel_1 = ChannelAttention_cbam(256)
        self.atten_depth_channel_2 = ChannelAttention_cbam(512)
        self.atten_depth_channel_3_1 = ChannelAttention_cbam(1024)
        self.atten_depth_channel_4_1 = ChannelAttention_cbam(2048)
        self.atten_depth_spatial_0 = SpatialAttention_cbam()
        self.atten_depth_spatial_1 = SpatialAttention_cbam()
        self.atten_depth_spatial_2 = SpatialAttention_cbam()
        self.atten_depth_spatial_3_1 = SpatialAttention_cbam()
        self.atten_depth_spatial_4_1 = SpatialAttention_cbam()

        # Components of T layer
        self.T_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # Components of predict layer
        self.predict_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predict_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predict_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predict_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.predict_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)
        )

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        # 第0层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        # layer0
        x_c = x.mul(self.atten_depth_channel_0(x_depth))
        x_s = x.mul(self.atten_depth_spatial_0(x_depth))
        x = x + x_c + x_s
        # layer0 end

        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x1_depth = self.resnet_depth.layer1(x_depth)

        # layer1
        x1_c = x1.mul(self.atten_depth_channel_1(x1_depth))
        x1_s = x1.mul(self.atten_depth_spatial_1(x1_depth))
        x1 = x1 + x1_c + x1_s
        # layer1 end

        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        # layer2
        x2_c = x2.mul(self.atten_depth_channel_2(x2_depth))
        x2_s = x2.mul(self.atten_depth_spatial_2(x2_depth))
        x2 = x2 + x2_c + x2_s
        # layer2 end

        x3_1 = self.resnet.layer3_1(x2)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        # layer3_1
        x3_1_c = x3_1.mul(self.atten_depth_channel_3_1(x3_1_depth))
        x3_1_s = x3_1.mul(self.atten_depth_spatial_3_1(x3_1_depth))
        x3_1 = x3_1 + x3_1_c + x3_1_s
        # layer3_1 end

        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        # layer4_1
        x4_1_c = x4_1.mul(self.atten_depth_channel_4_1(x4_1_depth))
        x4_1_s = x4_1.mul(self.atten_depth_spatial_4_1(x4_1_depth))
        x4_1 = x4_1 + x4_1_c + x4_1_s

        # layer TM
        x_t = self.T_layer0(x)
        x1_t = self.T_layer1(x1)
        x2_t = self.T_layer2(x2)
        x3_1_t = self.T_layer3(x3_1)
        x4_1_t = self.T_layer4(x4_1)

        # upsample
        fm2_1 = self.fuse2_1(x2_t, x1_t)
        fm3_1 = self.fuse3_1(x3_1_t, x2_t)
        fm3_2 = self.fuse3_2(fm3_1, fm2_1)
        fm4_1 = self.fuse4_1(x4_1_t, x3_1_t)
        fm4_2 = self.fuse4_2(fm4_1, fm3_1)
        fm4_3 = self.fuse4_3(fm4_2, fm3_2)

        # Fusion 1:
        h1 = self.gru_1(input_tensor=x_t, hidden_state=x_t)
        h2 = self.gru_1(input_tensor=x1_t, hidden_state=h1)
        h3 = self.gru_1(input_tensor=fm2_1, hidden_state=h2)
        h4 = self.gru_1(input_tensor=fm3_2, hidden_state=h3)
        h5 = self.gru_1(input_tensor=fm4_3, hidden_state=h4)

        # Fusion 2:
        h_5 = self.gru_2(input_tensor=fm4_3, hidden_state=fm4_3)
        h_4 = self.gru_2(input_tensor=fm3_2, hidden_state=h_5)
        h_3 = self.gru_2(input_tensor=fm2_1, hidden_state=h_4)
        h_2 = self.gru_2(input_tensor=x1_t, hidden_state=h_3)
        h_1 = self.gru_2(input_tensor=x_t, hidden_state=h_2)

        # produce initial saliency map by decoder1
        h1_c = torch.cat((h_1, h1), 1)
        h1_p = self.predict_layer1(h1_c)

        h2_c = torch.cat((h_2, h2), 1)
        h2_p = self.predict_layer2(h2_c)

        h3_c = torch.cat((h_3, h3), 1)
        h3_p = self.predict_layer3(h3_c)

        h4_c = torch.cat((h_4, h4), 1)
        h4_p = self.predict_layer4(h4_c)

        h5_c = torch.cat((h_5, h5), 1)
        h5_p = self.predict_layer5(h5_c)

        y = h1_p + h2_p + h3_p + h4_p + h5_p
        
        return y, h1_p, h2_p, h3_p, h4_p, h5_p

    # initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)
