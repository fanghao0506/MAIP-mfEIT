"""MBA-Net from the authors' archived MAIP implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayerNorm(nn.Module):
    """A wrapper module for LayerNorm to work with Conv2d outputs."""

    def __init__(self, num_features):
        super(ConvLayerNorm, self).__init__()
        self.num_features = num_features
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x = self.layer_norm(x)
        x = x.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x

class ResidualConv(nn.Module):

    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(ConvLayerNorm(input_dim), nn.LeakyReLU(negative_slope=0.0001), nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding), ConvLayerNorm(output_dim), nn.LeakyReLU(negative_slope=0.0001), nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
        self.conv_skip = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1), ConvLayerNorm(output_dim))

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class Squeeze_Excite_Block(nn.Module):

    def __init__(self, channel, reduction=12):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.LeakyReLU(negative_slope=0.0001, inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPP(nn.Module):

    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_block1 = nn.Sequential(nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]), nn.LeakyReLU(negative_slope=0.0001, inplace=True), ConvLayerNorm(out_dims))
        self.aspp_block2 = nn.Sequential(nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]), nn.LeakyReLU(negative_slope=0.0001, inplace=True), ConvLayerNorm(out_dims))
        self.aspp_block3 = nn.Sequential(nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]), nn.LeakyReLU(negative_slope=0.0001, inplace=True), ConvLayerNorm(out_dims))
        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Upsample_(nn.Module):

    def __init__(self, scale=2):
        super(Upsample_, self).__init__()
        self.upsample = nn.Upsample(mode='bilinear', scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)

class AttentionBlock(nn.Module):

    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()
        self.conv_encoder = nn.Sequential(ConvLayerNorm(input_encoder), nn.LeakyReLU(negative_slope=0.0001), nn.Conv2d(input_encoder, output_dim, 3, padding=1), nn.MaxPool2d(2, 2))
        self.conv_decoder = nn.Sequential(ConvLayerNorm(input_decoder), nn.LeakyReLU(negative_slope=0.0001), nn.Conv2d(input_decoder, output_dim, 3, padding=1))
        self.conv_attn = nn.Sequential(ConvLayerNorm(output_dim), nn.LeakyReLU(negative_slope=0.0001), nn.Conv2d(output_dim, 1, 1))

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class ResUnetPlusPlus(nn.Module):

    def __init__(self, channel=1, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(channel, filters[0], kernel_size=3, padding=1), ConvLayerNorm(filters[0]), nn.LeakyReLU(negative_slope=0.0001), nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1))
        self.input_skip = nn.Sequential(nn.Conv2d(channel, filters[0], kernel_size=3, padding=1))
        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)
        self.aspp_bridge = ASPP(filters[3], filters[4], rate=[1, 2, 3])
        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)
        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)
        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)
        self.aspp_out = ASPP(filters[1], filters[0])
        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], channel, 1), nn.LeakyReLU(negative_slope=0.0001))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)
        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)
        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)
        x5 = self.aspp_bridge(x4)
        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)
        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)
        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)
        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)
        return out

class MultiBranchNetwork(nn.Module):

    def __init__(self):
        super(MultiBranchNetwork, self).__init__()
        self.branches = nn.ModuleList([ResUnetPlusPlus() for _ in range(4)])
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=3, padding=1), ConvLayerNorm(32), nn.LeakyReLU(negative_slope=0.0001), nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 4, 1), nn.Sigmoid())
        self.attention = ChannelWeightedAttentionModule(4)

    def forward(self, x):
        channels = torch.split(x, 1, dim=1)
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_output = branch(channels[i])
            branch_outputs.append(branch_output)
        concatenated = torch.cat(branch_outputs, dim=1)
        conv1_output = self.conv1(concatenated)
        conv2_output = self.conv2(conv1_output)
        out = self.attention(conv2_output)
        return out

class ChannelWeightedAttentionModule(nn.Module):

    def __init__(self, channels):
        super(ChannelWeightedAttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(channels, channels))
        self.channel_weights = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        attention_scores = F.softmax(self.attention_weights, dim=-1)
        attention_scores_expanded = attention_scores.unsqueeze(0).expand(batch_size, -1, -1)
        x = x.view(batch_size, num_channels, -1)
        x = torch.bmm(attention_scores_expanded, x)
        x = x.view(batch_size, num_channels, height, width)
        x = x * self.channel_weights.view(1, -1, 1, 1)
        return x
