import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F




import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
            bias=False  # BatchNorm 已经包含偏置项
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # 减少内存拷贝


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GEFU(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.linear_value = nn.Linear(in_dim, hidden_dim)
        self.linear_gate = nn.Linear(in_dim, hidden_dim)
        if hidden_dim != out_dim:
            self.out_proj = nn.Linear(hidden_dim, out_dim)
        else:
            self.out_proj = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        value = self.linear_value(x)
        gate = torch.sigmoid(self.linear_gate(x))
        x = value * gate
        x = self.act(x)
        x = self.out_proj(x)
        return x
class GEFUBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.out_dim = out_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.gefu = GEFU(in_dim, hidden_dim, self.out_dim)

        if in_dim != self.out_dim:
            self.residual_proj = nn.Linear(in_dim, self.out_dim)
        else:
            self.residual_proj = nn.Identity()
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.gefu(x)
        residual = self.residual_proj(residual)
        x = x + residual
        return x
class Structural(nn.Module):
    def __init__(self, embedding_dim=21, max_seq_len=15, filter_num=64, filter_sizes=None):
        super(Structural, self).__init__()
        if filter_sizes is None:
            filter_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.convs = nn.ModuleList(
            [nn.Conv2d(embedding_dim, filter_num, fsz, stride=2, padding=(fsz[0] // 2, fsz[1] // 2)) for fsz in filter_sizes]
        )
        self.fc = nn.Linear(len(filter_sizes) * filter_num, 1280)
        self.dropout = nn.Dropout(0.6)
    def forward(self, graph, device):
        graph = graph.to(device)#(32,183,15,21)
        graph = graph.transpose(2, 3)
        graph = graph.transpose(1, 2)
        conv_outs = [F.relu(conv(graph)) for conv in self.convs]
        pooled_outs = [F.adaptive_avg_pool2d(conv_out, (1, 1)).view(graph.size(0), -1) for conv_out in conv_outs]
        concat_out = torch.cat(pooled_outs, 1)
        representation = self.fc(concat_out)
        representation = self.dropout(representation)
        return representation







class peptide(nn.Module):
    def __init__(self, d_model=1280,out_channels=1280, d_ff=1280, n_layers=4, n_heads=4, max_len=183):
        super(peptide, self).__init__()
        self.dilated_conv = DilatedConv1D(in_channels=1280, out_channels=640, kernel_size=3, dilation=2)
        # self.dilated_conv = DilatedConv1D(embed_dim=1280, num_classes=1280)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(320, 256)
        )
    def forward(self, esm_features):
        """
        esm_features: [B, L, 1280]  ← 直接传入预提取的 ESM 特征
        """
        x = esm_features
        x=x.permute(0, 2, 1)
        x=self.dilated_conv(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        logits = self.fc(x)
        return logits

class BioFeature(nn.Module):
    def __init__(self, input_dim=1280):
        """
        使用 1D-CNN 提取生物特征（如 BLOSUM62）

        Args:
            input_dim (int): 输入特征维度（默认 20）
            hidden_dim (int): 卷积层输出通道数
            kernel_size (int): 卷积核大小
            dropout (float): dropout 概率
        """
        super(BioFeature, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding=3 // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # 入1280，出64，drop0.6,esm.pt#0.7一层效果最好
            nn.BatchNorm1d(64),
        )

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_dim)
        return: shape (batch_size, hidden_dim, seq_len)
        """
        x = x.transpose(1, 2)  # -> (batch_size, input_dim, seq_len)
        x = self.cnn(x) # -> (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        x, indices = x.max(dim=1)
        return x



class ToxiPep_Model(nn.Module):
    def __init__(self,  d_model, d_ff, n_layers, n_heads, max_len, structural_config, cross_attention_dim=256):
        super(ToxiPep_Model, self).__init__()
        self.peptide_model = peptide(d_model, d_ff, n_layers, n_heads, max_len)
        self.bio_model=BioFeature(1280)
        self.structural_model = Structural(**structural_config)
        self.structural_linear = nn.Linear(1280, cross_attention_dim)
        self.fc = nn.Linear(cross_attention_dim, 1)
        self.gefub = nn.Sequential(
            GEFUBlock(in_dim=576, hidden_dim=cross_attention_dim,out_dim=cross_attention_dim),
            nn.Dropout(0.1),
            GEFUBlock(in_dim=cross_attention_dim, hidden_dim=cross_attention_dim,out_dim=cross_attention_dim),
            nn.Dropout(0.1),
        )
    def forward(self, input_esmids, graph_features, device):
        peptide_output = self.peptide_model(input_esmids)
        bio_output = self.bio_model(input_esmids)
        # bio_pep= torch.cat((peptide_output, bio_output), dim=1)
        structural_output = self.structural_model(graph_features, device)
        structural_output = self.structural_linear(structural_output)
        combined_features = torch.cat((structural_output,peptide_output,bio_output), dim=1)
        logits = self.gefub(combined_features)
        logits = self.fc(logits).squeeze(1)
        return logits

