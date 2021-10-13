import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_model import etri_resnet101

class ImageModel(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(ImageModel, self).__init__()

        self.image_convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # ( 100*100 이미지 기준 최종적으로 [batch_size, 512, 24, 24] shape의 output 도출 )
        # self.resnet = etri_resnet101()

        # self.hidden_Q = nn.Linear(hidden_size, hidden_size)
        # self.resnet_K = nn.Linear(2048, hidden_size)
        # self.hidden_V = nn.Linear(2048, hidden_size)

        self.width = 24 # 우선 100*100 기준 width = 24

        self.W_hv = nn.Linear(hidden_size + self.width*self.width, hidden_size)
        # self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(self.width*self.width, hidden_size)
        # self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, text_embedding, image, device):
        eps = 1e-6
        image_vector = self.image_convolution(image) # [batch_size, sequence_length, 24, 24]
        # image_vector = self.resnet(image) # [batch_size, 49, 2048]

        # Q = F.gelu(self.hidden_Q(text_embedding)) # [batch_size, 512, 768]
        # K = F.gelu(self.resnet_K(image_vector)) # [batch_size, 49, 768]
        # V = F.gelu(self.hidden_V(image_vector)) # [batch_size, 49, 768]

        # e = torch.bmm(Q, K.transpose(1, 2)) # [batch_size, 512, 49]

        # alpha = nn.Softmax(dim=2)(e)

        # multimodal_hidden = torch.bmm(e, V) # [batch_size, 512, 768]

        # embedding_output = self.dropout(
        #     self.LayerNorm(text_embedding + multimodal_hidden)
        # )
        # return embedding_output

        batch_size, sequence_length = image_vector.shape[0], image_vector.shape[1]
        image_vector = image_vector.view(batch_size, sequence_length, -1) # [batch_size, sequence_length, 24*24]

        weight_v = F.relu(self.W_hv(torch.cat((image_vector, text_embedding), dim=-1)))
        # weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(image_vector)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        text_and_image_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(text_and_image_embedding + text_embedding)
        )

        return embedding_output