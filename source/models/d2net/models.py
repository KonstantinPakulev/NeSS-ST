import torch
import torch.nn as nn
import torch.nn.functional as F


class D2Net(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super().__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(use_relu=use_relu, use_cuda=use_cuda)

        # self.detection = HardDetectionModule()
        self.detection = SoftDetectionModule()
        self.localization = HandcraftedLocalizationModule()

    def forward(self, batch):
        _, _, h, w = batch.size()
        x = self.dense_feature_extraction(batch)

        score, depth_wise_max_ids = self.detection(x)
        desc = F.normalize(x)

        loc = self.localization(x)

        depth_wise_max_ids = depth_wise_max_ids.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        loc = torch.gather(loc, dim=2, index=depth_wise_max_ids).squeeze(2).clamp(min=-0.5, max=0.5)

        return score, desc, loc


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512

        self.use_relu = use_relu

        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor([[0, 1., 0], [0, -2., 0], [0, 1., 0]]).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor([[0, 0, 0], [1., -2., 1.], [0, 0, 0]]).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        score = torch.max(detected, dim=1)[0].unsqueeze(1).float()

        return score


class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        depth_wise_max, depth_wise_max_ids = torch.max(batch, dim=1)
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0].unsqueeze(1)

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1, 1)

        return score, depth_wise_max_ids.unsqueeze(1)


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor([[0, -0.5, 0],
                                       [0, 0, 0],
                                       [0,  0.5, 0]]).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor([[0, 0, 0],
                                       [-0.5, 0, 0.5],
                                       [0, 0, 0]]).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor([[0, 1., 0],
                                        [0, -2., 0],
                                        [0, 1., 0]]).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor([[1., 0, -1.],
                                               [0, 0., 0],
                                               [-1., 0, 1.]]).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor([[0, 0, 0],
                                        [1., -2., 1.],
                                        [0, 0, 0]]).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        loc = torch.stack([step_i, step_j], dim=1)
        loc[torch.isnan(loc)] = 0.0

        return loc
