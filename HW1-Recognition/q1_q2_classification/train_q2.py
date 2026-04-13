import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Manual ResNet-18:
    conv1 -> bn1 -> relu -> maxpool
    layer1: 2 basic blocks
    layer2: 2 basic blocks
    layer3: 2 basic blocks
    layer4: 2 basic blocks
    avgpool -> fc
    """

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 layers = [2, 2, 2, 2]
        self.layer1 = self._make_layer(planes=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ##################################################################
        # FC for VOC classification
        ##################################################################
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

        self._init_weights()

    def _make_layer(self, planes, blocks, stride):
        downsample = None

        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride=stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def _init_weights(self):
        # Match torchvision-style initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # VOC classifier head initialization
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward_features(self, x):
        x = self.conv1(x)      # [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # [B, 64, 56, 56]

        x = self.layer1(x)     # [B, 64, 56, 56]
        x = self.layer2(x)     # [B, 128, 28, 28]
        x = self.layer3(x)     # [B, 256, 14, 14]
        x = self.layer4(x)     # [B, 512, 7, 7]

        x = self.avgpool(x)    # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        return x

    def forward(self, x):
        ##################################################################
        # Return raw logits, no sigmoid here
        ##################################################################
        x = self.forward_features(x)
        x = self.fc(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def load_imagenet_pretrained(self):
        """
        Load ImageNet pretrained weights from torchvision official ResNet-18.
        If num_classes != 1000, only backbone weights are loaded and fc is kept
        as VOC classifier head.
        """
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        official = torchvision.models.resnet18(weights=weights)
        state_dict = official.state_dict()

        if self.fc.out_features == 1000:
            msg = self.load_state_dict(state_dict, strict=True)
            assert len(msg.missing_keys) == 0, msg.missing_keys
            assert len(msg.unexpected_keys) == 0, msg.unexpected_keys
        else:
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            msg = self.load_state_dict(state_dict, strict=False)

            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, msg.missing_keys
            assert len(msg.unexpected_keys) == 0, msg.unexpected_keys

        return self


@torch.no_grad()
def torchvision_backbone_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def assert_resnet18_matches_torchvision(device):
    """
    Full-output equivalence test:
    manual ResNet-18 (1000 classes) vs official torchvision resnet18
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    ref = torchvision.models.resnet18(weights=weights).to(device).eval()
    mine = ResNet(num_classes=1000).to(device).eval()
    mine.load_imagenet_pretrained()

    x = torch.randn(2, 3, 224, 224, device=device)

    y_ref = ref(x)
    y_mine = mine(x)

    max_abs_diff = (y_ref - y_mine).abs().max().item()
    assert torch.allclose(y_ref, y_mine, atol=1e-6, rtol=1e-5), \
        f"Manual ResNet-18 != torchvision ResNet-18, max abs diff = {max_abs_diff}"

    print(f"[OK] Full output matches torchvision. max_abs_diff = {max_abs_diff:.3e}")


@torch.no_grad()
def assert_voc_backbone_matches_torchvision(device, num_classes):
    """
    Backbone-feature equivalence test for VOC model:
    compare penultimate 512-d features only, because fc differs.
    """
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    ref = torchvision.models.resnet18(weights=weights).to(device).eval()
    mine = ResNet(num_classes=num_classes).to(device).eval()
    mine.load_imagenet_pretrained()

    x = torch.randn(2, 3, 224, 224, device=device)

    feat_ref = torchvision_backbone_features(ref, x)
    feat_mine = mine.forward_features(x)

    max_abs_diff = (feat_ref - feat_mine).abs().max().item()
    assert torch.allclose(feat_ref, feat_mine, atol=1e-6, rtol=1e-5), \
        f"VOC backbone != torchvision backbone, max abs diff = {max_abs_diff}"

    print(f"[OK] VOC backbone features match torchvision. max_abs_diff = {max_abs_diff:.3e}")

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    ##################################################################
    args = ARGS(
        epochs=50,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr=1e-2,       #TODO,
        batch_size=32, #TODO,
        step_size=15,  #TODO,
        gamma=0.1,     #TODO
        save_at_end=True,
    )

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################
    # test with reference
    assert_resnet18_matches_torchvision(args.device)
    assert_voc_backbone_matches_torchvision(args.device, len(VOCDataset.CLASS_NAMES))
    # ImageNet backbone
    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)
    model.load_imagenet_pretrained()

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=8e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4,)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
