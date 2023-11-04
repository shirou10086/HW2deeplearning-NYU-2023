import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from engine import train_one_epoch, evaluate
import utils

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import torchvision.transforms as T
'''remember to :
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")'''

class MyTransform(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

coco_dataset = CocoDetection(root='data/train2017', annFile='data/annotations/instances_train2017.json', transform=MyTransform())

train_size = int(0.8 * len(coco_dataset))
test_size = len(coco_dataset) - train_size
train_dataset, test_dataset = random_split(coco_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

def collate_fn(batch):
    return tuple(zip(*batch))

backbone = torchvision.models.resnet50(pretrained=True).features
backbone.out_channels = 2048

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2,
)

model_resnet = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_resnet.to(device)

optimizer = torch.optim.SGD(model_resnet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model_resnet, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model_resnet, data_loader_test, device=device)
print("Training completed!")
