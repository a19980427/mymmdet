import torch

print(torch.cuda.is_available())

from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# faster_rcnn voc
config_file = '/home/qihang/sayuri/mmdetection-master/configs/swin/faster_rcnn_swin_t-p4-w7_fpn_3x_voc.py'
checkpoint_file = '/root/autodl-tmp/qihang/weights_cache/faster_rcnn_voc/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'demo.jpg'
result = inference_detector(model, img)

# show the results
show_result_pyplot(model, img, result)
