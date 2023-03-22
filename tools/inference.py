from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core import get_classes
import mmcv
import ssod
import warnings
from mmcv.runner import load_checkpoint

config_name = 'configs/consistent-teacher/consistent_teacher_r50_fpn_coco_720k_fulldata.py'
checkpoint = '/home/yangxingyi/yxy/Projects/Semi-Det/ckpt_logs/ConsistentTeacher/consistent_teacher_r50_fpn_coco_720k_fulldata/consistent_teacher_r50_fpn_coco_720k_fulldata_iter_720000-d932808f.pth'
img = mmcv.imread('assets/7374755946_b96148cfb3_z.jpg')
cfg = mmcv.Config.fromfile(config_name)
model = init_detector(config_name, checkpoint=None, device='cpu')
checkpoint = load_checkpoint(model, checkpoint, revise_keys=[(r'^teacher\.', '')])
if 'CLASSES' in cfg:
    model.CLASSES = cfg['CLASSES']

elif 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']

else:
    warnings.simplefilter('once')
    warnings.warn('Class names are not saved in the checkpoint\'s '
                    'meta data, use COCO classes by default.')
    model.CLASSES = get_classes('coco')

print(model.CLASSES)

result = inference_detector(model, img)
show_result_pyplot(model, img, result)