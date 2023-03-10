from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core import get_classes
import mmcv
import ssod
import warnings
from mmcv.runner import load_checkpoint

config_name = 'configs/consistent-teacher/consistent_teacher_r50_fpn_coco_720k_fulldata.py'
checkpoint = '../ckpt_logs/ConsistentTeacher/consistent_teacher_r50_fpn_coco_720k_fulldata/consistent_teacher_r50_fpn_coco_720k_fulldata_iter_720000-d932808f.pth'
img = mmcv.imread('assets/492060815_ec07c64c09_z.jpg')

model = init_detector(config_name, checkpoint=None, device='cpu')
checkpoint = load_checkpoint(model, checkpoint, revise_keys=[(r'^teacher\.', '')])
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    warnings.simplefilter('once')
    warnings.warn('Class names are not saved in the checkpoint\'s '
                    'meta data, use COCO classes by default.')
    model.CLASSES = get_classes('coco')
result = inference_detector(model, img)
show_result_pyplot(model, img, result)