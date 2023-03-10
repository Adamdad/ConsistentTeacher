# import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import ssod
# from mmdet.apis import inference_detector, init_detector, 
# from mmcv import Config
config_name = 'configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_2p.py'
checkpoint = '../ckpt_logs/ConsistentTeacher/consistent_teacher_r50_fpn_coco_180k_2p/consistent_teacher_r50_fpn_coco_180k_2p_iter_180000.pth'
img = mmcv.imread('assets/492060815_ec07c64c09_z.jpg')
# config = Config.fromfile(config_name)
model = init_detector(config_name, checkpoint, device='cuda:0')
result = inference_detector(model, img)
show_result_pyplot(model, img, result)