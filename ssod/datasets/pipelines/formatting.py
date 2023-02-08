import numpy as np
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formating import Collect
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor

from ssod.core import TrimapMasks

@PIPELINES.register_module()
class DefaultFormatBundleContrastive:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', "contrastive_proposals"]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results
    
    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, results):
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results


@PIPELINES.register_module()
class ExtraCollect(Collect):
    def __init__(self, *args, extra_meta_keys=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_keys = self.meta_keys + tuple(extra_meta_keys)


@PIPELINES.register_module()
class PseudoSamples(object):
    def __init__(
        self, with_bbox=False, with_mask=False, with_seg=False, fill_value=255
    ):
        """
        Replacing gt labels in original data with fake labels or adding extra fake labels for unlabeled data.
        This is to remove the effect of labeled data and keep its elements aligned with other sample.
        Args:
            with_bbox:
            with_mask:
            with_seg:
            fill_value:
        """
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.fill_value = fill_value

    def __call__(self, results):
        if self.with_bbox:
            results["gt_bboxes"] = np.zeros((0, 4))
            results["gt_labels"] = np.zeros((0,))
            if "bbox_fields" not in results:
                results["bbox_fields"] = []
            if "gt_bboxes" not in results["bbox_fields"]:
                results["bbox_fields"].append("gt_bboxes")
        if self.with_mask:
            num_inst = len(results["gt_bboxes"])
            h, w = results["img"].shape[:2]
            results["gt_masks"] = TrimapMasks(
                [
                    self.fill_value * np.ones((h, w), dtype=np.uint8)
                    for _ in range(num_inst)
                ],
                h,
                w,
            )

            if "mask_fields" not in results:
                results["mask_fields"] = []
            if "gt_masks" not in results["mask_fields"]:
                results["mask_fields"].append("gt_masks")
        if self.with_seg:
            results["gt_semantic_seg"] = self.fill_value * np.ones(
                results["img"].shape[:2], dtype=np.uint8
            )
            if "seg_fields" not in results:
                results["seg_fields"] = []
            if "gt_semantic_seg" not in results["seg_fields"]:
                results["seg_fields"].append("gt_semantic_seg")
        return results
