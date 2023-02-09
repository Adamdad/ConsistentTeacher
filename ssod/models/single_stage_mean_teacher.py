from collections import OrderedDict

import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.core.utils import reduce_mean
from mmdet.models import DETECTORS, build_detector
from ssod.utils import log_every_n, log_image_with_boxes
from ssod.utils.structure_utils import dict_split, weighted_loss

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid


@DETECTORS.register_module()
class SingleStageMeanTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super().__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
        self.writer = None

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        # split the data into labeled and unlabeled through 'tag'
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox)
                                   for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss['num_gts'] = torch.tensor(
                sum([len(b) for b in gt_bboxes]) / len(gt_bboxes)).to(gt_bboxes[0])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        if self.train_cfg.get('collect_keys', None):
            # In case of only sup or unsup images
            num_sup = len(data_groups["sup"]['img']
                          ) if 'sup' in data_groups else 0
            num_unsup = len(
                data_groups['unsup_student']['img']) if 'unsup_student' in data_groups else 0
            num_sup = img.new_tensor(num_sup)
            avg_num_sup = reduce_mean(num_sup).clamp(min=1e-5)
            num_unsup = img.new_tensor(num_unsup)
            avg_num_unsup = reduce_mean(num_unsup).clamp(min=1e-5)
            collect_keys = self.train_cfg.collect_keys
            losses = OrderedDict()
            for k in collect_keys:
                if k in loss:
                    v = loss[k]
                    if isinstance(v, torch.Tensor):
                        losses[k] = v.mean()
                    elif isinstance(v, list):
                        losses[k] = sum(_loss.mean() for _loss in v)
                else:
                    losses[k] = img.new_tensor(0)
            loss = losses
            for key in loss:
                if key.startswith('sup_'):
                    loss[key] = loss[key] * num_sup / avg_num_sup
                elif key.startswith('unsup_'):
                    loss[key] = loss[key] * num_unsup / avg_num_unsup
        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher output according to the order of student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx]
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        bbox_loss, proposal_list = self.bbox_loss(
            student_info["bbox_out"],
            pseudo_bboxes,
            pseudo_labels,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(bbox_loss)
        return loss

    def bbox_loss(
        self,
        bbox_out,
        pseudo_bboxes,
        pseudo_labels,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        num_gts = [len(bbox) for bbox in gt_bboxes]
        log_every_n(
            {"bbox_gt_num": sum(num_gts) / len(gt_bboxes)}
        )
        loss_inputs = bbox_out + [gt_bboxes, gt_labels, img_metas]
        losses = self.student.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
        )
        if len([n for n in num_gts if n > 0]) < len(num_gts) / 2.:
            # TODO: Design a better way to deal with images without pseudo bbox.
            losses = weighted_loss(
                losses, weight=self.train_cfg.get('background_weight', 1e-2))
        losses['num_gts'] = torch.tensor(
            sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)).to(
            gt_bboxes[0])
        bbox_list = self.student.bbox_head.get_bboxes(
            *bbox_out, img_metas=img_metas)

        log_image_with_boxes(
            "bbox",
            student_info["img"][0],
            gt_bboxes[0],
            bbox_tag="pseudo_label",
            labels=gt_labels[0],
            class_names=self.CLASSES,
            interval=500,
            img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"]
        )
        return losses, bbox_list

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        bbox_out = self.student.bbox_head(feat)
        student_info["bbox_out"] = list(bbox_out)
        student_info["img_metas"] = img_metas
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]
                             ).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        results = \
            self.teacher.bbox_head.simple_test_bboxes(
                feat, img_metas, rescale=False)
        proposal_list = [r[0] for r in results]
        proposal_label_list = [r[1] for r in results]
        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device)
                               for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError(
                "Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]
                             ).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
