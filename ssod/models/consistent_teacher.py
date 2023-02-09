from collections import OrderedDict

import numpy as np
import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.models import DETECTORS, build_detector
from ssod.utils import log_every_n, log_image_with_boxes
from ssod.utils.structure_utils import dict_split, weighted_loss

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


@DETECTORS.register_module()
class ConsistentTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super().__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

        num_classes = self.teacher.bbox_head.num_classes
        num_scores = self.train_cfg.num_scores
        self.register_buffer(
            'scores', torch.zeros((num_classes, num_scores)))
        self.iter = 0

    def set_iter(self, step):
        self.iter = step

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
        unsup_weight = self.unsup_weight
        if self.iter < self.train_cfg.get('warmup_step', -1):
            unsup_weight = 0
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        if self.train_cfg.get('collect_keys', None):
            # In case of only sup or unsup images
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
                [teacher_data["img_metas"][idx] for idx in tidx],
            )
        student_info = self.extract_student_info(**student_data)

        losses = self.compute_pseudo_label_loss(student_info, teacher_info)
        losses['gmm_thr'] = torch.tensor(
            teacher_info['gmm_thr']).to(teacher_data["img"].device)
        return losses

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
        gt_bboxes = [bbox[:, :4] for bbox in pseudo_bboxes]
        gt_labels = pseudo_labels
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

        # if self.writer is not None:
        log_image_with_boxes(
            "bbox",
            student_info["img"][0],
            gt_bboxes[0],
            bbox_tag="pseudo_label",
            labels=gt_labels[0],
            class_names=self.CLASSES,
            interval=500,
            img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
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

    def gmm_policy(self, scores, given_gt_thr=0.5, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr

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
        thrs = []
        for i, proposals in enumerate(proposal_list):
            scores = proposals[:, 4].clone()
            scores = scores.sort(descending=True)[0]
            if len(scores) == 0:
                thrs.append(1)  # no kept pseudo boxes
            else:
                num_gt = int(scores.sum() + 0.5)
                num_gt = min(num_gt, len(scores) - 1)
                thrs.append(scores[num_gt] - 1e-5)
        # filter invalid box roughly
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
                    for proposal, proposal_label, thr in zip(
                        proposal_list, proposal_label_list, thrs
                    )
                ]
            )
        )
        scores = torch.cat([proposal[:, 4] for proposal in proposal_list])
        labels = torch.cat(proposal_label_list)
        thrs = torch.zeros_like(scores)
        for label in torch.unique(labels):
            label = int(label)
            scores_add = (scores[labels == label])
            num_buffers = len(self.scores[label])
            scores_new = torch.cat([scores_add, self.scores[label]])[
                :num_buffers]
            self.scores[label] = scores_new
            thr = self.gmm_policy(
                scores_new[scores_new > 0],
                given_gt_thr=0,
                policy=self.train_cfg.get('policy', 'high'))
            thrs[labels == label] = thr
        mean_thr = thrs.mean()
        if len(thrs) == 0:
            mean_thr.fill_(0)
        mean_thr = float(mean_thr)
        log_every_n({"gmm_thr": mean_thr})
        teacher_info["gmm_thr"] = mean_thr
        thrs = torch.split(thrs, [len(p) for p in proposal_list])
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr_tmp,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label, thr_tmp in zip(
                        proposal_list, proposal_label_list, thrs
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
