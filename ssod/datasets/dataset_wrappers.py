from mmdet.datasets import DATASETS, ConcatDataset, build_dataset, CocoDataset


@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]


@DATASETS.register_module()
class CocoDataset_contrast_proposal(CocoDataset):
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['contrastive proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)