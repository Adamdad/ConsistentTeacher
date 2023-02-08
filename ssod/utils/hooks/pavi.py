# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import PaviLoggerHook

@HOOKS.register_module()
class PaviLoggerHookWithModelAssert(PaviLoggerHook):

    def __init__(self,
                 **kwargs):
        super(PaviLoggerHookWithModelAssert, self).__init__(**kwargs)

    @master_only
    def before_run(self, runner):
        super(PaviLoggerHookWithModelAssert, self).before_run(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.writer = self.writer



