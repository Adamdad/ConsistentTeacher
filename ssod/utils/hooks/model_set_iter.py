from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SetIterInfoHook(Hook):
    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_iter(curr_step)
