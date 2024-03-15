# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import logging

from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop, BaseLoop

from mmdet.registry import LOOPS
from mmengine.runner.amp import autocast


@LOOPS.register_module()
class TeacherStudentValLoop(ValLoop):
    """Loop for validation of model teacher and student."""

    def run(self):
        """Launch validation for model teacher and student."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

        predict_on = model.semi_test_cfg.get('predict_on', None)
        multi_metrics = dict()
        for _predict_on in ['teacher', 'student']:
            model.semi_test_cfg['predict_on'] = _predict_on
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            multi_metrics.update(
                {'/'.join((_predict_on, k)): v
                 for k, v in metrics.items()})
        model.semi_test_cfg['predict_on'] = predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')

@LOOPS.register_module()
class MultiSourceValLoop(BaseLoop):
    def __init__(self, runner, dataloader, evaluator, extra_dataloaders, extra_evaluators, fp16=False):
        super().__init__(runner, dataloader)
        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        
        # build extra dataloaders
        self.extra_dataloaders = []
        for extra_item in extra_dataloaders:
            if isinstance(extra_item, dict):
                # Determine whether or not different ranks use different seed.
                diff_rank_seed = runner._randomness_cfg.get(
                    'diff_rank_seed', False)
                self.extra_dataloaders.append(runner.build_dataloader(
                    extra_item, seed=runner.seed, diff_rank_seed=diff_rank_seed))
            else:
                self.extra_dataloaders.append(extra_item)
        
        # build extra evaluators
        self.extra_evaluators = []
        for extra_item in extra_evaluators:
            if isinstance(extra_item, (dict, list)):
                self.extra_evaluators.append(runner.build_evaluator(extra_item))  # type: ignore
            else:
                assert isinstance(extra_item, Evaluator), (
                    'evaluator must be one of dict, list or Evaluator instance, '
                    f'but got {type(extra_item)}.')
                self.extra_evaluators.append(extra_item)  # type: ignore
        
        assert len(self.extra_dataloaders) == len(self.extra_evaluators), 'the number of \
                extra dataloaders must be the same as evaluators'
        self.fp16 = fp16
        
    # def run(self):
    #     self.runner.call_hooks('before_val_epoch')
    #     for idx, data_batch in enumerate(self.dataloader):
    #         self.runner.call_hooks(
    #             'before_val_iter', batch_idx=idx, data_batch=data_batch)
    #         outputs = self.run_iter(idx, data_batch)
    #         self.runner.call_hooks(
    #             'after_val_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)
    #     metric = self.evaluator.evaluate()

    #     # add extra loop for validation purpose
    #     for idx, data_batch in enumerate(self.dataloader2):
    #         # add new hooks
    #         self.runner.call_hooks(
    #             'before_valloader2_iter', batch_idx=idx, data_batch=data_batch)
    #         self.run_iter(idx, data_batch)
    #         # add new hooks
    #         self.runner.call_hooks(
    #             'after_valloader2_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)
    #     metric2 = self.evaluator.evaluate()

    #     self.runner.call_hooks('after_val_epoch')
    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        # main dataloader
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        # source dataloader
        for source_id in range(len(self.extra_dataloaders)):
            self.runner.call_hook('before_val_epoch')
            self.runner.model.eval()
            extra_dataloader = self.extra_dataloaders[source_id]
            extra_evaluator = self.extra_evaluators[source_id]
            # add meta info 
            if hasattr(extra_dataloader.dataset, 'metainfo'):
                extra_evaluator.dataset_meta = extra_dataloader.dataset.metainfo
                self.runner.visualizer.dataset_meta = \
                    extra_dataloader.dataset.metainfo
            else:
                print_log(
                    f'Dataset {extra_dataloader.dataset.__class__.__name__} has no '
                    'metainfo. ``dataset_meta`` in evaluator, metric and '
                    'visualizer will be None.',
                    logger='current',
                    level=logging.WARNING)
            for idx, data_batch in enumerate(extra_dataloader):
                self.extra_run_iter(idx, data_batch, source_id)
            # compute metrics
            metrics.update(extra_evaluator.evaluate(len(self.extra_dataloaders[source_id].dataset)))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

    @torch.no_grad()
    def extra_run_iter(self, idx, data_batch: Sequence[dict], extra_source_idx):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.extra_evaluators[extra_source_idx].process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)