import os
import os.path as osp

import torch
import torch.distributed as dist

import mmcv
from mmcv.parallel import collate, scatter
from mmdet.core.evaluation.eval_hooks import CocoDistEvalmAPHook


class CocoDistEvalmAPHook_(CocoDistEvalmAPHook):
    
    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]
            
            if not isinstance(data_gpu['img'], list):
                data_gpu['img'] = [data_gpu['img']]
            if not isinstance(data_gpu['img_meta'][0], list):
                data_gpu['img_meta'] = [data_gpu['img_meta']]
            
            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            # batch_size = runner.world_size
            # if runner.rank == 0:
            #     for _ in range(batch_size):
            #         prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()
