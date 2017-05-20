from pathlib import Path

import mcnn.operations as operations
from mcnn.model import McnnModel
from mcnn.samples import HorizontalDataset

if __name__ == '__main__':
    root = Path.home() / 'data' / 'UCR_TS_Archive_2015'

    dataset_name = 'Plane'

    dataset = HorizontalDataset(root / dataset_name / (dataset_name + '_TRAIN'),
                                root / dataset_name / (dataset_name + '_TEST'))

    sample_length = 128
    pooling_factor = sample_length // 32
    local_filter_width = sample_length // 32
    # full_filter_width = pooling_factor // 4
    full_filter_width = pooling_factor

    model = McnnModel(batch_size=64,
                      downsample_strides=[2, 4, 8, 16],
                      smoothing_window_sizes=[8, 16, 32, 64],
                      pooling_factor=pooling_factor,
                      channel_count=256,
                      local_filter_width=local_filter_width,
                      full_filter_width=full_filter_width,
                      layer_size=256,
                      full_pool_size=4,
                      num_classes=dataset.target_classes_count,
                      learning_rate=1e-3,
                      sample_length=sample_length)

    should_train = True
    should_vis = False

    if should_train:
        operations.train(model,
                         dataset,
                         step_count=200,
                         checkpoint_dir=Path('checkpoints') / dataset_name,
                         log_dir=Path('logs') / dataset_name,
                         steps_per_checkpoint=100,
                         feature_name='')

    if should_vis:
        operations.deconv(model,
                          dataset,
                          sample_count=1000,
                          checkpoint_dir=Path('checkpoints') / dataset_name,
                          feature_name='')
