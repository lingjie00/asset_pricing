```bash

(deep_learning_asset_pricing)  Deep_Learning_Asset_Pricing git:(documentation) ✗ python3 run.py --config=config/config.json --logdir=output --saveBestFreq=128 --printOnConsole=True --saveLog=True --ignoreEpoch=32

/home/ling/miniconda3/envs/deep_learning_asset_pricing/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:523: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/ling/miniconda3/envs/deep_learning_asset_pricing/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:524: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/ling/miniconda3/envs/deep_learning_asset_pricing/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/ling/miniconda3/envs/deep_learning_asset_pricing/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/ling/miniconda3/envs/deep_learning_asset_pricing/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/ling/miniconda3/envs/deep_learning_asset_pricing/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:532: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
>==================> Read the following in config:
{
    "learning_rate": 0.001,
    "num_layers_moment": 0,
    "num_units_rnn": [
        4
    ],
    "macro_feature_file_test": "../datasets/macro/macro_test.npz",
    "optimizer": "Adam",
    "macro_feature_file": "../datasets/macro/macro_train.npz",
    "use_rnn": true,
    "weighted_loss": true,
    "hidden_dim": [
        64,
        64
    ],
    "cell_type_rnn_moment": "lstm",
    "cell_type_rnn": "lstm",
    "macro_feature_file_valid": "../datasets/macro/macro_valid.npz",
    "num_epochs_moment": 64,
    "tSize_test": 300,
    "tSize": 240,
    "individual_feature_dim": 46,
    "tSize_valid": 60,
    "num_condition_moment": 8,
    "loss_factor": 1.0,
    "num_layers_rnn_moment": 1,
    "individual_feature_file_valid": "../datasets/char/Char_valid.npz",
    "num_epochs": 1024,
    "individual_feature_file": "../datasets/char/Char_train.npz",
    "num_units_rnn_moment": [
        32
    ],
    "num_epochs_unc": 256,
    "dropout": 0.95,
    "macro_feature_dim": 178,
    "num_layers_rnn": 1,
    "sub_epoch": 4,
    "individual_feature_file_test": "../datasets/char/Char_test.npz",
    "hidden_dim_moment": [],
    "num_layers": 2,
    "macro_idx": null
}
>==================> Creating data layer
>==================> Data layer created
>==================> Trainable variables (scope=Model_Layer)
Name: Model_Layer/RNN_Layer/rnn/lstm_cell/kernel:0 and shape: (182, 16)
Name: Model_Layer/RNN_Layer/rnn/lstm_cell/bias:0 and shape: (16,)
Name: Model_Layer/NN_Layer/dense_layer_0/dense/kernel:0 and shape: (50, 64)
Name: Model_Layer/NN_Layer/dense_layer_0/dense/bias:0 and shape: (64,)
Name: Model_Layer/NN_Layer/dense_layer_1/dense/kernel:0 and shape: (64, 64)
Name: Model_Layer/NN_Layer/dense_layer_1/dense/bias:0 and shape: (64,)
Name: Model_Layer/NN_Layer/last_dense_layer/dense/kernel:0 and shape: (64, 1)
Name: Model_Layer/NN_Layer/last_dense_layer/dense/bias:0 and shape: (1,)
>==================> Number of parameters: 10417
/home/ling/miniconda3/envs/deep_learning_asset_pricing/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:112: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
>==================> Trainable variables (scope=Moment_Layer)
Name: Moment_Layer/RNN_Layer/rnn/lstm_cell/kernel:0 and shape: (210, 128)
Name: Moment_Layer/RNN_Layer/rnn/lstm_cell/bias:0 and shape: (128,)
Name: Moment_Layer/NN_Layer/last_dense_layer/dense/kernel:0 and shape: (78, 8)
Name: Moment_Layer/NN_Layer/last_dense_layer/dense/bias:0 and shape: (8,)
>==================> Number of parameters: 27640
>==================> Trainable variables (scope=Model_Layer)
Name: Model_Layer/RNN_Layer/rnn/lstm_cell/kernel:0 and shape: (182, 16)
Name: Model_Layer/RNN_Layer/rnn/lstm_cell/bias:0 and shape: (16,)
Name: Model_Layer/NN_Layer/dense_layer_0/dense/kernel:0 and shape: (50, 64)
Name: Model_Layer/NN_Layer/dense_layer_0/dense/bias:0 and shape: (64,)
Name: Model_Layer/NN_Layer/dense_layer_1/dense/kernel:0 and shape: (64, 64)
Name: Model_Layer/NN_Layer/dense_layer_1/dense/bias:0 and shape: (64,)
Name: Model_Layer/NN_Layer/last_dense_layer/dense/kernel:0 and shape: (64, 1)
Name: Model_Layer/NN_Layer/last_dense_layer/dense/bias:0 and shape: (1,)
>==================> Number of parameters: 10417
2021-08-04 14:08:27.905981: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2021-08-04 14:08:28.000457: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-04 14:08:28.000757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 7.90GiB freeMemory: 7.62GiB
2021-08-04 14:08:28.000771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2021-08-04 14:08:28.175339: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-08-04 14:08:28.175365: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2021-08-04 14:08:28.175371: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2021-08-04 14:08:28.175447: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7353 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
>==================> Random initialization
>==================> Start Training Unconditional Loss...
2021-08-04 14:08:36.909623: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2021-08-04 14:08:37.307746: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2021-08-04 14:08:38.566491: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2021-08-04 14:08:38.679219: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2021-08-04 14:08:44.835743: W tensorflow/core/framework/allocator.cc:122] Allocation of 394183200 exceeds 10% of system memory.
2021-08-04 14:08:45.618777: W tensorflow/core/framework/allocator.cc:122] Allocation of 394183200 exceeds 10% of system memory.
2021-08-04 14:08:47.160881: W tensorflow/core/framework/allocator.cc:122] Allocation of 394183200 exceeds 10% of system memory.



>==================> Doing epoch 0
>==================> Epoch 0 train/valid/test loss: 0.0107/0.0563/0.0668
>==================> Epoch 0 train/valid/test loss (residual): 0.9731/0.9880/0.9839
>==================> Epoch 0 train/valid/test sharpe: 0.1336/-0.0195/0.0070
>==================> Epoch 0 Elapse/Estimate: 15.17s/3883.89s
2021-08-04 14:08:49.567703: W tensorflow/core/framework/allocator.cc:122] Allocation of 394183200 exceeds 10% of system memory.
2021-08-04 14:08:49.927846: W tensorflow/core/framework/allocator.cc:122] Allocation of 394183200 exceeds 10% of system memory.



>==================> Doing epoch 128
>==================> Epoch 128 train/valid/test loss: 0.0000/0.0015/0.0037
>==================> Epoch 128 train/valid/test loss (residual): 0.9978/0.9979/0.9972
>==================> Epoch 128 train/valid/test sharpe: 0.7911/0.4838/0.2237
>==================> Saving current best checkpoint (sharpe)
>==================> Epoch 128 Elapse/Estimate: 446.41s/885.89s
>==================> Training Unconditional Loss Finished!

>==================> Start Updating Moment Conditions...
>==================> Restored checkpoint
>==================> Saving current best checkpoint (epoch 0)
>==================> Saving current best checkpoint (epoch 0)
>==================> Saving current best checkpoint (epoch 0)
>==================> Saving current best checkpoint (epoch 0)
>==================> Updating Moment Conditions Finished!

>==================> Start Training Conditional Loss...
>==================> Restored checkpoint
2021-08-04 14:24:52.812283: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2021-08-04 14:24:53.215805: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2021-08-04 14:24:54.498481: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2021-08-04 14:24:54.611871: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:666] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.



>==================> Doint epoch 0
>==================> Epoch 0 train/valid/test loss: 0.0077/0.0238/0.0341
>==================> Epoch 0 train/valid/test loss (residual): 0.9820/0.9909/0.9862
>==================> Epoch 0 train/valid/test sharpe: 0.1681/0.1068/-0.0487
>==================> Epoch 0 Elapse/Estimate: 10.77s/11028.96s



>==================> Doint epoch 128
>==================> Epoch 128 train/valid/test loss: 0.0002/0.0025/0.0016
>==================> Epoch 128 train/valid/test loss (residual): 0.9946/0.9969/0.9975
>==================> Epoch 128 train/valid/test sharpe: 1.3782/0.6319/0.3471
>==================> Epoch 128 Elapse/Estimate: 396.99s/3151.31s



>==================> Doint epoch 256
>==================> Epoch 256 train/valid/test loss: 0.0000/0.0015/0.0017
>==================> Epoch 256 train/valid/test loss (residual): 0.9945/0.9973/0.9976
>==================> Epoch 256 train/valid/test sharpe: 1.9077/0.6929/0.3389
>==================> Epoch 256 Elapse/Estimate: 781.59s/3114.21s



>==================> Doint epoch 384
>==================> Epoch 384 train/valid/test loss: 0.0000/0.0013/0.0014
>==================> Epoch 384 train/valid/test loss (residual): 0.9940/0.9973/0.9977
>==================> Epoch 384 train/valid/test sharpe: 2.1603/0.7147/0.3898
>==================> Epoch 384 Elapse/Estimate: 1177.38s/3131.52s



>==================> Doint epoch 512
>==================> Epoch 512 train/valid/test loss: 0.0001/0.0011/0.0011
>==================> Epoch 512 train/valid/test loss (residual): 0.9929/0.9973/0.9976
>==================> Epoch 512 train/valid/test sharpe: 2.2169/0.7592/0.4253
>==================> Epoch 512 Elapse/Estimate: 1577.60s/3149.05s



>==================> Doint epoch 640
>==================> Epoch 640 train/valid/test loss: 0.0000/0.0007/0.0016
>==================> Epoch 640 train/valid/test loss (residual): 0.9929/0.9975/0.9974
>==================> Epoch 640 train/valid/test sharpe: 2.7569/0.8125/0.3768
>==================> Epoch 640 Elapse/Estimate: 1980.54s/3163.93s



>==================> Doint epoch 768
>==================> Epoch 768 train/valid/test loss: 0.0000/0.0007/0.0011
>==================> Epoch 768 train/valid/test loss (residual): 0.9920/0.9975/0.9975
>==================> Epoch 768 train/valid/test sharpe: 2.7568/0.8660/0.4560
>==================> Epoch 768 Elapse/Estimate: 2413.27s/3213.51s



>==================> Doint epoch 896
>==================> Epoch 896 train/valid/test loss: 0.0000/0.0007/0.0018
>==================> Epoch 896 train/valid/test loss (residual): 0.9920/0.9975/0.9970
>==================> Epoch 896 train/valid/test sharpe: 3.1184/0.8764/0.3672
>==================> Epoch 896 Elapse/Estimate: 2834.07s/3235.33s
>==================> Training Conditional Loss Finished!

>==================> Saving last checkpoint
>==================> SDF Portfolio Sharpe Ratio (Evaluated on Sharpe): Train 2.920      Valid 1.013     Test 0.481
```
