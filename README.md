# PicoDet-Layout-detection

训练参数设置文件： 训练配置文件/picodet_lcnet_x1_0_layout.yml

已训练完成模型下载：https://pan.baidu.com/s/19d1wM_-kzZuVaQ-QStQplQ?pwd=1234 

paddel 模型位于 model_20_optim/

onnx 模型位于 onnx_infer/

待预测图片位于 test_img/

预测结果位于 test_result/

运行run_paddel_infer.bat 进行预测

或者运行命令进行预测：
```python
!python deploy/infer.py --model_dir=model_20_optim/ --image_dir=test_img/ --output_dir=test_result/ --device=GPU --batch_size=1 --cpu_threads=1 --threshold=0.5 
```

## 1. 环境配置

- **（1) 安装PaddlePaddle**

```bash
python -m pip install --upgrade pip

# GPU安装
python -m pip install "paddlepaddle-gpu>=2.3" -i https://mirror.baidu.com/pypi/simple

# CPU安装
python -m pip install "paddlepaddle>=2.3" -i https://mirror.baidu.com/pypi/simple
```

- **（2）安装其他依赖**

```bash
cd PaddleDetection
python3 -m pip install -r requirements.txt
```

## 2.数据准备

**数据分布：**

| File or Folder | Description    |
| :------------- | :------------- |
| `train/`       | 训练集图片     |
| `val/`         | 验证集图片     |
| `test/`        | 测试集图片     |
| `train.json`   | 训练集标注文件 |
| `val.json`     | 验证集标注文件 | 

**标注格式：**

json文件包含所有图像的标注，数据以字典嵌套的方式存放，包含以下key：

- info，表示标注文件info。

- licenses，表示标注文件licenses。

- images，表示标注文件中图像信息列表，每个元素是一张图像的信息。如下为其中一张图像的信息：

  ```
  {
      'file_name': 'PMC4055390_00006.jpg',    # file_name
      'height': 601,                      # image height
      'width': 792,                       # image width
      'id': 341427                        # image id
  }
  ```

- annotations，表示标注文件中目标物体的标注信息列表，每个元素是一个目标物体的标注信息。如下为其中一个目标物体的标注信息：

  ```
  {

      'segmentation':             # 物体的分割标注
      'area': 60518.099043117836, # 物体的区域面积
      'iscrowd': 0,               # iscrowd
      'image_id': 341427,         # image id
      'bbox': [50.58, 490.86, 240.15, 252.16], # bbox [x1,y1,w,h]
      'category_id': 1,           # category_id
      'id': 3322348               # image id
  }
  ```


## 3.模型训练

* 训练参数设置

训练参数设置文件： 训练配置文件/picodet_lcnet_x1_0_layout.yml

```yaml
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/LCNet_x1_0_pretrained.pdparams
weights: model_output/model_final
find_unused_parameters: True
use_ema: true
cycle_epoch: 10
snapshot_epoch: 1
epoch: 20

metric: COCO
# 类别数
num_classes: 4

TrainDataset:
  !COCODataSet
    # 修改为你自己的训练数据目录
    image_dir: train
    # 修改为你自己的训练数据标签文件
    anno_path: train.json
    # 修改为你自己的训练数据根目录
    dataset_dir: /root/publaynet/
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    # 修改为你自己的验证数据目录
    image_dir: val
    # 修改为你自己的验证数据标签文件
    anno_path: val.json
    # 修改为你自己的验证数据根目录
    dataset_dir: /root/publaynet/

TestDataset:
  !ImageFolder
    # 修改为你自己的测试数据标签文件
    anno_path: /root/publaynet/val.json
```

* 开始训练，在训练时，会默认下载PP-PicoDet预训练模型，这里无需预先下载。

```bash
python tools/train.py -c 训练配置文件/picodet_lcnet_x1_0_layout.yml  --eval
```

**注意：**如果训练时显存out memory，将TrainReader中batch_size调小，同时LearningRate中base_lr等比例减小。

## 4. 指标评估

训练中模型参数默认保存在`output/picodet_lcnet_x1_0_layout`目录下。在评估指标时，需要设置`weights`指向保存的参数文件。评估数据集可以通过 `训练配置文件/picodet_lcnet_x1_0_layout.yml`  修改`EvalDataset`中的 `image_dir`、`anno_path`和`dataset_dir` 设置。

```bash
# GPU 评估， weights 为待测权重
python tools/eval.py -c 训练配置文件/picodet_lcnet_x1_0_layout.yml -o weights=./output/picodet_lcnet_x1_0_layout/best_model
```

会输出以下信息，打印出mAP、AP0.5等信息。

```py
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.935
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.979
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.956
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.782
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.969
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.938
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.949
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.818
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.978
[08/15 07:07:09] ppdet.engine INFO: Total sample number: 11245, averge FPS: 24.405059207157436
[08/15 07:07:09] ppdet.engine INFO: Best test bbox ap is 0.935.
```

## 5. 模型导出与预测

### 5.1 模型导出

inference 模型（`paddle.jit.save`保存的模型） 一般是模型训练，把模型结构和模型参数保存在文件中的固化模型，多用于预测部署场景。 训练过程中保存的模型是checkpoints模型，保存的只有模型的参数，多用于恢复训练等。 与checkpoints模型相比，inference 模型会额外保存模型的结构信息，在预测部署、加速推理上性能优越，灵活方便，适合于实际系统集成。

版面分析模型转inference模型步骤如下：

```bash
python tools/export_model.py -c 训练配置文件/picodet_lcnet_x1_0_layout.yml -o weights=output/picodet_lcnet_x1_0_layout/best_model --output_dir=output_inference/
```

* 如无需导出后处理，请指定：`-o export.benchmark=True`（如果-o已出现过，此处删掉-o）
* 如无需导出NMS，请指定：`-o export.nms=False`

转换成功后，在目录下有三个文件：

```
output_inference/picodet_lcnet_x1_0_layout/
    ├── model.pdiparams         # inference模型的参数文件
    ├── model.pdiparams.info    # inference模型的参数信息，可忽略
    └── model.pdmodel           # inference模型的模型结构文件
```

### 5.2 模型推理

单张图片推理：
```bash
python deploy/python/infer.py --model_dir=output_inference/picodet_lcnet_x1_0_layout/ --image_file=docs/images/layout.jpg --device=CPU
```

目录中所有图片推理：
```bash
python deploy/python/infer.py --model_dir=output_inference/picodet_lcnet_x1_0_layout/ --image_dir=testimg/ --device=CPU
```

- --device：指定GPU、CPU设备

模型推理完成，会看到以下log输出

```
------------------------------------------
-----------  Model Configuration -----------
Model Arch: PicoDet
Transform Order:
--transform op: Resize
--transform op: NormalizeImage
--transform op: Permute
--transform op: PadStride
--------------------------------------------
class_id:0, confidence:0.9921, left_top:[20.18,35.66],right_bottom:[341.58,600.99]
class_id:0, confidence:0.9914, left_top:[19.77,611.42],right_bottom:[341.48,901.82]
class_id:0, confidence:0.9904, left_top:[369.36,375.10],right_bottom:[691.29,600.59]
class_id:0, confidence:0.9835, left_top:[369.60,608.60],right_bottom:[691.38,736.72]
class_id:0, confidence:0.9830, left_top:[369.58,805.38],right_bottom:[690.97,901.80]
class_id:0, confidence:0.9716, left_top:[383.68,271.44],right_bottom:[688.93,335.39]
class_id:0, confidence:0.9452, left_top:[370.82,34.48],right_bottom:[688.10,63.54]
class_id:1, confidence:0.8712, left_top:[370.84,771.03],right_bottom:[519.30,789.13]
class_id:3, confidence:0.9856, left_top:[371.28,67.85],right_bottom:[685.73,267.72]
save result to: output/layout.jpg
Test iter 0
------------------ Inference Time Info ----------------------
total_time(ms): 2196.0, img_num: 1
average latency time(ms): 2196.00, QPS: 0.455373
preprocess_time(ms): 2172.50, inference_time(ms): 11.90, postprocess_time(ms): 11.60
```

- Model：模型结构
- Transform Order：预处理操作
- class_id、confidence、left_top、right_bottom：分别表示类别id、置信度、左上角坐标、右下角坐标
- save result to：可视化版面分析结果保存路径，默认保存到`./output`文件夹
- Inference Time Info：推理时间，其中preprocess_time表示预处理耗时，inference_time表示模型预测耗时，postprocess_time表示后处理耗时
