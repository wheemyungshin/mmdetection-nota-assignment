# Introduction
This repository is a repository for Nota assignment.
The dataset in this repository is solely used for Nota home assignment. Data should not be used for any other purpose.

This repository contains:
1. A guide for installation of requirements, training and inference.
2. Qualitative results.
3. Explanation (focused on problem definitions and my modification on code).

The code highly depends on the opensource, MMdetection : [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

# A guide for installation of requirements, training and inference
## Overview
You can use [Google Colaboratory](https://colab.research.google.com/?utm_source=scs-index).
The overall script is here: UPDATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

Simply following all the script will install the requirements, start training and inference on the test dataset.
Rest of the section is for script explanation and how to set custom configs.

## Installation
Please refer to [Installation](docs/en/get_started.md/#Installation) for installation instructions.
Follow the steps below according to the "Best Practices" section in the link above:

```
git clone https://github.com/wheemyungshin/mmdetection-nota-assignment.git
pip install -U openmim
mim install mmcv-full
pip install -v -e /content/mmdetection-nota-assignment
```

To fix [a bug during installation](https://github.com/open-mmlab/mmdetection/issues/8227), a new script line is added:
```
sed -i 's/if collection_name:/if collection_name and collection_name in name2collection\.keys():/g' /usr/local/lib/python3.7/dist-packages/mim/commands/search.py
```

##Dataset
The provided dataset contains 1,688 training set and 184 test set.
Annotation format is Pascal VOC and there is no label for test set.
The number of classes is 5.
Size is not uniform. There exists size under ~128 or over 1280~.

After uploading the dataset onto the colab notebook, unzip the dataset.
```
mkdir data
mkdir data/facial_emotion_data
unzip [NOTA]facial_emotion_data.zip -d data/facial_emotion_data
python mmdetection-nota-assignment/converter.py
```

Before training, run "converter.py" to get the list of training set (trainval.txt).
```
python mmdetection-nota-assignment/converter.py
```


## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection. We provide [colab tutorial](demo/MMDet_Tutorial.ipynb) and [instance segmentation colab tutorial](demo/MMDet_InstanceSeg_Tutorial.ipynb), and other tutorials for:

- [with existing dataset](docs/en/1_exist_data_model.md)
- [with new dataset](docs/en/2_new_data_model.md)
- [with existing dataset_new_model](docs/en/3_exist_data_new_model.md)
- [learn about configs](docs/en/tutorials/config.md)
- [customize_datasets](docs/en/tutorials/customize_dataset.md)
- [customize data pipelines](docs/en/tutorials/data_pipeline.md)
- [customize_models](docs/en/tutorials/customize_models.md)
- [customize runtime settings](docs/en/tutorials/customize_runtime.md)
- [customize_losses](docs/en/tutorials/customize_losses.md)
- [finetuning models](docs/en/tutorials/finetune.md)
- [export a model to ONNX](docs/en/tutorials/pytorch2onnx.md)
- [export ONNX to TRT](docs/en/tutorials/onnx2tensorrt.md)
- [weight initialization](docs/en/tutorials/init_cfg.md)
- [how to xxx](docs/en/tutorials/how_to.md)

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).