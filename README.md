# HaMeR: Hand Mesh Recovery
* This is a fork of the [HaMeR](https://github.com/geopavlakos/hamer) and mainly used for utilizing the 3D info (pose and mesh) for our unique purpose tracking.

## Recomend installtion procedure
```bash
conda create -n LFD_10 python=3.10 -y
conda activare LFD_10

### ========== OUTSIDE OF Hamer repo ========== ###
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install mamba -c conda-forge
mamba install -c conda-forge opencv
pip install opencv-python opencv-python-headless

### ========== INSIDE OF VISION_MODULE FOLDER ========== ###
# git clone --recursive git@github.com:geopavlakos/hamer.git
git clone --recursive git@github.com:ClaireLC/hamer.git

cd hamer

pip install -e .[all]
pip install -v -e third-party/ViTPose


### ========== DOWNLOAD models and datas for Hamer ========== ###
bash fetch_demo_data.sh

# [HaMeR] checkpoints
gdown 17d50j7z9V_1RbmkpOLbznBwPt34C39KG # require sign up on the MANO webpage
unzip mano_v1_2.zip -d ./unziped
mv ./unziped/mano_v1_2/models/*.pkl _DATA/data/mano/
rm -r unziped
# rm -r mano_v1_2

# [Detectron2] checkpoints
mkdir -p ./_DATA/detectron_ckpts/
cd ./_DATA/detectron_ckpts/
wget https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
cd ../..

### ========== Demo run ========== ###
mkdir demo_out
python demo.py --img_folder example_data --out_folder demo_out --batch_size=48 --side_view --save_mesh --full_frame
```





# Origional README starts here
Code repository for the paper:
**Reconstructing Hands in 3D with Transformers**

[Georgios Pavlakos](https://geopavlakos.github.io/), [Dandan Shan](https://ddshan.github.io/), [Ilija Radosavovic](https://people.eecs.berkeley.edu/~ilija/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/), [David Fouhey](https://cs.nyu.edu/~fouhey/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-2305.20091-00ff00.svg)](https://arxiv.org/pdf/2312.05251.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://geopavlakos.github.io/hamer/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rQbQzegFWGVOm1n1d-S6koOWDo7F2ucu?usp=sharing)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/geopavlakos/HaMeR)

![teaser](assets/teaser.jpg)

## Installation
First you need to clone the repo:
```
git clone --recursive git@github.com:geopavlakos/hamer.git
cd hamer
```

We recommend creating a virtual environment for HaMeR. You can use venv:
```bash
python3.10 -m venv .hamer
source .hamer/bin/activate
```

or alternatively conda:
```bash
conda create --name hamer python=3.10
conda activate hamer
```

Then, you can install the rest of the dependencies. This is for CUDA 11.7, but you can adapt accordingly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

You also need to download the trained models:
```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

## Demo
```bash
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

## Training
First, download the training data to `./hamer_training_data/` by running:
```
bash fetch_training_data.sh
```

Then you can start training using the following command:
```
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to `./logs/`.

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SLAHMR](https://github.com/vye16/slahmr)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Additionally, we thank [StabilityAI](https://stability.ai/) for a generous compute grant that enabled this work.

## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{pavlakos2023reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={arxiv},
    year={2023}
}
```
