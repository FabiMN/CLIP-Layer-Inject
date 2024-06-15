# CLIP-Layer-Inject: injecting arbitrary models into CLIP
CLIP-Layer-Inject allows you to inject arbitrary models into linear layers in the MLP and Multihead Attention modules in
CLIP's visual and textual branches. 
CLIP-Layer-Inject was developed as part of my Masters thesis.

## Installation
```bash
git clone https://github.com/FabiMN/CLIP-Layer-Inject.git
cd CLIP-Layer-Inject

conda create -n clip_layer_inject python=3.8
conda activate clip_layer_inject

pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit
```

## Running the code
An example of how to inject various models into various layers is given in main.py.

To run main.py using the Caltech101 dataset, run the following command:
```bash
python main.py --config configs/caltech101.yaml
```

## Acknowledgement
This repository is build on top of prior work done by [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter),  [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter).

