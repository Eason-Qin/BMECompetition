# Model Overview
This repository originally forked from the origin implementation of UNETR: Transformers for 3D Medical Image Segmentation [1]. UNETR is the first 3D segmentation network that uses a pure vision transformer as its encoder without relying on CNNs for feature extraction.
This repository also implements the author's network design. **Refer to branch new_network 'networks/TCMix.py'**
The code presents a volumetric (3D) OAR segmentation application using the BTCV challenge dataset for SOTA pretrained weight and dataset competition required.

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Training

Note: input images are in size ```(96, 96, 96)``` which will be converted into non-overlapping patches of size ```(16, 16, 16)```.
The position embedding is performed using a perceptron layer. The ViT encoder follows standard hyper-parameters as introduced in [2].

### Finetuning
State-of-the-art pre-trained checkpoints of UNETR using BTCV dataset are provided by original authors @ MONAI (great work!). 

The weights can be founded at the following directory:

https://drive.google.com/file/d/1kR5QuRAuooYcTNLMnMj80Z9IgSs8jtLO/view?usp=sharing

Once downloaded, please place the checkpoint in the following directory or use ```--pretrained_dir``` to provide the address of where the model is placed:

```./pretrained_models```


## Citation
```
@article{hatamizadeh2021unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Yang, Dong and Roth, Holger and Xu, Daguang},
  journal={arXiv preprint arXiv:2103.10504},
  year={2021}
}
```

## References
[1] Hatamizadeh, Ali, et al. "UNETR: Transformers for 3D Medical Image Segmentation", 2021. https://arxiv.org/abs/2103.10504.

[2] Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
", 2020. https://arxiv.org/abs/2010.11929.