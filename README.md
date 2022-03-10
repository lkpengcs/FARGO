# FARGO: A Joint Framework for FAZ and RV Segmentation from OCTA Images

Pytorch implementation for our joint framework for FAZ and RV segmentation from OCTA images. For FAZ segmentation, we use RV segmentation as an auxiliary task and perform ROI extraction. For RV segmentation, we use a coarse-to-fine fashion and employ attention modules.

![Network](https://github.com/lkpengcs/FARGO/blob/main/figs/model.png)

![Module](https://github.com/lkpengcs/FARGO/blob/main/figs/attention_module.png)

## Paper

Please cite our [paper](https://link.springer.com/chapter/10.1007/978-3-030-87000-3_5) if you find the code useful for your research.

```
@inproceedings{peng2021fargo,
  title={Fargo: A joint framework for faz and rv segmentation from octa images},
  author={Peng, Linkai and Lin, Li and Cheng, Pujin and Wang, Zhonghua and Tang, Xiaoying},
  booktitle={International Workshop on Ophthalmic Medical Image Analysis},
  pages={42--51},
  year={2021},
  organization={Springer}
}
```

## Dataset

You can download the **OCTA-500** dataset via this [link](https://ieee-dataport.org/open-access/octa-500).

![Result](https://github.com/lkpengcs/FARGO/blob/main/figs/result.png)

## Usage

### Prerequisite

- Python 3.7+
- Pytorch 1.8+