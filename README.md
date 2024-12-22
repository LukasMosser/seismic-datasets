# Seismic-Datasets
Seismic and geophysical datasets collected from various sources and unified under huggingface datasets.

IMPORTANT: I do not take credit for any of these datasets, they are the products of the original authors and you should cite their corresponding papers. The only reason I have created this is to simplify access to the datasets for future use. I highly encourage not storing datasets on google drives and provide corresponding dataloaders for datasets on zenodo via huggingface.

For all datasets I have tried to reproduce the train/valid/test splits as best as possible from the provided codes. 


## Installation
To install first install uv and then install the environment as following. 
```bash
uv python install
uv venv
```

## Running the conversion scripts

The conversion scripts are not intended to be run by anyone else but rather serve as an example how to create similar datasets and to provide transparency about how the datasets have been created when pushing them to huggingface.

If you do want to run the scripts, make sure to change the base_path and also the target repositories on huggingface. You can then run the scripts like so:
```bash
uv run conversion/convert_crossdomainfoundationmodeladaption_datasets.py
```

## Datasets

The following datasets have been converted:

### Seismic SuperResolution by [JintaoLee-Roger](https://github.com/JintaoLee-Roger)

Seismic Super Resolution from [JintaoLee-Roger/SeismicSuperResolution](https://github.com/JintaoLee-Roger/SeismicSuperResolution) with the data being hosted in the following [Google Drive](https://drive.google.com/drive/folders/1DuMdclOdeXDgGBOhsHSlEdTB_LvhIH-X). The dataset is now available on [huggingface](https://huggingface.co/datasets/porestar/superresolutiondataset) and can be loaded as such:
```
# pip install datasets
from datasets import 
dataset = load_dataset("porestar/superresolutiondataset")
```

#### Citation

Use the following citation if you use the dataset:

J. Li, X. Wu and Z. Hu, "Deep Learning for Simultaneous Seismic Image Super-Resolution and Denoising," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-11, 2022, Art no. 5901611, doi: 10.1109/TGRS.2021.3057857.
BibTex

@article{deep2022li,
   author={Li, Jintao and Wu, Xinming and Hu, Zhanxuan},
   journal={IEEE Transactions on Geoscience and Remote Sensing}, 
   title={Deep Learning for Simultaneous Seismic Image Super-Resolution and Denoising}, 
   year={2022},
   volume={60},
   number={5901611},
   pages={1-11},
   doi={10.1109/TGRS.2021.3057857}}



### Seismic Foundation Model by [shenghanlin](https://github.com/shenghanlin)

Seismic Super Resolution from [shenghanlin/SeismicFoundationModel](https://github.com/shenghanlin/SeismicFoundationModel) with the data being hosted in the following [https://rec.ustc.edu.cn/](https://rec.ustc.edu.cn/share/d6cd54a0-e839-11ee-982a-9748e54ad7a4). The datasets are now available on huggingface:
- Denoising: [porestar/seismicfoundationmodel-denoise](https://huggingface.co/datasets/porestar/seismicfoundationmodel-denoise)
- Denoising Field: [porestar/seismicfoundationmodel-denoise-field](https://huggingface.co/datasets/porestar/seismicfoundationmodel-denoise-field)
- Facies: [porestar/seismicfoundationmodel-facies](https://huggingface.co/datasets/porestar/seismicfoundationmodel-facies)
- Geobody: [porestar/seismicfoundationmodel-geobody](https://huggingface.co/datasets/porestar/seismicfoundationmodel-geobody)
- Interpolation: [porestar/seismicfoundationmodel-interpolation](https://huggingface.co/datasets/porestar/seismicfoundationmodel-interpolation)
- Inversion Synthetic: [porestar/seismicfoundationmodel-inversion-synthetic](https://huggingface.co/datasets/porestar/seismicfoundationmodel-inversion-synthetic)
- Inversion Seam: [porestar/seismicfoundationmodel-inversion-seam](https://huggingface.co/datasets/porestar/seismicfoundationmodel-inversion-seam)


#### Citation

Use the following citation if you use the dataset:

@article{sheng2023seismic,
  title={Seismic Foundation Model (SFM): a new generation deep learning model in geophysics},
  author={Sheng, Hanlin and Wu, Xinming and Si, Xu and Li, Jintao and Zhang, Sibio and Duan, Xudong},
  journal={arXiv preprint arXiv:2309.02791},
  year={2023}
}

### Cross-Domain Foundation Model Adaptation by [ProgrammerZXG](https://github.com/ProgrammerZXG)

Cross-Domain Foundation Model Adaptation from [ProgrammerZXG/Cross-Domain-Foundation-Model-Adaptation](https://github.com/ProgrammerZXG/Cross-Domain-Foundation-Model-Adaptation) with the data being hosted on [Zenodo](https://zenodo.org/records/12798750). 
The datasets are now available on hugginface:

- Crater: [porestar/crossdomainfoundationmodeladaption-crater](https://huggingface.co/datasets/porestar/crossdomainfoundationmodeladaption-crater)
- DAS: [porestar/crossdomainfoundationmodeladaption-das](https://huggingface.co/datasets/porestar/crossdomainfoundationmodeladaption-das)
- Geobody: [porestar/crossdomainfoundationmodeladaption-geobody](https://huggingface.co/datasets/porestar/crossdomainfoundationmodeladaption-geobody)
- Seismic Facies: [porestar/crossdomainfoundationmodeladaption-seismicfacies](https://huggingface.co/datasets/porestar/crossdomainfoundationmodeladaption-seismicfacies)
- Deepfault: [porestar/crossdomainfoundationmodeladaption-deepfault](https://huggingface.co/datasets/porestar/crossdomainfoundationmodeladaption-deepfault)

#### Citation

Use the following citation if you use the dataset:

```
Guo, Z., Wu, X., Liang, L., Sheng, H., Chen, N., & Bi, Z. (2024). Data from "Cross-Domain Foundation Model Adaptation: Pioneering Computer Vision Models for Geophysical Data Analysis" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12798750
```