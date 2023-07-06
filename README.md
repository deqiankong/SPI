# SPI - Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference

Code for the paper **Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference**. [[PDF]](https://arxiv.org/pdf/2306.01153.pdf)

The paper will be presented at ICML 2023. This code has been written using PyTorch. If you use source codes in this repository in your work, please cite the following papers:

<pre>
@article{xu2023diverse,
  title={Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference},
  author={Xu, Yan and Kong, Deqian and Xu, Dehong and Ji, Ziwei and Pang, Bo and Fung, Pascale and Wu, Ying Nian},
  journal={arXiv preprint arXiv:2306.01153},
  year={2023}
}
</pre>

# Environment

Install the environment with the following command lines:

```console
conda env create -f dial.yml python=3.10
conda activate dial-env
```

# Data

To run both training and prediction of SPI, you need to prepare the data for experiments.

1. Download datasets from the official links or [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/EuM6RFbNnyZOiLRyp_SIEtsBRAWq85TI2WaZywJWPGTYHw?e=rxbicW).

2. Put the downloaded datasets under `data` folder, named as `wizard_of_wikipedia` and `holle`, respectively.

3. Preprocess data by unwraping dialogues and converting the dialogues into data samples. Run the following command 
line under the main folder:

```console
python src/data_utils/wow_proc.py --preproc_dir data/processed_wow
python src/data_utils/holle_proc.py --preproc_dir data/processed_holle
```

4. The processed data will be stored under the `preproc_dir`. Do not modify the above path, or you will have to modify 
the hard-coded path in `src/data_utils/wizard_of_wikipedia.py` and `src/data_utils/holle.py`.


# Training

For reproducibility, you can access our pre-trained weights from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/Euyhj33uFtdLnBcWe4bHBukB4rjbXSaoWRbG2PZ6Mcdt9Q?e=WCs3ap). You can also train the models yourself:

```console
sh run_spi.sh
```

In `run_spi.sh`, the command line for training four different models are provided. Please use them based on your need.


# Prediction

Given one checkpoint, we can evaluate our model with the following command line. The script will compute the perplexity 
of generating the gold responses and generate responses given the data samples in the test set.

```console
sh predict_spi.sh
```

# Coming Soon
Our code without cleansing and pre-training model weights are available [here](https://drive.google.com/drive/folders/1FVRA01uPUVdJ5rzN_mFcwpZAznr8XDiX?usp=share_link) tentatively.
