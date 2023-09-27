# SPI - Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference

Code for the paper **Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference**. [[PDF]](https://arxiv.org/pdf/2306.01153.pdf)

The paper will be presented at ICML 2023. This code has been written using PyTorch. If you use source codes in this repository in your work, please cite the following papers:

<pre>
@inproceedings{pmlr-v202-xu23j,
	author = {Xu, Yan and Kong, Deqian and Xu, Dehong and Ji, Ziwei and Pang, Bo and Fung, Pascale and Wu, Ying Nian},
	booktitle = {Proceedings of the 40th International Conference on Machine Learning},
	editor = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
	month = {23--29 Jul},
	pages = {38518--38534},
	pdf = {https://proceedings.mlr.press/v202/xu23j/xu23j.pdf},
	publisher = {PMLR},
	series = {Proceedings of Machine Learning Research},
	title = {Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference},
	url = {https://proceedings.mlr.press/v202/xu23j.html},
	volume = {202},
	year = {2023},
	bdsk-url-1 = {https://proceedings.mlr.press/v202/xu23j.html}
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

1. Download datasets from the official links.

2. Put the downloaded datasets under `data` folder, named as `wizard_of_wikipedia` and `holle`, respectively.

3. Preprocess data by unwraping dialogues and converting the dialogues into data samples. Run the following command 
line under the main folder:

```console
python src/data_utils/wow_proc.py --preproc_dir data/processed_wow
python src/data_utils/holle_proc.py --preproc_dir data/processed_holle
```

4. The processed data will be stored under the `preproc_dir`. Do not modify the above path, or you will have to modify 
the hard-coded path in `src/data_utils/wizard_of_wikipedia.py` and `src/data_utils/holle.py`.

**Alternatively**, you can also download the pre-processed data directly from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/EuM6RFbNnyZOiLRyp_SIEtsBRAWq85TI2WaZywJWPGTYHw?e=rxbicW).


# Training

For reproducibility, you can access our pre-trained weights from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/Euyhj33uFtdLnBcWe4bHBukB4rjbXSaoWRbG2PZ6Mcdt9Q?e=WCs3ap). You can also train the models yourself:

```console
sh run_spi.sh
```

In `run_spi.sh`, the command line for training four different models is provided. Please use them based on your needs.


# Prediction

Given one checkpoint, we can evaluate our model with the following command line. The script will 1) compute the perplexity 
of generating the gold responses and 2）generate responses given the data samples in the test set.

```console
sh predict_spi.sh
```

# Coming Soon
Our code without cleansing and pre-training model weights are available [here](https://drive.google.com/drive/folders/1FVRA01uPUVdJ5rzN_mFcwpZAznr8XDiX?usp=share_link) tentatively.
