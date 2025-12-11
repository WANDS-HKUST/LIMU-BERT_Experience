<p align="center">
  <img src="img/logo.png" alt="Project Logo" width="260" />
</p>

<h1 align="center">🚀 LIMU-BERT-X: A Large-Scale Real-World IMU Sensor Foundation Model</h1>

<p align="center">
  <a href="#" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  </a>
  <a href="#" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </a>
  <a href="https://github.com/WANDS-HKUST/LIMU-BERT_Experience" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/stars/WANDS-HKUST/LIMU-BERT_Experience?style=social&cacheSeconds=3600" alt="GitHub stars">
  </a>
  <a href="https://dl.acm.org/doi/10.1145/3680207.3765261" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Paper-MobiCom%202025-ff69b4?logo=academia&logoColor=white" alt="MobiCom 2025 paper">
  </a>
</p>



---

## 📌 Introduction

**LIMU-BERT-X** presents the first real-world, nationwide deployment of a **human activity recognition (HAR)** foundation model in the on-demand food delivery industry. Built on the LIMU-BERT sensor foundation model, this project demonstrates how large-scale IMU data, self-supervised learning, and lightweight on-device inference can transform operational decision-making at massive scale.

Deployed with Ele.me over two years, LIMU-BERT-X now supports **500,000 couriers across 367 cities**, making **7.5 billion on-device predictions per day**. Leveraging **1.43 million hours of real-world sensor data** for pretraining and minimal labeled data, the system achieves **over 90% activity recognition accuracy** nationwide and powers multiple business-critical applications:

- 🚴 **Trajectory segmentation** for detecting riding, walking, and key delivery events  
- ⬆️ **Elevation change detection** using IMU-only modeling  
- ⏱️ **Improved ETA/ETS prediction**, reducing time estimation errors across millions of orders  
- 💰 **Dynamic and fair pricing**, contributing to **0.44 billion RMB annual cost savings**

This project has been accepted to 📄 **[MobiCom 2025 Experience Paper](https://dl.acm.org/doi/10.1145/3680207.3765261)** and we open-source the pretrained LIMU-BERT model—trained on **1.43 million hours** of sensor data from 60K couriers and 1.1K phone models—providing a strong foundation for future research in mobile sensing and ubiquitous computing.

This repository builds upon our earlier open-source implementation **[LIMU-BERT-Public](https://github.com/dapowan/LIMU-BERT-Public)**, which contains the official source code of our paper *[LIMU-BERT](https://dl.acm.org/doi/abs/10.1145/3485730.3485937)*, published at 📄 **ACM SenSys 2021** and awarded 🏆 **Best Paper Runner-Up**. LIMU-BERT-X extends this foundation with large-scale real-world deployment, additional datasets, and industry-level model optimization.


---

## 📦 Installation

```bash
git clone https://github.com/WANDS-HKUST/LIMU-BERT_Experience.git
cd LIMU-BERT_Experience
pip install -r requirements.txt
```

## 📘 Instructions

Before diving into this repository, we recommend readers have a basic understanding of **self-supervised learning** and the overall design philosophy behind LIMU-BERT. For background and foundational concepts, you may refer to:

- 📄 **[Our original LIMU-BERT paper (SenSys 2021)](https://dl.acm.org/doi/abs/10.1145/3485730.3485937)** — introduces the core self-supervised learning framework for IMU sensing  
- 📄 **[Our latest experience paper (MobiCom 2025)](https://dl.acm.org/doi/10.1145/3680207.3765261)** — details the large-scale real-world deployment and commercial adoption of the model

### Training Workflow

In our framework, the training pipeline consists of **two phases**:

- **1. Self-Supervised Training Phase**:
Train **LIMU-BERT** using large-scale **unlabeled IMU data** with [`pretrain.py`](./pretrain.py). This phase learns generalizable IMU representations using contrastive and predictive objectives.

- **2. Supervised Training Phase**: Train a downstream **classifier** on labeled IMU data using the pretrained model with [`bert_classifier.py`](./bert_classifier.py). This phase adapts LIMU-BERT to specific tasks such as activity recognition.

As we have already pretrained **LIMU-BERT-X** using extensive unlabeled IMU data, the pretrained weights are provided in  
[`limu_bert_x.pt`](./weights/limu_bert_x.pt). You may **skip the self-supervised training phase** and directly use these pretrained weights for downstream tasks. Alternatively, you can **fine-tune LIMU-BERT-X on your own dataset** to better handle domain shift.

### Self-Supervised Training Phase (Optional)
Run [`pretrain.py`](./pretrain.py). Example:

```bash
python pretrain.py v4 hhar 20_120 -s limu_v1 -f limu_bert_x
```

This command fine-tunes LIMU-BERT-X using the configuration defined in the the _based_v4_ of [limu_bert.json](./config/limu_bert.json), with the hhar dataset "data_20_120.npy" and "label_20_120.npy". The trained model will be saved as "limu_v1.pt" in the _saved/pretrain_base_hhar_20_120_ folder. The mask and train settings are defined in the [mask.json](./config/mask.json) and [pretrain.json](./config/pretrain.json), respectively.

Note that **LIMU-BERT-X** uses a sequence length of **20**, which is slightly different from the original **LIMU-BERT**, which uses a sequence length of **120**.

---

## 📚 Citation

If you find **LIMU-BERT-X** or **LIMU-BERT** useful in your research, please cite our papers:

```bibtex
@inproceedings{xu2025experience,
  title={Experience Paper: Adopting Activity Recognition in On-demand Food Delivery Business},
  author={Xu, Huatao and Zhang, Yan and Gao, Wei and Shen, Guobin and Li, Mo},
  booktitle={Proceedings of the 31st Annual International Conference on Mobile Computing and Networking},
  pages={1015--1028},
  year={2025}
}

@inproceedings{xu2021limu,
  title={Limu-bert: Unleashing the potential of unlabeled data for imu sensing applications},
  author={Xu, Huatao and Zhou, Pengfei and Tan, Rui and Li, Mo and Shen, Guobin},
  booktitle={Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems},
  pages={220--233},
  year={2021}
}
```

We also encourage you to refer to and cite our closely related 📄 **MobiCom 2023 paper [UniHAR](https://dl.acm.org/doi/abs/10.1145/3570361.3613299)**.

```bibtex
@inproceedings{xu2023practically,
  title={Practically adopting human activity recognition},
  author={Xu, Huatao and Zhou, Pengfei and Tan, Rui and Li, Mo},
  booktitle={Proceedings of the 29th Annual International Conference on Mobile Computing and Networking},
  pages={1--15},
  year={2023}
}
```

## 📬 Contact

If you have questions, suggestions, or collaboration interests, feel free to reach out:

- **Huatao Xu** — 735820057@qq.com  
- Or open an issue or pull request in this repository.

We welcome contributions from the community!