<p align="center">
  <img src="img/logo.png" alt="Project Logo" width="260" />
</p>

<h1 align="center">🚀 LIMU-BERT: A Foundation Model Pretrained with Nationwide Data</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8%2B-blue"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green"></a>
  <a href="https://github.com/WANDS-HKUST/LIMU-BERT_Experience">
  <img src="https://img.shields.io/github/stars/WANDS-HKUST/LIMU-BERT_Experience?style=social">
</a>
</p>

---

## 📌 Introduction

**LIMU-BERT_Experience** presents the first real-world, nationwide deployment of a **human activity recognition (HAR)** foundation model in the on-demand food delivery industry. Built on the LIMU-BERT sensor foundation model, this project demonstrates how large-scale IMU data, self-supervised learning, and lightweight on-device inference can transform operational decision-making at massive scale.

Deployed with Ele.me over two years, LIMU-BERT now supports **500,000 couriers across 367 cities**, making **7.5 billion on-device predictions per day**. Leveraging **858M+ unlabeled IMU samples** for pretraining and minimal labeled data, the system achieves **over 90% activity recognition accuracy** nationwide and powers multiple business-critical applications:

- 🚴 **Trajectory segmentation** for detecting riding, walking, and key delivery events  
- ⬆️ **Elevation change detection** using IMU-only modeling  
- ⏱️ **Improved ETA/ETS prediction**, reducing time estimation errors across millions of orders  
- 💰 **Dynamic and fair pricing**, contributing to **0.44 billion RMB annual cost savings**

This repository open-sources the pretrained LIMU-BERT model—trained on **1.43 million hours** of sensor data from 60K couriers and 1.1K phone models—providing a strong foundation for future research in mobile sensing and ubiquitous computing.

This repository builds upon our earlier open-source implementation **[LIMU-BERT-Public](https://github.com/dapowan/LIMU-BERT-Public)**, which contains the official source code of our paper *LIMU-BERT*, published at **ACM SenSys 2021** and awarded **Best Paper Runner-Up**. LIMU-BERT_Experience extends this foundation with large-scale real-world deployment, additional datasets, and industry-level model optimization.


---

## 📦 Installation

```bash
git clone https://github.com/WANDS-HKUST/LIMU-BERT_Experience.git
cd LIMU-BERT_Experience
pip install -r requirements.txt

