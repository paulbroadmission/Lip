# Denoising Autoencoder (DAE) 及其應用論文精選

---

## 1. Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P.-A. (2010)
**APA 格式：**
Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P.-A. (2010). Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *Journal of Machine Learning Research, 11*, 3371–3408.

**連結：** [PDF 下載](./DAE01_StackedDenoisingAutoencoders.pdf) | [官方網頁](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

**內容精華要點：**
- 本文提出「堆疊式去噪自編碼器」（Stacked Denoising Autoencoders, SDAE），將多層 DAE 疊加以建構深層神經網路。
- 每層 DAE 以局部去噪準則訓練，能有效學習高階特徵表示。
- 在多個分類任務（如 MNIST、CIFAR-10）上，SDAE 顯著降低錯誤率，超越傳統 autoencoder 及部分深度信念網路（DBN）。
- 證明 SDAE 能自動學習有用且具泛化能力的特徵，對無監督預訓練有重要貢獻。
- 本文為深度學習特徵學習領域的經典之作。

---

## 2. Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P.-A. (2008)
**APA 格式：**
Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th International Conference on Machine Learning* (pp. 1096–1103).

**連結：** [PDF 下載](./DAE02_ExtractingRobustFeatures.pdf) | [官方網頁](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)

**內容精華要點：**
- 本文首次提出 Denoising Autoencoder（DAE）架構，透過將輸入資料加噪聲，訓練網路還原原始資料。
- DAE 能學習到對輸入擾動具魯棒性的特徵，有助於提升下游任務表現。
- 實驗證明 DAE 優於傳統 autoencoder，能更好地捕捉資料分布結構。
- 本文奠定後續深度去噪自編碼器及無監督特徵學習的基礎。
- 內容涵蓋 DAE 訓練流程、理論分析及多項實驗結果。

---

## 3. Sakurada, M., & Yairi, T. (2014)
**APA 格式：**
Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders with nonlinear dimensionality reduction. In *Proceedings of the MLSDA Workshop* (pp. 4:4–4:11).

**連結：** [PDF 下載](./DAE03_AnomalyDetectionAutoencoders.pdf) | [官方網頁](https://www.researchgate.net/publication/268318375_Anomaly_Detection_using_Autoencoders_with_Nonlinear_Dimensionality_Reduction)

**內容精華要點：**
- 本文探討以 autoencoder 進行非線性降維，應用於異常偵測。
- 與 PCA、kernel PCA 等傳統方法比較，autoencoder 能偵測更細微的異常。
- 進一步結合 denoising autoencoder，提升異常偵測的準確度與魯棒性。
- 實驗涵蓋多種資料集，證明方法在工業監控、感測器資料等場景具實用性。
- 內容包含理論分析、方法流程與詳細實驗結果。

---

## 4. Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016)
**APA 格式：**
Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. *ICML 2016 Anomaly Detection Workshop*.

**連結：** [PDF 下載](./DAE04_LSTMEncoderDecoderAnomalyDetection.pdf) | [arXiv 連結](https://arxiv.org/pdf/1607.00148v2.pdf)

**內容精華要點：**
- 本文提出 LSTM Encoder-Decoder 架構，僅用正常資料訓練，利用重建誤差進行異常偵測。
- 適用於可預測與不可預測、週期性與非週期性、多感測器時序資料。
- 在多個公開資料集上驗證其效能，顯示方法具高準確率與泛用性。
- 分析不同參數設定對異常偵測表現的影響。
- 提供完整的理論推導與實驗設計，對時序異常偵測領域有重要貢獻。

---

## 5. Lu, X., Tsao, Y., Matsuda, S., & Hori, C. (2013)
**APA 格式：**
Lu, X., Tsao, Y., Matsuda, S., & Hori, C. (2013). Speech enhancement based on deep denoising autoencoder. In *Proceedings of Interspeech 2013* (pp. 436–440).

**連結：** [PDF 下載](./DAE05_SpeechEnhancement.pdf) | [官方網頁](https://www.isca-archive.org/interspeech_2013/i13_0436.html)

**內容精華要點：**
- 本文將深度 DAE 應用於語音增強，採用 noisy-clean 配對訓練方式。
- 證明隨著網路深度增加，語音增強效果顯著提升。
- 與傳統最小均方誤差（MMSE）語音增強法比較，DAE 方法表現更佳。
- 實驗涵蓋多種噪聲環境，驗證方法的泛用性與穩健性。
- 為深度學習於語音處理領域的重要應用範例。 