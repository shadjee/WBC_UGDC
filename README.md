TITLE
Uncertainty-Guided Deep Ensemble (UGDE) with Conformal Prediction for White Blood Cell (WBC) Classification

Description
This project implements a Deep Ensemble of CNN models (MobileNetV3, EfficientNet-B0, ShuffleNetV2) integrated with Monte Carlo Dropout and Conformal Prediction (ICP) to achieve accurate, reliable, and interpretable white blood cell subtype classification.  
The framework provides uncertainty quantification, calibrated predictions, and Grad-CAM-based visual explanations, enabling trustworthy AI deployment in clinical hematology diagnostics.


Dataset Information
The model was trained on the Bodzás et al. (2023) high-resolution WBC dataset comprising 16,027 single-cell images from 78 patients, including both normal and leukemic samples across nine classes:
- Basophils  
- Eosinophils  
- Lymphoblasts  
- Lymphocytes  
- Monocytes  
- Myeloblasts  
- Neutrophil (Band)  
- Neutrophil (Segmented)  
- Normoblasts  

Images were stained using May-Grünwald and Giemsa protocols. To ensure class balance, each class was augmented to ~3,000 samples, resulting in 27,000 total images.

An external validation was performed using the Raabin-WBC dataset (14,000+ annotated cells) to evaluate generalization.


Code Overview
The pipeline integrates:
1. Data Preprocessing & Augmentation  
   - Random crops, flips, affine transforms, and normalization.
2. Model Training (Deep Ensemble)  
   - Trains MobileNetV3-Large, EfficientNet-B0, and ShuffleNetV2 independently.  
   - Uses Adam optimizer, learning rate = 1e-3, batch size = 128, 10 epochs.  
   - Tracks loss, accuracy, and calibration performance.
3. Uncertainty Estimation
   - Monte Carlo Dropout (20 stochastic forward passes) during inference for predictive variance and entropy estimation.
4. Inductive Conformal Prediction (ICP)
   - Splits dataset into train / calibration / test (80 % / 10 % / 10 %).  
   - Computes non-conformity scores and coverage at α = 0.05, 0.10, 0.20.
5. Visualization & Evaluation
   - Confusion matrices  
   - Calibration curves  
   - Grad-CAM interpretability maps  
   - Risk–coverage analysis  

All analyses are implemented in PyTorch, with plotting via Matplotlib + Seaborn.

Usage Instructions
1. Dataset Preparation  
   - Organize the dataset directory as:  
     ```
     /path/to/dataset/
        ├── Basophil/
        ├── Eosinophil/
        ├── Lymphoblast/
        ├── Lymphocyte/
        ├── Monocyte/
        ├── Myeloblast/
        ├── Neutrophil_Band/
        ├── Neutrophil_Segmented/
        └── Normoblast/
     
   - Update the path in the code:
     python
     data_dir = "/path/to/your/WBC_dataset"
     ```
2. Run Training and Evaluation
   - Execute the Python script or Jupyter notebook:
     bash
     python UGDE_WBC_Conformal_Prediction.py
     
     or open and run `UGDE_WBC_Conformal_Prediction.ipynb` in Jupyter/Colab.
3. Outputs Generated
   - Training/validation accuracy and loss plots  
   - Classification reports per model  
   - Ensemble confusion matrix and calibration plots  
   - Conformal prediction coverage results  
   - Grad-CAM interpretability heatmaps


Requirements
| Library | Version (Recommended) |
|----------|-----------------------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 |
| torchvision | ≥ 0.16 |
| numpy | ≥ 1.25 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.12 |
| efficientnet-pytorch | ≥ 0.7 |

Install dependencies:
bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn efficientnet-pytorch

Methodology Summary
| Component | Description |
|------------|-------------|
| Base Models | MobileNetV3-Large, EfficientNet-B0, ShuffleNetV2 |
| Training Strategy | Independent training + ensemble averaging |
| Uncertainty Modeling | Monte Carlo Dropout during inference |
| Confidence Calibration | Temperature scaling + Expected Calibration Error (ECE) |
| Conformal Prediction | Inductive CP with α = 0.05, 0.10, 0.20 |
| Interpretability | Grad-CAM heatmaps highlighting discriminative morphological regions |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score, ECE, Coverage, Avg. Set Size |


Performance Summary
| Model | Accuracy | Macro F1 | Notes |
|--------|-----------|----------|-------|
| MobileNetV3 | 0.98 | 0.98 | Stable convergence |
| EfficientNet-B0 | 0.98 | 0.98 | Smooth generalization |
| ShuffleNetV2 | 0.99 | 0.99 | Slightly superior on rare classes |
| UGDE Ensemble | 0.99 | 0.99 | Best overall performance, well-calibrated |

- ECE (after calibration): ≈ 0.008  
- Coverage (α = 0.05): 94 % – 95 %  
- External accuracy (Raabin-WBC): 94 %  


 Visualization Examples
- Confusion Matrices – class-wise accuracy comparison  
- Calibration Curves – alignment between confidence and accuracy  
- Grad-CAM Heatmaps – highlight nuclear/cytoplasmic focus regions  
- Risk–Coverage Curves – trade-off between reliability and coverage  

Evaluation Environment
Developed and tested on:
- OS: macOS / Windows 10  
- CPU/GPU: Intel Core i7 / NVIDIA Quadro P4000  
- RAM: 16 GB  
- Framework: PyTorch  

 Results and Discussion
- UGDE achieved near-perfect performance with biologically consistent Grad-CAM attention maps.  
- Monte Carlo dropout uncertainty aligned with ambiguous morphology, ensuring safe deployment.  
- Conformal prediction added statistical reliability without increasing set size.  
- External validation confirmed strong generalization across staining and device variations.

Limitations
- Current taxonomy covers nine WBC subtypes; expansion is needed for rare cells (e.g., plasma or promyelocytes).  
- Ensemble + MC dropout increases inference cost (~150 forward passes per image).  
- Deployment optimization and real-time inference remain future directions.

Conclusion
The proposed Uncertainty-Guided Deep Ensemble (UGDE) delivers:
- State-of-the-art accuracy (~99 %)  
- Meaningful uncertainty quantification (MC Dropout + Conformal Prediction)  
- Clinically interpretable Grad-CAM explanations

This framework represents a practical, trustworthy AI pipeline for hematology diagnostics, emphasizing accuracy, reliability, and transparency.
