# EEG Motor Imagery Classification using CSP + LDA (LOO Cross-Validation)

## Project Summary

This project implements a motor imagery EEG classification system using **Common Spatial Patterns (CSP)** for feature extraction and a **cascade of Linear Discriminant Analysis (LDA)** classifiers for multi-class classification. The performance is evaluated using **Leave-One-Out Cross-Validation (LOO)** on all trials. Two training approaches are supported:

- Training a single model on all subjects combined  
- Training a separate model for each subject  

## Dataset Description

The dataset used in this project is a multi-class motor imagery EEG dataset containing brain signals recorded from **15 subjects** performing different motor tasks. Each trial corresponds to one of **four distinct classes**, representing the following motor imagery tasks:

- **Class 1**: Right arm movement 
- **Class 2**: Right thumb movement  
- **Class 3**: Right foot movement  
- **Class 4**: No movement 

Each trial contains EEG data recorded from multiple channels over a fixed duration during the motor imagery task. For more information, see the `Recording.pdf` file.

---

## Technical Details

- **Feature Extraction**: Common Spatial Patterns (CSP)
- **Classifier**: Cascaded One-vs-Other LDA
- **Filter Optimization**: Grid search for CSP filter count per subject
- **Normalization**: Z-score normalization on extracted features
- **Validation Strategy**: Leave-One-Out Cross-Validation (LOO)
- **Evaluation Metrics**:
  - Overall accuracy
  - Per-class accuracy
  - Confusion matrix

---

## Project Structure

### `data_review.mlx`
Live script to visualize and explore the raw EEG signals, including time-series plots and basic statistical inspection.

### `preprocess`
Scripts for preprocessing the raw EEG data:
- Band-pass filtering
- Trial extraction
- Saving structured EEG data per subject/class

Output: Files like `preprocessed_subj_#.mat` will be saved for each subject.

### `grid search`
Scripts for selecting the optimal number of CSP filters per subject using cross-validation.  
**Note:** This step is optional — its results are already embedded in later steps.

### `train and evaluate`
Main training and evaluation scripts:
- `train_model_all_subj.m`: Trains one unified model across all subjects  
- `train_model_per_subj.m`: Trains a separate model for each subject  

Both scripts generate accuracy reports and confusion matrices.

### `models`
Trained model files are saved here:
- `cascade_model_all.mat` for all-subject training
- `cascade_model_subj_#.mat` for per-subject training

### `confusion matrix`
Contains confusion matrix plots (both train and test) for each experiment and model.

---

## How to Run

1. **Download the dataset** and place it in a folder called `dataset/` at the project root.

2. **Preprocess the data**:  
   Run the script inside the `preprocess` folder to filter and extract trials for each subject.

3. (**Optional**) **Run grid search** (from the `grid search/` folder) to estimate the optimal number of CSP filters per subject.

4. Choose one of the following training approaches:

   - To train **one model for all subjects**:
     ```matlab
     train_model_all_subj
     ```

   - To train **a separate model per subject**:
     ```matlab
     train_model_per_subj
     ```

5. Trained models will be saved in the `models/` folder.

6. Confusion matrices will be saved in the `confusion matrix/` folder.

---

## Final Results

Update this section with your final results:

- **Overall Accuracy**: 74.50%
- **Per-Class Accuracy**:
  - Class 1: 62.00%
  - Class 2: 66.00%
  - Class 3: 79.00%
  - Class 4: 91.00%

Confusion matrix of the unified model trained across all subjects:
![model_all_subj](https://github.com/user-attachments/assets/ad9c8e7a-9d28-403d-ad7f-ef0720f33dd4)

---

## Developer

**Amirhossein Chami**  
M.Sc. student in Communication Systems  
University of Tehran  



