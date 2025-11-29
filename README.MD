# Aerial Object Classification & Detection

This project aims to develop a deep learning-based solution that can classify aerial images into two categories — Bird or Drone — and optionally perform object detection to locate and label these objects in real-world scenes.

The solution is critical for various applications, including: 
*   **Security Surveillance**: Identifying unauthorized drones in restricted airspace.
*   **Wildlife Protection**: Detecting birds near sensitive areas like wind farms or airports to prevent accidents.
*   **Airspace Safety**: Ensuring safe air travel by distinguishing between birds and drones.

This initiative involves building a custom CNN classification model, leveraging transfer learning, and optionally implementing YOLOv8 for real-time object detection. The final solution will be deployed using Streamlit for interactive use.

## Notebook Outputs
[Drive Output Link](https://drive.google.com/drive/folders/1KIvW7pUQMRG9M56_pe0ZydGwE7jcLYbm?usp=sharing)

## Streamlit app
[Link to Streamlit app](https://arial-object-detection-and-classification-shakti.streamlit.app/)

## Approach

This project adopted a structured deep learning approach to address the aerial object classification and detection problem. The methodology involved several key stages:

1.  **Understand the Dataset**
    *   Inspect dataset folder structure
    *   Check number of images per class
    *   Identify class imbalance
    *   Visualize sample images

2.  **Data Preprocessing**
    *   Normalize pixel values to [0, 1]
    *   Resize images to a fixed size (224×224 for classification)

3.  **Data Augmentation**
    *   Apply transformations: rotation, flipping, zoom, brightness, cropping

4.  **Model Building (Classification)**
    *   Custom CNN: Conv layers, pooling, dropout, batch normalization, dense output layer
    *   Transfer Learning: Load models like ResNet50, MobileNet, EfficientNetB0 and fine-tune

5.  **Training & Evaluation**
    *   Set up callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
    *   Train models (with and without Mixup)
    *   Evaluate using classification report, confusion matrix, ROC, PR curves

6.  **Object Detection (Optional)**
    *   Prepare dataset for YOLOv8 (generate bounding box annotations)
    *   Train YOLOv8 model
    *   Perform inference and visualize detections

7.  **Model Interpretability**
    *   Implement Grad-CAM to visualize model decisions

8.  **Deployment (Streamlit)**
    *   Develop a Streamlit application for interactive inference


## Tech Stack Used

### Programming Languages
* Python

### Deep Learning Frameworks
* TensorFlow/Keras
* Ultralytics (for YOLOv8)

### Key Libraries
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* PIL (Pillow)
* OpenCV (cv2)

### Deployment
* Streamlit


## Key Findings

This section summarizes the key findings from the evaluation of the initial classification model, the Mixup-trained classification model, and the YOLOv8 object detection model.

### 1. Initial Classification Model (EfficientNetB0)

*   **Overall Accuracy:** The model achieved a high overall accuracy of **0.98** on the test set.
*   **Class-wise Performance:** Both 'bird' and 'drone' classes demonstrated strong and balanced performance:
    *   **Bird Class:** Precision = 0.98, Recall = 0.98, F1-score = 0.98
    *   **Drone Class:** Precision = 0.98, Recall = 0.97, F1-score = 0.97
*   **ROC AUC:** The ROC curve indicated an **AUC value close to 1.0**, signifying excellent discriminative power.

This model showed very strong performance out-of-the-box on the classification task.

### 2. Mixup-Trained Classification Model

*   **Training History Observations:** During training, the model showed high training accuracies (e.g., 0.9872) and low loss (0.1171) with good validation accuracy (0.8190) and loss (0.3791) at the final epoch.
*   **Class-wise Performance (Test Set):**
    *   **Bird Class (0):** Precision = 0.5641, Recall = 0.5455
    *   **Drone Class (1):** Precision = 0.4388, Recall = 0.4574
*   **Overall Metrics (Test Set):**
    *   Macro F1 Score: 0.5013
    *   Balanced Accuracy: 0.5015

**Observation:** The metrics (especially F1 and balanced accuracy) computed on the test set for the Mixup-trained model are **unexpectedly low** compared to its training history and validation during training. This discrepancy suggests a potential issue with the evaluation methodology for this specific model, possibly related to the `tf_mixup_fn` implementation or how `y_true_discrete` and `y_pred_discrete` were extracted and compared.

### 3. YOLOv8 Object Detection Model

*   **Overall Performance (Test Set):**
    *   **mAP50:** 0.827
    *   **mAP50-95:** 0.532
    *   **Overall Precision:** 0.818
    *   **Overall Recall:** 0.792
*   **Class-wise Performance:**
    *   **Bird Class:** Precision = 0.774, Recall = 0.678, mAP50 = 0.731, mAP50-95 = 0.445
    *   **Drone Class:** Precision = 0.861, Recall = 0.907, mAP50 = 0.923, mAP50-95 = 0.619

The YOLOv8 model demonstrated **strong object detection capabilities**, particularly exhibiting excellent performance for the 'drone' class with high recall and mAP scores.

## Summary and Conclusion

This project successfully developed and evaluated deep learning models for aerial object classification (Bird or Drone) and object detection using YOLOv8. The overarching goal was to create a robust solution for critical applications such as wildlife protection, security surveillance, and airspace safety.

### Key Achievements:

1.  **Initial Classification Model Success:** The EfficientNetB0-based classification model demonstrated high performance, achieving an impressive 0.98 accuracy, precision, recall, and F1-score on the test set. This indicates its strong capability in distinguishing between birds and drones.
2.  **YOLOv8 Object Detection:** The YOLOv8 model, after training for 50 epochs on a dedicated object detection dataset, achieved a mAP50 of 0.827 and a mAP50-95 of 0.532 on the test set. It showed particularly strong performance for the 'drone' class (mAP50 of 0.923), proving its effectiveness in locating and identifying objects in real-world aerial scenes.
3.  **Comprehensive Toolkit:** The notebook developed a wide array of utilities, ranging from advanced data preprocessing and augmentation (including Mixup), learning rate scheduling, model interpretability (Grad-CAM), and deployment helpers (TFLite, TF.js, Streamlit app stub), enhancing the project's practicality and extensibility.

### Challenges and Insights:

*   **Mixup Evaluation Discrepancy:** While the Mixup augmentation was intended to improve generalization, the evaluation metrics (F1 and balanced accuracy) for the Mixup-trained classification model were unexpectedly low compared to its training performance. This highlights a potential issue with the evaluation methodology for soft labels or a need for further investigation into the `tf_mixup_fn` implementation and its interaction with evaluation metrics.
*   **Data Handling:** Effective data preprocessing, augmentation, and managing class imbalance (using class weights) were crucial for achieving stable training and good performance across models.
*   **YOLOv8 Integration:** Seamless integration and training of YOLOv8 required careful management of dataset paths and YAML configurations, emphasizing the importance of robust data pipelines for object detection tasks.

### Practical Applications & Future Potential:

The developed solution has direct applicability in:
*   **Wildlife Conservation:** Preventing bird strikes near airports or wind farms.
*   **Security:** Detecting unauthorized drones in sensitive areas.
*   **Environmental Monitoring:** Tracking animal populations without human intervention.

Future work could involve exploring additional advanced augmentation techniques, hyperparameter optimization for all models (especially the Mixup-trained classifier), integrating the YOLOv8 model into a more comprehensive Streamlit application for real-time inference, and exploring other state-of-the-art object detection architectures for further performance gains.

## Summary:

### Data Analysis Key Findings

*   **Project Description:** The project aims to develop a deep learning solution to classify aerial images as 'Bird' or 'Drone' with optional object detection, critical for security surveillance, wildlife protection, and airspace safety. The technical approach involves a custom CNN, transfer learning (ResNet50, MobileNet, EfficientNetB0), YOLOv8 for object detection, and Streamlit for deployment.
*   **Methodological Approach:** The project followed a structured deep learning workflow including dataset understanding, data preprocessing (normalization, resizing to 224x224), data augmentation (rotation, flipping, zoom, brightness, cropping), model building (custom CNN and transfer learning), training & evaluation with various callbacks, optional YOLOv8 object detection, Grad-CAM for interpretability, and Streamlit for deployment.
*   **Tech Stack:** The project utilized Python, TensorFlow/Keras, Ultralytics (for YOLOv8), along with key libraries like NumPy, Matplotlib, Seaborn, Scikit-learn, PIL, OpenCV, and Streamlit for deployment.
*   **Model Performance - Initial Classification Model (EfficientNetB0):** Achieved an impressive overall accuracy of 0.98 on the test set, with balanced class-wise performance (precision, recall, F1-score of 0.98 for 'bird' and 0.97 for 'drone') and an ROC AUC close to 1.0.
*   **Model Performance - Mixup-Trained Classification Model:** Despite high training accuracy (0.9872) and validation accuracy (0.8190), the test set evaluation yielded unexpectedly low metrics, with a Macro F1 Score of 0.5013 and Balanced Accuracy of 0.5015, suggesting a potential issue with the evaluation methodology for soft labels or the `tf_mixup_fn` implementation.
*   **Model Performance - YOLOv8 Object Detection Model:** Demonstrated strong capabilities with an mAP50 of 0.827 and mAP50-95 of 0.532. It showed particularly excellent performance for the 'drone' class (mAP50 = 0.923), while the 'bird' class also performed well (mAP50 = 0.731).
*   **Project Achievements:** Successfully developed high-performing classification and object detection models, and created a comprehensive toolkit covering data processing, augmentation, interpretability, and deployment.
*   **Project Challenges & Insights:** The discrepancy in the Mixup-trained model's evaluation highlighted a potential issue. Effective data handling and careful YOLOv8 integration were identified as crucial.

### Insights or Next Steps

*   **Investigate Mixup Discrepancy:** A thorough investigation is needed into the evaluation methodology for the Mixup-trained classification model, specifically addressing how soft labels interact with evaluation metrics and reviewing the `tf_mixup_fn` implementation to resolve the unexpectedly low test performance.
*   **Deployment and Optimization:** Integrate the high-performing YOLOv8 model into a comprehensive Streamlit application for real-time inference and further optimize the classification models, potentially exploring advanced augmentation techniques and hyperparameter tuning to enhance overall system robustness and user experience.
