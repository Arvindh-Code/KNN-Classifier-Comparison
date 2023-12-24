# KNN Classifier Comparison

This repository contains Python code for implementing the K-Nearest Neighbors (KNN) classifier from scratch and comparing its performance with the scikit-learn library's KNN implementation. The comparison is based on accuracy scores obtained through k-fold cross-validation on three different datasets: Car Evaluation, Breast Cancer, and Hayes-Roth.

## Files

- **knn_from_scratch.py**: Contains the implementation of the KNN classifier from scratch, dataset preprocessing functions, and k-fold cross-validation.
- **knn_sklearn_comparison.py**: Utilizes scikit-learn's KNN implementation for comparison and evaluates accuracy using k-fold cross-validation.
- **car_from_scratch.txt, car_sklearn.txt**: Output files storing accuracy scores for the Car Evaluation dataset.
- **breast_cancer_from_scratch.txt, breast_cancer_sklearn.txt**: Output files storing accuracy scores for the Breast Cancer dataset.
- **hayes_roth_from_scratch.txt, hayens_roth_sklearn.txt**: Output files storing accuracy scores for the Hayes-Roth dataset.
- **README.md**: Documentation explaining the code and the purpose of each file.

## Usage

1. **knn_from_scratch.py**: Run this script to implement the KNN classifier from scratch and obtain accuracy scores for the three datasets. The results are saved in the respective output files.

    ```bash
    python knn_from_scratch.py
    ```

2. **knn_sklearn_comparison.py**: Run this script to compare the KNN classifier implemented from scratch with scikit-learn's KNN. Accuracy scores are obtained and saved in the respective output files.

    ```bash
    python knn_sklearn_comparison.py
    ```

3. **T-Test**: Statistical significance of the differences between accuracy scores from scratch and scikit-learn implementations is tested using the t-test. This is performed in the same script and results are displayed in the console.

## Results

The README includes details on each script, their purpose, and the dataset used. It also provides information on how to run the scripts and interpret the results. The T-Test section highlights whether the differences in accuracy between the two implementations are statistically significant.

## Author

- Aravindh Gopalsamy
- gopal98aravindh@gmail.com

## License

This project is not open for external use or distribution. All rights reserved.

## Important Note for Students

**Warning:** This code is intended for educational purposes only. Please do not use this code for any assignment, and consider it as a reference implementation. Use your own implementation for academic assignments.

Feel free to explore the code and use it as a reference for understanding KNN implementation and model evaluation.
