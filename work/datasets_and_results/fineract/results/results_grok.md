### Key Points
- Research suggests that traditional machine learning algorithms like J48, Random Forest, and SVM can detect code smells in the Apache Fineract project with high accuracy, often above 99%.
- It seems likely that increasing the number of attributes from 13 to 57 does not always improve performance and may even decrease it for some algorithms.
- The evidence leans toward challenges like class imbalance and feature selection affecting the effectiveness of these algorithms for code smell detection.

---

### Introduction to Apache Fineract
Apache Fineract is an open-source platform designed to provide financial institutions with a reliable, robust, and affordable core banking solution, particularly for the underbanked and unbanked populations. It features open APIs, multi-tenancy, and strong security measures, making it suitable for global financial inclusion efforts. You can learn more about it at [Apache Fineract Official Website](https://fineract.apache.org/) and its GitHub repository at [Apache Fineract GitHub](https://github.com/apache/fineract).

### Reasons for Selection
Fineract was chosen for this study due to its large, complex, and actively maintained nature, which is ideal for analyzing code smells. Being Java-based, it aligns well with the machine learning models used, and its open-source availability on GitHub facilitates metric extraction and analysis.

### Summary of WEKA Experiment Results
The WEKA experiments evaluated SMO, RandomForest, and J48 algorithms with 13 and 57 attributes, achieving high accuracies (often above 99%) but showing varied performance based on attribute count and algorithm choice. Details are tabulated below for clarity.

---

### Survey Note: Detailed Analysis of Code Smell Detection in Apache Fineract Using Traditional Machine Learning Algorithms

#### Introduction and Context
This report addresses the analysis of code smell detection in the Apache Fineract project using traditional machine learning algorithms, as part of a WEKA experiment. Apache Fineract, an open-source platform for financial services, was selected due to its relevance and complexity, making it a suitable candidate for studying code quality issues. The objective is to tabulate and summarize the experiment results, analyze problems in code smell detection, and provide an introduction to Fineract with reasons for its selection.

#### Background on Apache Fineract
Apache Fineract is designed to offer a reliable, robust, and affordable core banking solution, particularly targeting the underbanked and unbanked populations globally. It is a mature project with open APIs, multi-tenancy support, and strong security features, used by hundreds of institutions. Its Java-based architecture and active maintenance on GitHub make it ideal for code smell analysis. For further details, refer to [Apache Fineract Official Website](https://fineract.apache.org/) and [Apache Fineract GitHub](https://github.com/apache/fineract).

#### Reasons for Selecting Fineract
Fineract was chosen for this study due to several factors:
- **Size and Complexity**: As a large, actively maintained open-source project, it is likely to accumulate technical debt, making it suitable for code smell detection studies.
- **Java-Based**: Aligns with the machine learning models (J48, RandomForest, SMO) commonly applied to Java code metrics.
- **Open-Source Availability**: Its source code on GitHub facilitates the extraction of code metrics and application of machine learning techniques, enhancing research feasibility.

#### WEKA Experiment Results: Tabulation and Summary
The WEKA experiments involved three algorithms—SMO, RandomForest, and J48—applied to datasets with 13 and 57 attributes, using 10-fold cross-validation on 4059 instances. The results, focusing on accuracy, Kappa statistic, and class-wise metrics, are summarized in the following table:

| Algorithm       | Attributes | Accuracy   | Kappa Statistic | TP Rate (yes) | FP Rate (yes) | Precision (yes) | Recall (yes) | F-Measure (yes) | TP Rate (no) | FP Rate (no) | Precision (no) | Recall (no) | F-Measure (no) |
|-----------------|------------|------------|-----------------|---------------|---------------|-----------------|--------------|-----------------|--------------|--------------|----------------|--------------|----------------|
| SMO             | 13         | 99.9507%   | 0.9442          | 1.000         | 0.000         | 0.895           | 1.000        | 0.944           | 1.000        | 0.000        | 1.000          | 1.000        | 1.000          |
| RandomForest    | 13         | 99.9261%   | 0.9139          | 0.941         | 0.000         | 0.889           | 0.941        | 0.914           | 1.000        | 0.059        | 1.000          | 1.000        | 1.000          |
| J48             | 13         | 99.9507%   | 0.9442          | 1.000         | 0.000         | 0.895           | 1.000        | 0.944           | 1.000        | 0.000        | 1.000          | 1.000        | 1.000          |
| J48             | 57         | 99.9261%   | 0.9139          | 0.941         | 0.000         | 0.889           | 0.941        | 0.914           | 1.000        | 0.059        | 1.000          | 1.000        | 1.000          |
| SMO             | 57         | 99.9507%   | 0.9442          | 1.000         | 0.000         | 0.895           | 1.000        | 0.944           | 1.000        | 0.000        | 1.000          | 1.000        | 1.000          |
| RandomForest    | 57         | 99.9015%   | 0.8745          | 0.824         | 0.000         | 0.933           | 0.824        | 0.875           | 1.000        | 0.176        | 0.999          | 1.000        | 1.000          |

**Summary Observations**:
- SMO and J48 with 13 attributes achieved the highest performance, with 99.9507% accuracy and a Kappa statistic of 0.9442, indicating excellent agreement beyond chance.
- RandomForest with 13 attributes performed slightly worse, with 99.9261% accuracy and a Kappa of 0.9139.
- Increasing attributes to 57 did not improve performance; J48 and RandomForest showed slight degradation, with RandomForest at 57 attributes having the lowest Kappa (0.8745).
- The "no" class (absence of code smells) was perfectly classified in all experiments, while the "yes" class had varying precision and recall, reflecting the challenge of detecting rare code smells.

#### Detailed Analysis: Problems in Code Smell Detection
While the algorithms achieved high accuracy, several challenges were identified:

1. **Class Imbalance**:
   - The dataset is highly imbalanced, with only 17 "yes" instances (code smells) out of 4059, or 0.42%. This imbalance can bias classifiers toward the majority class, potentially leading to overfitting.
   - Despite high Kappa statistics, the imbalance could affect reliability, especially for detecting rare but critical code smells.
   - **Potential Solution**: Techniques like oversampling the minority class, undersampling the majority, or using class weights could address this issue.

2. **Feature Selection and Redundancy**:
   - The experiments showed that increasing attributes from 13 to 57 did not improve and sometimes decreased performance, suggesting many additional attributes are irrelevant or redundant.
   - For instance, J48's decision tree with both 13 and 57 attributes relied solely on "is_blob" and "totalmethodsqty," ignoring other features, indicating potential feature redundancy.
   - SMO's attribute weights also showed some features with zero weight, further supporting this observation.
   - **Potential Solution**: Feature selection methods (e.g., correlation analysis, recursive feature elimination) could identify critical attributes, reducing dimensionality and improving model performance.

3. **Overfitting Risks**:
   - RandomForest with 57 attributes showed a performance drop, suggesting possible overfitting due to increased features. This is consistent with the complexity of ensemble methods on high-dimensional data.
   - **Potential Solution**: Regularization (e.g., limiting tree depth) or pruning could mitigate overfitting, especially for RandomForest and J48.

4. **Interpretability Concerns**:
   - J48 provided interpretable decision trees, using simple rules like "is_blob" and "totalmethodsqty," which is valuable for developers needing to understand flagged code smells.
   - However, SMO and RandomForest are less interpretable due to their complex internal structures, which could hinder practical application in code refactoring.
   - **Potential Solution**: Enhancing interpretability through feature importance analysis (e.g., SMO weights) or rule extraction from complex models could bridge this gap.

5. **Generalization Across Projects**:
   - The models were trained and tested on Fineract, raising questions about their generalizability to other projects or different code smells.
   - Given Fineract's specific domain (financial services), the models might not perform as well on other software types.
   - **Potential Solution**: Cross-project validation or testing on diverse datasets could assess generalizability.

6. **Evolving Codebase**:
   - As an actively maintained project, Fineract's codebase evolves, with new code added and existing code refactored. Code smells may change over time, potentially reducing model effectiveness.
   - **Potential Solution**: Periodic retraining or online learning approaches could adapt models to evolving codebases, ensuring long-term relevance.

7. **False Positives and False Negatives**:
   - Even with high accuracy, false positives (flagging clean code as smelly) and false negatives (missing actual smells) can be costly. For example, RandomForest with 57 attributes had 3 false negatives, potentially leaving problematic code undetected.
   - False positives could lead to unnecessary refactoring, wasting developer time, while false negatives could perpetuate technical debt.
   - **Potential Solution**: Cost-sensitive learning or adjusting classification thresholds could prioritize minimizing critical errors, balancing developer effort and code quality.

#### Conclusion
The WEKA experiments demonstrate that traditional machine learning algorithms can achieve high accuracy (often above 99%) for code smell detection in Apache Fineract, with SMO and J48 with 13 attributes performing best. However, challenges such as class imbalance, feature selection, overfitting, interpretability, generalization, evolving codebases, and error costs highlight areas for improvement. Fineract's selection was justified by its size, complexity, Java-based nature, and open-source availability, making it an ideal subject for this study. Future work could focus on addressing these challenges to develop more effective and practical code smell detection tools, aligning with the objectives of your MSc AI Capstone Project.

**Citations**:
- [Apache Fineract Official Website](https://fineract.apache.org/)
- [Apache Fineract GitHub Repository](https://github.com/apache/fineract)
- [Fineract Platform Documentation](https://fineract.apache.org/docs/current/)