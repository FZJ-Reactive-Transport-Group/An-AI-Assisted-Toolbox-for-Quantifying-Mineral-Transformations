# An-AI-Assisted-Toolbox-for-Quantifying-Mineral-Transformations

An AI-Assisted Toolbox for Quantifying Mineral Transformations: Investigating the Metastability of Amorphous Calcium Carbonate (ACC) with Microfluidics

This GitHub repository houses a specialized toolbox developed to facilitate the quantitative analysis of mineral transformations, with a particular focus on the study of amorphous calcium carbonate (ACC) metamorphosis. Integrating sophisticated image processing algorithms and deep learning models, specifically convolutional neural networks employing U-net architecture, this tool automates the analysis of extensive datasets. It accurately determines the temporal dynamics of mineral transformations and identifies resultant mineral polymorphs. Tailored to examine the intricacies of ACC transformations, the toolbox accommodates a broad spectrum of mineralogical research, offering a robust platform for advancing the scientific understanding of mineral transformation processes.

## Overview of the Toolbox Workflow

### Preprocessing Steps

- **Image Conversion and Noise Reduction**: Utilizes the OpenCV library for converting images into greyscale, emphasizing structural features for more effective subsequent analyses.
- **Bilateral Filtering and Non-local Means Denoising**: Reduces noise while preserving edges, enhancing the clarity of precipitates.
- **Binarization and Adaptive Thresholding**: Converts images to binary format, using adaptive thresholding to handle varying lighting and ensure consistent segmentation.
- **Morphological Operations**: Employs operations, especially opening, to refine segmented images, separate adjacent objects, and refine borders while preserving original structures.

### Convolutional Neural Network (CNN) Integration

- **Core Component**: Features a CNN model trained on annotated images of ACC, calcite, and vaterite precipitates for advanced image recognition and classification.
- **Training and Segmentation**: Leverages the StarDist Python package, enabling semantic segmentation of precipitates in microscopy images and identification of different mineral phases based on morphology.

### Post-processing and Feature Extraction

- **Advanced Algorithms**: Integrates feature extraction algorithms for precise crystal border detection and differentiation of mineral morphologies.
- **Contour Detection**: Utilizes advanced algorithms for detecting shapes with varying degrees of circularity, aiding in the identification of distinct mineral morphologies.

## Technical Specifications

- **Programming Language**: Python 3.9.13
- **Key Libraries and Frameworks**:
  - **Image Processing**: OpenCV (version 4.6.0.66) for image conversion, noise reduction, and contour detection.
  - **Numerical Computations**: SciPy (version 1.7.3) and NumPy (version 1.21.5) for adaptive thresholding and morphological operations.
  - **Deep Learning**: TensorFlow (version 2.11.0) and Keras (version 2.11.0) within the StarDist model framework (version 0.8.3) for CNN development and training.

## Capabilities

- **Automated Analysis**: Automates the analysis of extensive datasets from droplet microfluidic experiments, capturing the dynamics of mineral transformations.
- **Mineral Polymorph Identification**: Robustly quantifies the influence of various factors, including confinement and additives, on crystal polymorphism.
- **Versatility**: Designed to accommodate a wide range of mineralogical research, enabling a deeper understanding of mineral transformation processes.

## Technical Specifications and Resources

The AI-assisted toolbox for quantifying mineral transformations leverages Python 3.9.13 and a suite of open-source libraries and frameworks, designed to facilitate the processing and analysis of image data from droplet microfluidic experiments. This section details the core components and provides links to external resources for users interested in further exploration or customization of the toolbox.

### Core Libraries and Frameworks

#### Image Processing and Analysis:
- **OpenCV (version 4.6.0.66)**: Used for image preprocessing, including greyscale conversion and noise reduction.
  - [OpenCV Documentation](https://opencv.org/)

#### Numerical and Scientific Computing:
- **SciPy (version 1.7.3)** and **NumPy (version 1.21.5)**: Employed for adaptive thresholding, morphological operations, and numerical computations.
  - [SciPy Documentation](https://www.scipy.org/)
  - [NumPy Documentation](https://numpy.org/)

#### Deep Learning and Neural Networks:
- **TensorFlow (version 2.11.0)** and **Keras (version 2.11.0)** within the **StarDist model framework (version 0.8.3)**: Powers the toolbox's CNN model, based on the U-net architecture, for segmentation and classification of mineral phases.
  - [TensorFlow Guide](https://www.tensorflow.org/)
  - [Keras Documentation](https://keras.io/)
  - [StarDist GitHub](https://github.com/mpicbg-csbd/stardist)

### Further Learning and Customization

These tools offer in-depth documentation and community support for further learning. Whether aiming to understand the methodologies more deeply or to adapt the toolbox for additional applications, the following resources are recommended:

- Explore advanced image processing techniques with **[OpenCV](https://opencv.org/)**.
- Delve into scientific computing with **[SciPy](https://www.scipy.org/)** and **[NumPy](https://numpy.org/)**.
- Learn more about neural network architectures with **[TensorFlow](https://www.tensorflow.org/)** and **[Keras](https://keras.io/)**.
- Specialize in image segmentation using **[StarDist](https://github.com/mpicbg-csbd/stardist)**.

Designed as a dynamic and adaptable platform, this toolbox encourages customization to meet specific mineralogical research needs, facilitating the automated quantification of mineral transformations.
