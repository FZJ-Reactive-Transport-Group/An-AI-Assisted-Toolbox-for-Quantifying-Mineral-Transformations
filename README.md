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

## Detailed Guide on Image Annotation for Machine Learning

Before leveraging the AI-Assisted Toolbox for Quantifying Mineral Transformations, it is essential to prepare your dataset through a meticulous image annotation process. This preparatory step is fundamental in training the toolbox's convolutional neural network (CNN) model to recognize and classify various mineral phases accurately.

### Step-by-Step Annotation Process

1. **Dataset Compilation**: Begin by compiling a diverse set of images from your microfluidic experiments. Ensure this dataset encompasses the full range of mineral transformations you aim to analyze, capturing different phases, conditions, and outcomes.

2. **Choosing the Right Annotation Tool**: Select an annotation tool that aligns with your project's needs. The tool should offer precision in marking areas of interest and be user-friendly. Here are three recommended tools, each suitable for different aspects of annotation:
   - **ImageJ**: Ideal for scientific images, allowing detailed annotations and measurements. [Download ImageJ](https://imagej.nih.gov/ij/download.html)
   - **VGG Image Annotator (VIA)**: A versatile, web-based tool for annotating images, videos, and audio. It's user-friendly and suitable for beginners. [Access VIA](http://www.robots.ox.ac.uk/~vgg/software/via/)

3. **Manual Annotation Process**: Carefully annotate your images by delineating the areas of interest, such as specific mineral phases or features critical for your study. This step might involve drawing precise boundaries around regions or tagging certain features within the images.

### Best Practices for Effective Annotation

- **Consistency**: Maintain consistency in your annotations across the dataset. Consistent annotations are crucial for training the model effectively, as inconsistencies can lead to confusion and inaccuracies.

- **Quality Assurance**: Prioritize the quality of annotations over quantity. High-quality, precise annotations significantly enhance the model's learning capability and overall accuracy.

- **Familiarity with Data**: Deeply understand the features, phases, and transformations present in your images. A thorough knowledge of your subject matter is key to creating accurate and meaningful annotations.

- **Collaboration and Review**: If possible, collaborate with peers for the annotation process. Having multiple reviewers can help ensure the accuracy and consistency of the annotations.

### Importance of Detailed Annotations

Detailed and accurate annotations are the cornerstone of effective machine learning model training. They serve as the "ground truth" that teaches the model how to interpret the images it will analyze. The quality of these annotations directly impacts the model's ability to identify mineral phases and transformations accurately, making it a critical step in your research workflow.

By adhering to these guidelines and utilizing the recommended tools for annotation, you prepare your image data for successful analysis with the AI-Assisted Toolbox for Quantifying Mineral Transformations. This step, while time-consuming, is vital for unlocking the full analytical potential of the toolbox in advancing your mineralogical studies.

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

## Getting Started
This section guides you through setting up your environment to use the AI-Assisted Toolbox for Quantifying Mineral Transformations.

## Prerequisites
Python 3.9 or higher
Git (for cloning the repository)
A compatible operating system (Windows or Linux)
Installation
### Clone the repository: Clone this GitHub repository to your local machine to get started with the toolbox.
`git clone https://github.com/yourrepository/An-AI-Assisted-Toolbox-for-Quantifying-Mineral-Transformations.git`
### Create a virtual environment (optional but recommended): Using a virtual environment helps manage dependencies and avoid conflicts with other Python projects.
`python -m venv venv
source venv/bin/activate`  
On Windows, use `venv\Scripts\activate`
### Install the required libraries: Navigate to the cloned repository's directory and install the required libraries listed in the requirements.txt.
`pip install -r requirements.txt`

## Running the Notebooks
1. Launch Jupyter Notebook or JupyterLab: With the dependencies installed, you can start Jupyter Notebook or JupyterLab to open and run the provided notebooks.
`jupyter notebook` or `jupyter lab`
2. Navigate to the cloned repository's directory within the Jupyter interface and open the notebook you wish to run.
3. Follow the instructions within each notebook to proceed with your analysis.


## Example Usage
After setting up your environment, you're ready to use the toolbox:

1. Prepare your dataset according to the guidelines provided in the notebooks.
2. Annotate your images, ensuring you have 60-80 well-annotated images to optimize the model's learning. Use recommended tools like ImageJ, VGG Image Annotator (VIA).
3. Execute the notebooks in order, starting from preprocessing to training and testing the model.
Note
The quality, accuracy of shapes to be detected, and lighting conditions significantly affect the segmentation quality and the model's effectiveness. It's crucial to ensure your annotated images reflect the conditions of the microscopic shapes under investigation for optimal results.

## Contributing

We welcome contributions from the community! Whether it's adding new features, improving documentation, or reporting bugs, your help makes this toolbox better for everyone.

To contribute:
- Fork the repository.
- Create a new branch for your feature or fix.
- Submit a pull request with a clear description of your changes.

Please refer to our [Contribution Guidelines](LINK) for more detailed information.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments

- This work was supported by the Helmholtz AI projects via funding the T6 project (grant ZT-1-PF-5-084). The experiments were conducted within the frame of the VESPA II project which was funded by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (BMUV, grant 02E11607D).
- Special thanks to all our beta testers for their valuable feedback.

## Frequently Asked Questions (FAQ)

- **Q: Can I use the toolbox on macOS?**
- A: Currently, the toolbox is optimized for Windows and Linux. macOS compatibility is planned for future updates.

- **Q: How many images do I need to start analyzing my data?**
- A: A set of 60-80 well-annotated images is recommended for meaningful analysis. The quality, accuracy of shapes to be detected, and lighting conditions significantly affect the segmentation quality and the model's ability to detect the microscopic shapes under investigation.

- **Q: What should I do if I encounter an error during installation?**
- A: Ensure all prerequisites are installed and check the error message for clues. For further assistance, please open an issue in the GitHub repository.
