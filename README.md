# Image Classifier Console App

## Overview

This is a simple console application for image classification using a pre-trained Inception model. The application allows users to interactively predict the class of an image and displays the result along with a visual representation of the image in ASCII art.

## Features

- **Interactive Image Classification:** Users can input the image name to predict its class interactively.
- **Display Results:** The application displays the predicted classified label and score for each image.
- **Visual Representation:** The app converts the image to ASCII art for visual representation in the console.

## Technologies Used

- **ML.NET:** Microsoft's open-source machine learning framework for building custom machine learning models.
- **TensorFlow:** Used for loading the pre-trained Inception model.
- **SkiaSharp:** Utilized for image manipulation and conversion to ASCII art.
- **ConsoleTables:** Used to format and display tabular data in the console.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/ImageClassifier.git
    cd ImageClassifier
    ```

2. Ensure you have the required NuGet packages installed. You can use the following command in the project directory:

    ```bash
    dotnet restore
    ```

3. Download the tutorial assets directory .ZIP file containing the dataset from [here](link_to_dataset_zip).

4. Extract the contents of the downloaded ZIP file into the `assets` directory in your project.

5. Download the InceptionV1 machine learning model from [here](link_to_inception_model).

6. Place the downloaded InceptionV1 model (usually a file with the extension .pb) in the `assets/inception` directory.

7. Build and run the application:

    ```bash
    dotnet run
    ```

8. Enter the image name when prompted to predict its class. Type 'exit' to quit the application.

## Tutorial

For this image classification application, refer to the official Microsoft Learn tutorial: [Image Classification with ML.NET](https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification).

## Installing Packages

To run the Image Classifier Console App, you need to install the following NuGet packages:

1. **Microsoft. ML.ImageAnalytics:**

    ```bash
    dotnet add package Microsoft.ML.ImageAnalytics
    ```

2. **SciSharp.TensorFlow.Redist:**

    ```bash
    dotnet add package SciSharp.TensorFlow.Redist
    ```

3. **Microsoft. ML.TensorFlow:**

    ```bash
    dotnet add package Microsoft.ML.TensorFlow
    ```

4. **Microsoft.ML:**

    ```bash
    dotnet add package Microsoft.ML
    ```

5. **ConsoleTables:**

    ```bash
    dotnet add package ConsoleTables --version 2.4.2
    ```
## Sample Output

Upon running the Image Classifier Console App and providing the image name for prediction, you will see output similar to the following:

```plaintext
Interactive Image Classification. Enter 'exit' to quit.
Enter the image name to predict...toaster3

Individual Image Prediction:
+----------------------+-----------------------------+--------+
| Image                | Predicted Classified Label | Score  |
+----------------------+-----------------------------+--------+
| toaster3.jpg         | Toaster                     | 0.9882 |
+----------------------+-----------------------------+--------+

```


## Notes
- Ensure that the required dependencies and permissions are set up for loading TensorFlow models.

[link_to_dataset_zip]: https://example.com/path/to/dataset.zip
[link_to_inception_model]: https://example.com/path/to/inception_model.pb
