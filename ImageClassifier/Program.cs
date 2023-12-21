using ConsoleTables;
using ImageClassifier;
using Microsoft.ML;
using Microsoft.ML.Data;
using SkiaSharp;
using System.Drawing;
using System.Text;

namespace ImageClassifier
{
    class Program
    {
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

        private static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            ITransformer model = GenerateModel(mlContext);

            Console.WriteLine("Interactive Image Classification. Enter 'exit' to quit.");

            while (true)
            {
                Console.Write("Enter the image name to predict...");
                string imageCode = Console.ReadLine();

                if (imageCode.ToLower() == "exit")
                    break;

                string imagePath = Path.Combine(_imagesFolder, $"{imageCode}.jpg");

                if (File.Exists(imagePath))
                {
                    ClassifySingleImage(mlContext, model, imagePath);
                }
                else
                {
                    Console.WriteLine($"Image not found for code: {imageCode}");
                }
            }

            Console.ReadLine();
        }

        struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        public static void ClassifySingleImage(MLContext mlContext, ITransformer model, string imagePath)
        {
            if (File.Exists(imagePath))
            {
                var imageData = new ImageData()
                {
                    ImagePath = imagePath
                };

                var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
                var prediction = predictor.Predict(imageData);

                var table = new ConsoleTable("Image", "Predicted Classified Label", "Score");

                Console.ForegroundColor = ConsoleColor.Green;

                table.AddRow(Path.GetFileName(imageData.ImagePath), prediction.PredictedLabelValue, prediction.Score?.Max().ToString("F4"));

                Console.ResetColor();

                Console.WriteLine("Individual Image Prediction:");
                Console.WriteLine(table);

                DisplayAsciiArt(imagePath);
            }
            else
            {
                Console.WriteLine($"File not found: {imagePath}");
            }
        }

        public static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
                    .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            // Load the training data from the specified TSV file
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

            // Fit the pipeline to the training data, which trains the machine learning model
            ITransformer model = pipeline.Fit(trainingData);

            // Load the test data from the specified TSV file
            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);

            IDataView predictions = model.Transform(testData);

            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);

            // Display the individual predictions on the test data
            DisplayResults(imagePredictionData);

            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {string.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
            return model;
        }

        public static void DisplayResults(IEnumerable<ImagePrediction> predictions)
        {
            Console.WriteLine("Prediction with test data.");
            var table = new ConsoleTable("Image", "Predicted Classified Label", "Score");

            foreach (var prediction in predictions)
            {
                table.AddRow(Path.GetFileName(prediction.ImagePath), prediction.PredictedLabelValue, prediction.Score?.Max().ToString("F4"));
            }
            Console.WriteLine(table);
        }
        public static void DisplayAsciiArt(string imagePath)
        {
            try
            {
                // Load the image
                Bitmap image = new Bitmap(imagePath);
                int consoleWidth = Console.WindowWidth - 1;
                int consoleHeight = Console.WindowHeight - 1;
                Bitmap resizedImage = new Bitmap(image, new Size(consoleWidth, consoleHeight));

                string asciiArt = ImageToAscii(resizedImage);

                Console.WriteLine(asciiArt);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error displaying image: {ex.Message}");
            }
        }

        public static string ImageToAscii(Bitmap image)
        {
            char[] asciiChars = { ' ', '.', ':', '-', '=', '+'};

            StringBuilder asciiArt = new StringBuilder();

            for (int y = 0; y < image.Height; y += 6)
            {
                for (int x = 0; x < image.Width; x += 3)
                {
                    Color pixelColor = image.GetPixel(x, y);

                    // Calculate the brightness of the pixel
                    int brightness = (int)(pixelColor.GetBrightness() * (asciiChars.Length - 1));

                    // Append the corresponding ASCII character to the StringBuilder
                    asciiArt.Append(asciiChars[brightness]);
                }
                asciiArt.AppendLine();
            }

            return asciiArt.ToString();
        }
    }
}