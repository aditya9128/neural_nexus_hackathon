## Why This Dataset?

Everyone saw a 45 GB beast of a dataset and walked the other way. We saw it and said, "let's make it work." The NIH Chest X-ray dataset is massive, messy, and unapologetic about crashing your RAM — so we fought back with Dask for lazy computation, Parquet for efficient storage, and aggressive garbage collection to keep memory in check. No downsampling, no shortcuts, no trimming the data to fit the machine. We made the machine fit the data.

## How the Pipeline Works

Before the model can learn anything, we need to wrangle the raw data into something usable. The pipeline starts by scanning through the NIH Chest X-ray dataset — which is spread across multiple folders like `images_001`, `images_002`, and so on — and building a quick lookup map of every PNG file it finds. Then it reads the massive metadata CSV using Dask so we don't blow up memory, cleans up the column names, and converts the disease labels (things like "Atelectasis|Effusion") into a simple multi-hot vector of 14 columns. Once that's done, everything gets saved as a Parquet file so we never have to repeat this work again.

The data is split carefully by using PatientID for uniqueness to avoid any leakage between training, validation, and test sets. If the official NIH split files are available, those are used — otherwise, the pipeline falls back to a random 70/15/15 split. Either way, the split happens at the patient level, meaning all X-rays from the same person stay in the same bucket. The images themselves aren't loaded into memory upfront; instead, the PyTorch dataset only stores file paths and labels, and each image is read, resized to 224x224, and augmented on the fly during training. Training images get random crops, flips, rotations, and color jitter to help the model generalize, while validation and test images just get a clean resize and normalization.

Training runs on a DenseNet121 backbone pretrained on ImageNet, with a custom classification head bolted on top. The whole thing uses mixed-precision (FP16) to save GPU memory and gradient accumulation to effectively double the batch size without needing more VRAM. A cosine annealing schedule gradually drops the learning rate, and because the disease labels are wildly imbalanced (Hernia is super rare compared to Infiltration), the loss function uses per-class weights so the model doesn't just learn to predict the common stuff. The best model checkpoint — picked by whichever epoch has the highest validation AUROC — gets saved automatically.

Once training wraps up, the pipeline loads that best checkpoint and runs it across the entire test set. It reports AUROC and F1 scores for each of the 14 diseases individually, plus the overall averages. The raw prediction probabilities are also dumped to a CSV so you can do your own analysis later. If you just want to throw a single X-ray at the model, there's a `predict_single_image` function that prints out the top diseases with a little visual bar chart right in the terminal — handy for quick sanity checks.

## What The UI Adds

The project is not just a training script sitting in a folder. It also includes a frontend and backend setup that turns the model into something people can actually use. A user can upload a chest X-ray through the interface, wait a moment for the prediction to run, and then view the most likely conditions along with confidence scores, recommendations, and a visual explanation of what the model focused on.

The UI is built to feel practical rather than intimidating. Instead of dumping raw numbers on the screen, it organizes the output into clear sections: uploaded image preview, prediction results, a heatmap viewer, and recommendation cards. That makes the project easier to demo and much more useful for explaining the model to someone who is not deep into machine learning code.

## Gemini API In Simple Terms

After the model predicts the most likely disease, the backend sends that top result to the Gemini API. Gemini is used here as a language layer, not as the image classifier itself. The model handles the medical image prediction, and Gemini turns that prediction into more readable guidance like precautions, basic do's and don'ts, and when someone should consider seeing a doctor.

That separation is important. The chest X-ray model is responsible for the computer vision part, while Gemini helps present the result in a more human way. If the API is unavailable, the project still falls back to built-in recommendations so the app does not completely break.

## How The Heatmap Works

One of the most useful parts of this project is the Grad-CAM heatmap. After prediction, the backend looks at the top predicted class and traces back which parts of the image influenced that decision the most. It does this by capturing the model's activations and gradients from the convolutional layers, combining them into a class-specific attention map, and resizing that map to match the original X-ray.

From there, the app generates two views: a standalone heatmap and an overlay version blended on top of the original image. In plain terms, the brighter and warmer regions are the spots the model paid more attention to when making its decision. It is not a diagnosis by itself, but it gives users a much better sense of why the model responded the way it did instead of asking them to trust a probability score blindly.
