from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

# Load the pretrained model
learn_inf = torch.jit.load("checkpoints/transfer_exported.pt")


# Function to classify the uploaded image
def classify_image(image_path):
    # Load the image
    img = Image.open(image_path)
    img.load()

    # Display the image (optional)
    img.show()

    # Transform the image to tensor
    timg = T.ToTensor()(img).unsqueeze_(0)

    # Call the model to get predictions
    softmax = learn_inf(timg).data.cpu().numpy().squeeze()

    # Get the indexes of the classes ordered by softmax (larger first)
    idxs = np.argsort(softmax)[::-1]

    # Print the top 5 classes with their probabilities
    print("Top 5 predictions:")
    for i in range(5):
        p = softmax[idxs[i]]
        landmark_name = learn_inf.class_names[idxs[i]]
        print(f"{landmark_name} (prob: {p:.2f})")


if __name__ == "__main__":
    image_path = "static_images/test/16.Eiffel_Tower/3828627c8730f160.jpg"
    classify_image(image_path)
