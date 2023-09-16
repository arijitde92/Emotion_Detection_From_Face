import torch
import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from preprocess_data import get_paths_and_labels, get_transforms, get_num_classes, label_dict
from FER_Dataset import FERDataset
from Model import Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
MODEL_SAVE_DIR = 'trained_model'


def test(model, test_loader):
    correct = 0
    model.eval()
    num_samples = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, num_samples, (100. * correct / num_samples)))
    return 100. * correct / num_samples


def test_image(image_path, model, save_image=True, save_image_dir='test_results', test_transform=None):
    # Load data and get label
    image = cv2.imread(image_path)
    file_name = os.path.basename(image_path)
    if test_transform is not None:
        data = test_transform(image=image)["image"]
    model.eval()
    data.to(DEVICE)
    with torch.no_grad():
        output = model(data.float())
        prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        for key, value in label_dict.items():
            if prediction == value:
                emotion = key
                break
        print("Detected Emotion:", emotion)
        if save_image:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (10, 500)
            font_scale = 1
            font_color = (255, 255, 255)
            thickness = 1
            line_type = 2

            cv2.putText(image, emotion,bottom_left_corner, font, font_scale, font_color, thickness, line_type)

            # Display the image
            cv2.imshow("Result", image)

            # Save image
            output_path = os.path.join(save_image_dir, file_name)
            print("Writing image to ", output_path)
            cv2.imwrite(output_path, image)
            cv2.waitKey(0)


if __name__ == '__main__':
    model = Net(dropout=0.2, num_classes=get_num_classes())
    print("Loading Model")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'best_model.pth')))
    print(f"Using {DEVICE} for training")
    model.to(DEVICE)
    print("Creating Data loaders")
    _, test_transforms = get_transforms()
    _, _, test_data = get_paths_and_labels(val_split=0.2)

    test_dataset = FERDataset(test_data[0], test_data[1], test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    print("Starting Testing")
    t_acc = test(model, test_loader)
