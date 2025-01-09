import asyncio
import websockets
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError, ImageFile
import cv2
import numpy as np
import io
import os
import json
import logging
import traceback

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("server.log"),  # Log to a file named server.log
        logging.StreamHandler()  # Also output logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 5 MB

# Ensemble Model Definition
class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        # Model 1: ResNet50
        self.model1 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs1 = self.model1.fc.in_features
        self.model1.fc = nn.Linear(num_ftrs1, num_classes)

        # Model 2: EfficientNet B0
        self.model2 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs2 = self.model2.classifier[1].in_features
        self.model2.classifier[1] = nn.Linear(num_ftrs2, num_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = (x1 + x2) / 2  # Average the outputs
        return x

# YOLOv4 Model Loader
def load_yolo_model(weights_path, config_path, names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# YOLOv4 Inference
def detect_car_with_yolo(image, net, classes, output_layers, threshold=0.3):  # Adjusted threshold
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    car_detected = False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == "car" and confidence > threshold:
                logger.info(f"Car detected with confidence {confidence:.2f}")
                car_detected = True

    if not car_detected:
        logger.info("Car not detected. Sending empty response.")
    return car_detected

# Load PyTorch model
def load_model(model_path, num_classes, device):
    model = EnsembleModel(num_classes=num_classes)
    try:
        logger.info(f"Loading model from {model_path}...")
        # Adjusted torch.load to specify weights_only=True if available
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            # For PyTorch versions that do not support weights_only
            state_dict = torch.load(model_path, map_location=device)
        # Adjust layers for the checkpoint's class count
        checkpoint_num_classes = state_dict["model1.fc.weight"].shape[0]
        logger.info(f"Checkpoint was trained with {checkpoint_num_classes} classes.")

        # Adjust ResNet50 classifier
        model.model1.fc = nn.Linear(model.model1.fc.in_features, checkpoint_num_classes)
        
        # Adjust EfficientNet classifier
        model.model2.classifier[1] = nn.Linear(model.model2.classifier[1].in_features, checkpoint_num_classes)

        # Load state dict
        model.load_state_dict(state_dict)
        logger.info("Model state dict loaded.")

        # Adjust layers back to the current number of classes if necessary
        if checkpoint_num_classes != num_classes:
            model.model1.fc = nn.Linear(model.model1.fc.in_features, num_classes)
            model.model2.classifier[1] = nn.Linear(model.model2.classifier[1].in_features, num_classes)
            logger.info(f"Adjusted model to {num_classes} classes.")
        else:
            logger.info("Model's output layer matches the number of classes.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

    model.to(device)
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model

# Preprocessing
def preprocess_image(image_stream, transform):
    try:
        logger.debug("Opening image...")
        image = Image.open(image_stream).convert('RGB')
        logger.debug("Image opened successfully.")
        tensor = transform(image).unsqueeze(0)
        logger.debug("Image transformed into tensor.")
        return tensor
    except UnidentifiedImageError:
        logger.error("Failed to identify image.")
        raise ValueError("Invalid or corrupted image.")
    except OSError as e:
        logger.error(f"Error loading image: {e}")
        raise ValueError("Image is truncated or corrupted.")
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise ValueError(f"Image preprocessing error: {e}")

# Transform definition
def get_transform(image_size=(299, 299)):
    return transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

async def receive_large_message(websocket):
    """Assemble image data received in chunks."""
    chunks = []
    try:
        while True:
            chunk = await websocket.recv()  # Receive each chunk
            if isinstance(chunk, bytes):
                chunks.append(chunk)
                logger.debug(f"Received chunk of size {len(chunk)} bytes.")
            elif isinstance(chunk, str) and chunk == "END":
                logger.info("Received END signal.")
                break  # Exit loop when END signal is received
    except websockets.exceptions.ConnectionClosedOK:
        logger.warning("Connection closed while receiving data.")
    return b"".join(chunks)

# WebSocket handler
async def handle_client(websocket, lock, model, transform, class_names, yolo_net, yolo_classes, yolo_output_layers):
    try:
        logger.info("Connection received from client.")

        # Receive binary data (image) from the client
        try:
            
            # Receive chunks of data
            logger.info("Receiving image data in chunks...")
            binary_data = await receive_large_message(websocket)
            logger.info(f"Total received data size: {len(binary_data)} bytes.")
        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Client disconnected before sending data: {e}")
            return  # Exit the handler gracefully

        # Validate image size
        if len(binary_data) > MAX_IMAGE_SIZE:
            error_msg = "Image size exceeds limit."
            logger.warning(error_msg)
            await websocket.send(json.dumps({"error": error_msg}))
            return

        # Wrap binary data in BytesIO for processing
        image_stream = io.BytesIO(binary_data)
        logger.debug(f"Received image data of size {len(binary_data)} bytes.")

        image = np.array(Image.open(image_stream).convert('RGB'))

        if not detect_car_with_yolo(image, yolo_net, yolo_classes, yolo_output_layers):
            logger.info("Car not detected. Sending empty response.")
            await websocket.send("")
            return
        
        # Preprocess the image
        logger.info("Preprocessing image...")
        input_tensor = preprocess_image(image_stream, transform)
        input_tensor = input_tensor.to(device)
        logger.info("Image preprocessed successfully.")

        # Make a prediction
        logger.info("Running model inference...")
        async with lock:  # Ensure thread safety for model inference
           outputs = await asyncio.get_event_loop().run_in_executor(
               None, model_forward, model, input_tensor)
           probabilities = torch.nn.functional.softmax(outputs, dim=1)
           confidence, pred = torch.max(probabilities, 1)
        logger.info("Model inference completed.")

        confidence_value = confidence.item()
        logger.debug(f"Confidence score: {confidence_value}")

        # if confidence_value < 0.1:
        #    predicted_class = class_names[pred.item()]
        #    response = ""  # Send an empty string to the client
        #    logger.info(f"Confidence below threshold; sending empty response. {predicted_class}")
        # else:
        predicted_class = class_names[pred.item()]
        # Split the predicted class into company and model name
    #    if '_' in predicted_class:
    #        company, model_name = predicted_class.split('_', 1)
    #        model_name = model_name.replace('_', ' ')  # Replace underscores with spaces in the model name
    #    else:
    #        company = predicted_class
    #        model_name = ""
        # Format as "model_name company" with a space
        response = f"{predicted_class}".strip()
        logger.info(f"Predicted class: {response}")

        # Send the prediction back to the client
        try:
           await websocket.send(response)
           logger.info("Response sent to client.")
        except Exception as e:
           logger.error(f"Error sending response to client: {e}")
           traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
           logger.error(f"Traceback: {traceback_str}")

    except websockets.exceptions.ConnectionClosed as e:
        logger.warning(f"Client disconnected: {e}")
    except Exception as e:
        logger.error(f"Error in server: {e}")
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logger.error(f"Traceback: {traceback_str}")
        # Send error message to client if possible
        try:
            await websocket.send(f"Server error: {str(e)}")
        except:
            pass  # Connection may already be closed

def model_forward(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs

# WebSocket server main
async def main():
    lock = asyncio.Lock()  # Thread safety for model inference

    # Start the server and pass model, transform, and class_names to handle_client
    server = await websockets.serve(
        lambda ws: handle_client(ws, lock, model, transform, class_names, yolo_net, yolo_class_names, yolo_output_layers),
        HOST, PORT, 
        max_size=None
        # compression=None  # Disable compression if client does not support it
    )
    logger.info("Server started and waiting for connections...")
    logger.info(f"Connect to the WebSocket at ws://{HOST}:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    # Parameters
    HOST = "0.0.0.0"
    PORT = 8765
    model_path = "./model/car_model_recognition_final.pt"
    dataset_folder = "./dataset"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=(299, 299))

    yolo_weights = "./model/yolov4.weights"
    yolo_config = "./model/yolov4.cfg"
    yolo_classes = "./model/coco.names"
    yolo_net, yolo_class_names, yolo_output_layers = load_yolo_model(yolo_weights, yolo_config, yolo_classes)

    class_names = sorted([d.name for d in os.scandir(dataset_folder) if d.is_dir()])

    # Load class names from the dataset folder
    if os.path.exists(dataset_folder):
        logger.info(f"Loading class names from dataset folder {dataset_folder}...")
        class_names = sorted([
            d.name for d in os.scandir(dataset_folder) if d.is_dir()
        ])
    else:
        logger.error(f"Dataset folder not found at {dataset_folder}.")
        class_names = []

    # Ensure the class count matches the model
    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("No class names found in dataset folder. Please provide a valid dataset.")
    else:
        logger.info(f"{num_classes} class names loaded.")

    # Load the model
    model = load_model(model_path, num_classes, device)

    # Start the WebSocket server
    logger.info(f"Starting WebSocket server at ws://{HOST}:{PORT}")
    asyncio.run(main())
