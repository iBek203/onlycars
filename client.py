import asyncio
import websockets
from PIL import Image
import io

async def send_image(image_path, websocket_url):
    try:
        # Open and resize the image
        image = Image.open(image_path)
        
        # Save resized image to BytesIO
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        binary_data = buffer.read()

        # Connect to the WebSocket server
        async with websockets.connect(websocket_url) as websocket:
            print(f"Connected to server: {websocket_url}")

            # Send the binary image data
            await websocket.send(binary_data)
            print("Resized image sent to server.")

            # Send the "END" signal
            await websocket.send("END")
            print('Sent "END" signal to server.')

            # Receive the response
            response = await websocket.recv()
            print(f"Response from server: {response}")

    except Exception as e:
        print(f"Error: {e}")

# Define image path and WebSocket server URL
image_path = "./traker2.jpg"  # Replace with the path to your image
websocket_url = "ws://localhost:8765"  # Replace with your WebSocket server URL

# Run the client
asyncio.run(send_image(image_path, websocket_url))
