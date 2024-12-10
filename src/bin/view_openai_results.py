import http.server
import socketserver
import csv
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import base64

load_dotenv()


class ImageViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_csv(self):
        """Load and parse the CSV file."""
        with open(CSV_PATH, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header if exists
            return list(reader)

    def get_image_base64(self, image_path):
        """Read image file and convert to base64."""
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode("utf-8")

                # Determine MIME type based on file extension
                ext = image_path.lower().split(".")[-1]
                mime_type = f'image/{ext if ext != "jpg" else "jpeg"}'

                return f"data:{mime_type};base64,{img_base64}"
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def do_GET(self):
        """Handle GET requests."""
        # Parse the path to get the index
        path = urlparse(self.path).path

        if path == "/":
            index = 0
        else:
            try:
                index = (
                    int(path[1:]) - 1
                )  # Remove leading slash and convert to 0-based index
            except ValueError:
                self.send_error(404, "Invalid page")
                return

        # Load data
        images_data = self.load_csv()

        # Validate index
        if not (0 <= index < len(images_data)):
            self.send_error(404, "Page not found")
            return

        # Get current image data
        image_path, ground_truth, predicted = images_data[index]

        # Load and encode image
        image_data = self.get_image_base64(image_path)
        if not image_data:
            self.send_error(500, f"Failed to load image: {image_path}")
            return

        # Convert string representations of lists to actual lists
        ground_truth = eval(ground_truth)
        predicted = eval(predicted)

        # Generate navigation links
        prev_link = f"/{index}" if index > 0 else None
        next_link = f"/{index + 2}" if index < len(images_data) - 1 else None

        # Create HTML response
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image {index + 1}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    max-height: 500px;
                }}
                .navigation {{
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    margin: 20px 0;
                }}
                .labels {{
                    margin: 20px 0;
                }}
                .counter {{
                    text-align: center;
                    margin: 10px 0;
                }}
                .filename {{
                    font-family: monospace;
                    color: #666;
                    margin: 10px 0;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="image-container">
                <img src="{image_data}" alt="Image {index + 1}">
            </div>
            <div class="filename">{image_path}</div>
            <div class="counter">
                Image {index + 1} of {len(images_data)}
            </div>
            <div class="labels">
                <h3>Ground Truth Labels:</h3>
                <div>{', '.join(ground_truth)}</div>
                <h3>Predicted Labels:</h3>
                <div>{', '.join(predicted)}</div>
            </div>
            <div class="navigation">
                {f'<a href="{prev_link}"><button>Previous</button></a>' if prev_link else '<button disabled>Previous</button>'}
                {f'<a href="{next_link}"><button>Next</button></a>' if next_link else '<button disabled>Next</button>'}
            </div>
        </body>
        </html>
        """

        self.wfile.write(html.encode())


def run_server(csv_path, port=8000):
    """Start the HTTP server."""
    global CSV_PATH
    CSV_PATH = csv_path

    with socketserver.TCPServer(("", port), ImageViewerHandler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()


if __name__ == "__main__":
    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    json_file = os.path.join(base_directory, "openai_results.csv")

    run_server(json_file)
