from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from utils import initialize_model, search, preprocess_image, tokenize_text

app = Flask(__name__)

# Initialize model and configuration
model, preprocess, embeddings, df = initialize_model()
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'coco_images_resized'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/coco_images_resized/<path:filename>')
def serve_images(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for the web application.
    Handles text, image, and hybrid queries with optional PCA.
    """
    results = []
    if request.method == "POST":
        query_type = request.form["query_type"]
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))
        text_query = request.form.get("text_query", "").strip()
        use_pca = "use_pca" in request.form
        n_components = int(request.form.get("n_components", 5))
        image_query_path = None

        if "image_query" in request.files:
            file = request.files["image_query"]
            if file:
                filename = secure_filename(file.filename)
                image_query_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(image_query_path)

        if query_type == "text_query" and text_query:
            text_query_embedding = tokenize_text(text_query, model)
            results = search(text_query_embedding, embeddings, df, use_pca=use_pca, n_components=n_components)

        elif query_type == "image_query" and image_query_path:
            image_query_embedding = preprocess_image(image_query_path, preprocess, model)
            results = search(image_query_embedding, embeddings, df, use_pca=use_pca, n_components=n_components)

        elif query_type == "hybrid_query" and image_query_path and text_query:
            text_query_embedding = tokenize_text(text_query, model)
            image_query_embedding = preprocess_image(image_query_path, preprocess, model)
            hybrid_query_embedding = 0.5 * text_query_embedding + (1.0 - hybrid_weight) * image_query_embedding
            results = search(hybrid_query_embedding, embeddings, df, use_pca=use_pca, n_components=n_components)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
