import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
import torch
from torchvision import transforms, utils as vutils
from PIL import Image
import numpy as np
from network.Transformer import Transformer  # your model import

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# --- User auth dummy storage ---
users = {}

# --- Folder paths ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
GALLERY_FOLDER = 'static/gallery'
MODEL_FOLDER = 'pretrained_model'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GALLERY_FOLDER, exist_ok=True)

# --- Cartoon model mapping ---
model_map = {
    'Miyazaki': 'miyazaki_net_G_float.pth',
    'Hosoda': 'hosoda_net_G_float.pth',
    'Shinkai': 'shinkai_net_G_float.pth',
    'Kon Satoshi': 'kon_satoshi_G_float.pth'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Authentication routes
# --------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if username in users:
            flash("Username already exists!")
        else:
            users[username] = {'email': email, 'password': password}
            flash("Registration successful! Please log in.")
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user['password'] == password:
            session['user'] = username
            flash("Logged in successfully!")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password!")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.")
    return redirect(url_for('login'))


# --------------------------
# Cartoonize app routes
# --------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if 'user' not in session:
        flash("Please log in to access the cartoonizer.")
        return redirect(url_for('login'))

    output_image = None

    if request.method == "POST":
        file = request.files.get("image")
        style = request.form.get("style")

        if not file or not style:
            flash("Please upload an image and select a style.")
            return redirect(url_for('index'))

        filename = file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        model_file = model_map.get(style)
        model_path = os.path.join(MODEL_FOLDER, model_file)

        model = Transformer()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)

        image = Image.open(input_path).convert("RGB")
        w, h = image.size
        ratio = h / float(w)

        load_size = 450
        if ratio > 1:
            h = load_size
            w = int(h / ratio)
        else:
            w = load_size
            h = int(w * ratio)

        image = image.resize((w, h), Image.BICUBIC)
        image = np.array(image)[:, :, [2, 1, 0]]
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = -1 + 2 * image
        image = image.to(device)

        with torch.no_grad():
            output = model(image)[0]
            output = output[[2, 1, 0], :, :]
            output = (output.cpu().float() * 0.5 + 0.5)

        output_filename = f"cartoon_{style.lower()}_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        gallery_path = os.path.join(GALLERY_FOLDER, output_filename)

        vutils.save_image(output, output_path)
        vutils.save_image(output, gallery_path)

        output_image = output_filename

    return render_template("index.html", output_image=output_image, styles=model_map.keys())


@app.route("/gallery")
def gallery():
    if 'user' not in session:
        flash("Please log in to access the gallery.")
        return redirect(url_for('login'))

    images = os.listdir(GALLERY_FOLDER)
    return render_template("gallery.html", images=images)


@app.route("/delete_image", methods=["POST"])
def delete_image():
    if 'user' not in session:
        flash("Please log in to perform this action.")
        return redirect(url_for('login'))

    image_name = request.form.get("image_name")
    image_path = os.path.join(GALLERY_FOLDER, image_name)
    if os.path.exists(image_path):
        os.remove(image_path)
    return redirect(url_for('gallery'))


if __name__ == "__main__":
    app.run(debug=True)
