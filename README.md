# Image to image translation GUI

As part of my end of study insternship, I built this Streamlit UI application that allows users to perform various image-to-image translation tasks using their custom images.

## Installation

1. **Create a new virtual environment** (recommended):

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
   ```

2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the necessary packages specified in the `requirements.txt` file.

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run main.py
   ```

   This will launch the Streamlit UI in your default web browser.

2. **Use the UI**:
   - Select the desired image-to-image translation task from the dropdown menu.
   - Upload the input image(s) required for the selected task or you can draw in case of edges2shoes task.
   - Click the "Translate" button to perform the translation.
   - The translated output image(s) will be displayed in the UI.

## Tasks

The following image-to-image translation tasks are currently supported:

- **Edge-to-Shoes**: Generate a realistic shoe image from an edge map.
- **Other tasks will be added soon.**

## Models

The project currently uses the following deep learning models:

- **Pix2Pix**
- **CycleGAN**

## Contributing

Contributions are welcome! If you would like to add more features or improve the existing functionality, please feel free to submit a pull request.

## Acknowledgments

- The Pix2Pix and CycleGAN models were built using the [Official pix2pix and cyclegan repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) deep learning framework.
- The Streamlit UI was created using the [Streamlit](https://streamlit.io/) library.
