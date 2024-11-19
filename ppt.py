from flask import Flask, request, jsonify
import zipfile
import xml.dom.minidom
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dictionary to store slide embeddings
embeddings_dict = {}

def list_ppt_folder_contents(pptx_file):
    try:
        # Open the .pptx file as a ZIP archive
        with zipfile.ZipFile(pptx_file, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            ppt_folder_files = [f for f in all_files if f.startswith("ppt/")]
            if ppt_folder_files:
                return ppt_folder_files
            else:
                print("No 'ppt/' folder found in the .pptx file.")
    except zipfile.BadZipFile:
        print(f"Error: {pptx_file} is not a valid .pptx file.")
    except FileNotFoundError:
        print(f"Error: File '{pptx_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# To look into a particular .xml file in the 'ppt/' folder, you can use the following function
def read_ppt_folder_file(pptx_file, file_name):
    try:
        # Open the .pptx file as a ZIP archive
        with zipfile.ZipFile(pptx_file, 'r') as zip_ref:
            # Read the contents of the specified file in the 'ppt/' folder
            with zip_ref.open(f"{file_name}") as file:
                content = file.read()
                return content
    except zipfile.BadZipFile:
        print(f"Error: {pptx_file} is not a valid .pptx file.")
    except FileNotFoundError:
        print(f"Error: File '{pptx_file}' not found.")
    except KeyError:
        print(f"Error: File '{file_name}' not found in the 'ppt/' folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




# Function to extract and encode slides
def process_pptx(pptx_file_path):
    all_files_list = list_ppt_folder_contents(pptx_file_path)
    slides = [f for f in all_files_list if f.startswith("ppt/slides/slide")]
    slide_embeddings = {}
    
    for slide in slides:
        xml_content = read_ppt_folder_file(pptx_file_path, slide)
        xml_content = xml.dom.minidom.parseString(xml_content)
        pretty_xml_as_string = xml_content.toprettyxml()

        # Get all content between <p:txBody> and </p:txBody> tags
        text_content = re.findall(r'<a:t>(.*?)</a:t>', pretty_xml_as_string)

        if text_content:
            # Join all text into a single string for each slide
            combined_text = " ".join(text_content)
            # create embeddings for each slide
            embeddings = model.encode([combined_text])
            # Only take the number part of the slide name
            slide_number = re.search(r'\d+', slide).group()
            slide_embeddings[slide_number] = embeddings[0]

    return slide_embeddings


def find_most_similar_slide(query, embeddings_dict):
    query_embedding = model.encode([query])[0]
    similarities = {}
    for slide, embedding in embeddings_dict.items():
        # Reshape embedding to 2D arrays for cosine similarity
        embedding = embedding.reshape(1, -1)
        query_embedding = query_embedding.reshape(1, -1)

        # Calculate similarity
        similarity = cosine_similarity(query_embedding, embedding)
        similarities[slide] = similarity[0][0]
    if not similarities:
        raise ValueError("No valid slides found for comparison.")
    most_similar_slide = max(similarities, key=similarities.get)
    return most_similar_slide, similarities[most_similar_slide]



@app.route('/upload', methods=['POST'])
def upload_pptx():
    global embeddings_dict

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pptx_file = request.files['file']
    try:
        # Save file temporarily
        file_path = f"./{pptx_file.filename}"
        pptx_file.save(file_path)

        # Process the file to extract embeddings
        embeddings_dict = process_pptx(file_path)

        return jsonify({"message": "PPTX file processed successfully", "slides": len(embeddings_dict)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask route to query slides
@app.route('/query', methods=['GET'])
def query_slide():
    global embeddings_dict

    if not embeddings_dict:
        return jsonify({"error": "No slides have been processed. Upload a PPTX file first."}), 400

    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        most_similar_slide, similarity = find_most_similar_slide(query, embeddings_dict)
        return jsonify({"most_similar_slide": most_similar_slide, "similarity": float(similarity)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)