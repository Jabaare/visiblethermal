import face_recognition
import os
import cv2

def load_gallery_images(gallery_path):
    """Loads and encodes all images from the gallery folder."""
    known_encodings = []
    known_names = []

    for file_name in os.listdir(gallery_path):
        # Load each image from the gallery
        image_path = os.path.join(gallery_path, file_name)
        image = face_recognition.load_image_file(image_path)

        # Encode the face (assuming one face per gallery image)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Ensure a face was found
            known_encodings.append(encodings[0])
            # Use the file name (without extension) as the label
            known_names.append(os.path.splitext(file_name)[0])

    return known_encodings, known_names

def identify_subjects_in_probe(probe_image_path, known_encodings, known_names, threshold=0.6):
    """Identifies all faces in the probe image with a confidence threshold."""
    # Load and encode the probe image
    probe_image = face_recognition.load_image_file(probe_image_path)
    probe_encodings = face_recognition.face_encodings(probe_image)

    if not probe_encodings:
        return "No faces detected in the probe image."

    results = []
    for i, probe_encoding in enumerate(probe_encodings):
        # Compare the probe face with known faces
        matches = face_recognition.compare_faces(known_encodings, probe_encoding)
        distances = face_recognition.face_distance(known_encodings, probe_encoding)

        if not any(matches):
            results.append(f"Face {i + 1}: No match found.")
            continue

        # Find the best match
        best_match_index = distances.argmin()
        best_match_distance = distances[best_match_index]
        
        if best_match_distance < threshold:
            best_match_name = known_names[best_match_index]
            results.append(f"Face {i + 1}: Identified as {best_match_name} (Confidence: {1 - best_match_distance:.2f})")
        else:
            results.append(f"Face {i + 1}: No match found (Below confidence threshold).")

    return results

# Paths
gallery_path = "gallery"  # Folder containing known images
probe_image_path = "probe.jpg"  # Probe image to identify
confidence_threshold = 0.6  # Confidence threshold for face identification

# Step 1: Load gallery images
known_encodings, known_names = load_gallery_images(gallery_path)

# Step 2: Identify subjects in the probe image
results = identify_subjects_in_probe(probe_image_path, known_encodings, known_names, threshold=confidence_threshold)

# Print results
if isinstance(results, str):  # If no faces were detected
    print(results)
else:
    for result in results:
        print(result)
