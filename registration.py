import face_recognition
import os
import cv2
import logging
import argparse
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load gallery images and encode faces
def load_gallery_images(gallery_path):
    """Loads and encodes all images from the gallery folder."""
    if not os.path.exists(gallery_path):
        logging.error(f"Gallery path '{gallery_path}' does not exist.")
        return [], []

    known_encodings = []
    known_names = []

    for file_name in os.listdir(gallery_path):
        image_path = os.path.join(gallery_path, file_name)
        try:
            image = face_recognition.load_image_file(image_path)
        except Exception as e:
            logging.warning(f"Failed to load {file_name}: {str(e)}")
            continue

        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file_name)[0])
            logging.info(f"Encoded {file_name}")
        else:
            logging.warning(f"No face found in {file_name}")

    return known_encodings, known_names

# Identify faces in the probe image
def identify_subjects_in_probe(probe_image_path, known_encodings, known_names, threshold=0.6):
    probe_image = face_recognition.load_image_file(probe_image_path)
    probe_encodings = face_recognition.face_encodings(probe_image)

    if not probe_encodings:
        return ["No faces detected in the probe image."]

    results = []
    for i, probe_encoding in enumerate(probe_encodings):
        matches = face_recognition.compare_faces(known_encodings, probe_encoding)
        distances = face_recognition.face_distance(known_encodings, probe_encoding)

        if not any(matches):
            results.append(f"Face {i + 1}: No match found.")
            continue

        best_match_index = distances.argmin()
        best_match_distance = distances[best_match_index]
        
        if best_match_distance < threshold:
            best_match_name = known_names[best_match_index]
            results.append(f"Face {i + 1}: Identified as {best_match_name} (Confidence: {1 - best_match_distance:.2f})")
        else:
            results.append(f"Face {i + 1}: No match found (Below confidence threshold).")

    return results

# Draw bounding boxes and results on image
def draw_results_on_image(probe_image_path, results):
    image = cv2.imread(probe_image_path)
    face_locations = face_recognition.face_locations(face_recognition.load_image_file(probe_image_path))

    for (top, right, bottom, left), result in zip(face_locations, results):
        color = (0, 255, 0) if "Identified" in result else (0, 0, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process multiple probe images
def process_multiple_probes(probe_folder, known_encodings, known_names, threshold=0.6):
    all_results = {}
    for probe_image in os.listdir(probe_folder):
        probe_path = os.path.join(probe_folder, probe_image)
        results = identify_subjects_in_probe(probe_path, known_encodings, known_names, threshold)
        all_results[probe_image] = results
        logging.info(f"\nResults for {probe_image}:")
        for result in results:
            print(result)
        draw_results_on_image(probe_path, results)

    # Save results to a JSON file
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=4)

# Main function to parse arguments and execute
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition from Probe Images")
    parser.add_argument('--gallery', type=str, default="gallery", help="Path to gallery folder")
    parser.add_argument('--probe', type=str, default="probe.jpg", help="Path to probe image or folder")
    parser.add_argument('--threshold', type=float, default=0.6, help="Confidence threshold for identification")
    args = parser.parse_args()

    # Step 1: Load gallery images
    known_encodings, known_names = load_gallery_images(args.gallery)

    # Step 2: Process probe image(s)
    if os.path.isdir(args.probe):
        process_multiple_probes(args.probe, known_encodings, known_names, threshold=args.threshold)
    else:
        results = identify_subjects_in_probe(args.probe, known_encodings, known_names, threshold=args.threshold)
        for result in results:
            print(result)
        draw_results_on_image(args.probe, results)

# Run the script with the following command:
python registration.py --gallery gallery --probe probe.jpg --threshold 0.6