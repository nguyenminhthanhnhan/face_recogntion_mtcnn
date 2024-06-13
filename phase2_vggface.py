import cv2
import numpy as np
import sqlite3
from deepface import DeepFace
from datetime import datetime
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Connect to the SQLite database
conn = sqlite3.connect("face_encodings_vggface.db")
cursor = conn.cursor()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Initialize the set to keep track of recognized names
sent_names = set()

def calculate_similarity(embedding1, embedding2):
    # Convert the embeddings to NumPy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Calculate the cosine similarity between the embeddings
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

def calculate_confidence(similarity):
    # Calculate the confidence score based on the cosine similarity
    confidence = similarity
    return confidence

def recognize_face(face_embedding):
    # Retrieve all face encodings and names from the database
    cursor.execute("SELECT encoding, name FROM face_encodings_deepface")
    rows = cursor.fetchall()
    max_similarity = -1
    recognized_name = "Unknown"
    # Iterate over the retrieved rows
    for row in rows:
        # Retrieve the stored embedding and name
        stored_embedding = np.frombuffer(row[0], dtype=np.float64)
        stored_name = row[1]
        # Calculate the cosine similarity between the face embedding from the camera and the stored embedding
        similarity = calculate_similarity(face_embedding, stored_embedding)
        # Update the recognized name if the similarity is greater than the maximum similarity so far
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = stored_name
    # Calculate the confidence score based on the similarity
    confidence = calculate_confidence(max_similarity)
    # Adjust the threshold for considering a face as "Unknown"
    if confidence < 0.6:  # Adjust the threshold as needed
        recognized_name = "Unknown"
    return recognized_name, confidence

# Open the default camera (change index if you have multiple cameras)
camera = cv2.VideoCapture(0) 
# Load the VGG-Face model for face recognition
model = DeepFace.build_model("VGG-Face")

# Initialize the face detector from MTCNN
face_detector = MTCNN()

while True:
    # Read the camera frame
    ret, frame = camera.read()

    # Check if the frame is empty
    if not ret or frame is None:
        print("Error: Could not read frame from the camera.")
        break

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using MTCNN
    faces = face_detector.detect_faces(frame_rgb)

    # Iterate over the detected faces
    for face in faces:
        # Extract the bounding box coordinates
        x, y, w, h = face['box']

        # Align the face region using landmarks (if needed)
        # ...

        # Crop the face region from the frame
        face_region = frame_rgb[y:y+h, x:x+w]

        # Generate the 128-dimensional face embedding using the VGG-Face model
        embedding = DeepFace.represent(face_region, model_name="VGG-Face", enforce_detection=False)

        # Check if any embeddings were generated
        if len(embedding) > 0:
            face_embedding = embedding[0]["embedding"]

            # Recognize the face by comparing the embedding with the stored encodings
            recognized_name, confidence = recognize_face(face_embedding)

            if recognized_name:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display the recognized name and confidence above the face rectangle
                text = f"{recognized_name}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Log information (can be modified based on your requirements)
                if recognized_name not in sent_names and recognized_name != "Unknown":
                    # Write down the current time
                    current_time = now.strftime("%H:%M:%S")

                    print(f"Recognized: {recognized_name}, Confidence: {confidence:.2f}, Time: {current_time}")

                    sent_names.add(recognized_name)

    # Display the frame with recognized faces
    resize_frame = cv2.resize(frame, (640, 480))  # Set the desired window size
    cv2.imshow("Face Recognition", resize_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the database connection
camera.release()
cv2.destroyAllWindows()
conn.close()
