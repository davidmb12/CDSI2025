import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib
from scipy.stats import mode
from imgaug import augmenters as iaa

labels_definition = {
    "Complete":[1,0,0],
    "Incomplete":[0,1,0],
    "Null":[0,0,1]
}
value_to_label = {
    (1,0,0):"Complete",
    (0,1,0):"Incomplete",
    (0,0,1):"Null",
}
def load_video_files(root_folder):
    video_files = []
    labels = []
    # os.walk traverses the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # Check if the file extension matches (case-insensitive)
            label =(str.split(filename,"-")[1])
            label_val =(str.split(label,".")[0])
            labels.append(labels_definition[label_val])
            video_files.append(os.path.join(dirpath, filename))
    result = list(zip(labels,video_files))
    return result



def format_landmarks(landmarks):
    keys = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
    formatted = []
    for key in keys:
        if key in landmarks:
            formatted.extend(landmarks[key])
        else:
            formatted.extend([0, 0])  # Pad missing key with zeros
    return np.array(formatted)


def data_collection(video_path, num_frames=256, frame_skip=10):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    landmarks_list = []
    frame_count = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Jump to the specific frame
        ret, frame = cap.read()
        if not ret or len(landmarks_list) >= num_frames:
            break

        frame_count += frame_skip  # Skip frames based on frame_skip

        # Convert to RGB and process pose
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            formatted = np.array([
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x / frame_width,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y / frame_height,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x / frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y / frame_height,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x / frame_width,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y / frame_height,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x / frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y / frame_height,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x / frame_width,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y / frame_height,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x / frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y / frame_height
            ])
            landmarks_list.append(formatted)

    cap.release()
    pose.close()
    return np.array(landmarks_list)

def extract_landmarks(video_paths):
    data = []
    labels = []
    for label, video_path in video_paths:
        try:
            landmarks = data_collection(video_path)
            if len(landmarks) > 0:  # Skip videos with no landmarks
                data.append(landmarks)
                labels.extend([label] * len(landmarks))
        except Exception as e:
            print(f"Error extracting landmarks from {video_path}: {e}")

    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels


def augment_frames(frames):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0,1.0)),
        iaa.Add((-10,10)),
        iaa.Multiply((0.8,1.2)),
    ])
    return seq(images=frames)

def save_video(frames, output_path, fps=256):
    # Get frame dimensions
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the output video
    for frame in frames:
        frame = cv2.resize(frame,(width,height))
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    
    
def build_model(data,labels):
    x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size =0.2,random_state=42)
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    predict = model.predict(x_test)
    accuracy = accuracy_score(y_test,predict)
    print(f"Precision del modelo: {accuracy * 100:.2f}%")
    joblib.dump(model,'video_recognition_03.pkl')
    return model

def validate_model(model, video_path):
    # Collect landmarks data from the video
    n_data = data_collection(video_path)

    # Predict using the trained model
    predictions = model.predict(n_data)

    # Map predictions to their corresponding labels
    predicted_labels = []
    for pred in predictions:
        pred_label = tuple(pred)
        if pred_label in value_to_label:
            predicted_labels.append(pred_label)

    # Handle case where no matches are found
    if not predicted_labels:
        print("No matches found.")
        return

    # Determine the most common prediction (mode)
    final_label = mode(predicted_labels).mode

    # Convert to human-readable label
    print("Predicted Label:", value_to_label[tuple(final_label)])
    
def load_model(model_path):
    model = joblib.load(model_path)
    return model


if __name__ == "__main__":
    #Retrain model 
    # folder_path = './Videos/'
    # output_path = './Videos/_Augmented/'
    
    # videos = load_video_files(folder_path)
    # data,labels = extract_landmarks(videos)
    
    # model = build_model(data,labels)
    
    #Load and test model
    model = load_model('./video_recognition_03.pkl')
    #Add validation videos path
    validate_model(model,"../../../../../Desktop/Videos/Daniel_02.mp4")