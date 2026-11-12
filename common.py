import cv2
import numpy as np

# State Farm 10-Class Definitions
CLASS_LABELS = {
    0: "Safe driving",
    1: "Texting (right)",
    2: "Talking on phone (right)",
    3: "Texting (left)",
    4: "Talking on phone (left)",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger"
}

def run_live_inference(model, window_name="Distracted Driver Monitor"):
    """Captures live webcam feed, runs inference using the loaded model, and displays results."""
    cap = cv2.VideoCapture(0)
    print(f"Starting {window_name}... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess: BGR -> RGB -> 224x224
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        input_tensor = np.expand_dims(resized_frame, axis=0).astype(np.float32)
        
        # Inference
        predictions = model.predict(input_tensor, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Label and Color (Green = Safe, Red = Distracted)
        label = f"{CLASS_LABELS[class_idx]}: {confidence * 100:.1f}%"
        color = (0, 255, 0) if class_idx == 0 else (0, 0, 255)
        
        # Display on original frame resolution
        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()