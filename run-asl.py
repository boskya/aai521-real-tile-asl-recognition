# asl_app.py
import os

# Disable GPU and threading
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Now import TensorFlow
import tensorflow as tf

# Force CPU only
tf.config.set_visible_devices([], 'GPU')

# Single thread
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Disable oneDNN optimizations (causes crashes on some Macs)
tf.config.optimizer.set_jit(False)

import tensorflow as tf
# Configure TensorFlow to use single thread
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import mediapipe as mp
from collections import deque, Counter
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="wide"
)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}

@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.applications import EfficientNetB0
        
        st.info("Building model with ImageNet base...")
        
        # Build with ImageNet weights (important!)
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',  # Load pretrained ImageNet weights
            input_shape=(224, 224, 3)
        )
        
        # Build full model
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(29, activation='softmax')
        ])
        
        # Build the model
        model.build((None, 224, 224, 3))
        
        st.info("Loading trained weights...")
        
        # Load ALL weights (will overwrite ImageNet with trained)
        model.load_weights('asl_model_weights.weights.h5')
        
        st.success("✓ Model loaded!")
        return model
        
    except Exception as e:
        st.error(f"Error: {e}")
        
        # Try alternative: load full model directly
        try:
            st.info("Trying direct load...")
            model = tf.keras.models.load_model('asl_model_weights.weights.h5')
            return model
        except:
            pass
            
        return None


@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands, mp_hands, mp_drawing

# Classes
CLASSES = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                  'Y', 'Z', 'del', 'nothing', 'space'])

# Preprocessing functions
def get_hand_bbox(hand_landmarks, frame_width, frame_height, padding_percent=1.0):
    """
    Get bounding box with GENEROUS padding (like training images)
    Training images have hand filling only ~40-50% of frame
    padding_percent=1.0 means add 100% of hand size as padding on each side
    """
    x_coords = [landmark.x * frame_width for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * frame_height for landmark in hand_landmarks.landmark]
    
    # Get tight bounds
    x_min_tight = int(min(x_coords))
    x_max_tight = int(max(x_coords))
    y_min_tight = int(min(y_coords))
    y_max_tight = int(max(y_coords))
    
    # Calculate hand dimensions
    hand_width = x_max_tight - x_min_tight
    hand_height = y_max_tight - y_min_tight
    
    # Add MUCH MORE padding (100% = double the space on each side!)
    padding_x = int(hand_width * padding_percent)
    padding_y = int(hand_height * padding_percent)
    
    x_min = max(0, x_min_tight - padding_x)
    x_max = min(frame_width, x_max_tight + padding_x)
    y_min = max(0, y_min_tight - padding_y)
    y_max = min(frame_height, y_max_tight + padding_y)
    
    # Make it square by expanding smaller dimension
    width = x_max - x_min
    height = y_max - y_min
    
    if width > height:
        diff = width - height
        y_min = max(0, y_min - diff // 2)
        y_max = min(frame_height, y_max + diff // 2)
        
        # Handle boundary
        if (y_max - y_min) < width:
            if y_min == 0:
                y_max = min(frame_height, y_min + width)
            elif y_max == frame_height:
                y_min = max(0, y_max - width)
    else:
        diff = height - width
        x_min = max(0, x_min - diff // 2)
        x_max = min(frame_width, x_max + diff // 2)
        
        # Handle boundary
        if (x_max - x_min) < height:
            if x_min == 0:
                x_max = min(frame_width, x_min + height)
            elif x_max == frame_width:
                x_min = max(0, x_max - height)
    
    return x_min, y_min, x_max, y_max

def segment_hand(img_bgr, hand_landmarks):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    points = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append([x, y])
    
    points = np.array(points)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    hand_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    background = np.ones_like(img_bgr) * 255
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background, background, mask=mask_inv)
    hand_with_white_bg = cv2.add(hand_only, background)
    
    return hand_with_white_bg, mask

def preprocess_hand_region(hand_region):
    img_resized = cv2.resize(hand_region, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

def predict_sign(model, hands, mp_hands, mp_drawing, frame, use_segmentation=True):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            h, w = frame.shape[:2]
            x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, w, h, padding_percent=1.8)
            
            # Draw bounding box
            cv2.rectangle(frame_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            if use_segmentation:
                hand_segmented, _ = segment_hand(frame, hand_landmarks)
                hand_region = hand_segmented[y_min:y_max, x_min:x_max]
            else:
                hand_region = frame[y_min:y_max, x_min:x_max]

            if hand_region is not None:
                st.sidebar.write("### Debug: Hand Region")
                st.sidebar.image(hand_region, caption=f"Cropped: {hand_region.shape}", channels="BGR")
    
                resized = cv2.resize(hand_region, (224, 224))
                st.sidebar.image(resized, caption="Resized to 224x224", channels="BGR")
            
            if hand_region.size > 0:
                processed = preprocess_hand_region(hand_region)
                prediction = model.predict(processed, verbose=0)
                predicted_idx = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_class = CLASSES[predicted_idx]
                
                # Get top 5
                top_5_idx = np.argsort(prediction[0])[-5:][::-1]
                top_5 = [(CLASSES[idx], prediction[0][idx]) for idx in top_5_idx]
                
                return frame_rgb, predicted_class, confidence, top_5, hand_region
    
    return frame_rgb, None, None, None, None

# Streamlit UI
st.title("🤟 Real-Time ASL Alphabet Recognition")
st.markdown("### University of San Diego - AAI-521 Computer Vision Project")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    model_choice = st.selectbox(
        "Select Model",
        ["Model 2 (Augmented + Fine-tuned)", "Model 1 (Baseline)"]
    )
    
    use_segmentation = st.checkbox("Use Background Segmentation", value=True)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    This application demonstrates real-time ASL alphabet recognition using:
    - **Transfer Learning** with EfficientNet-B0
    - **MediaPipe Hands** for hand detection
    - **Background Segmentation** for robustness
    
    **Instructions:**
    1. Allow camera access
    2. Show ASL alphabet signs to the camera
    3. Keep hand visible and well-lit
    """)

# Load model and MediaPipe
model = load_model()
hands, mp_hands, mp_drawing = initialize_mediapipe()

if model is None:
    st.error("Failed to load model. Please check the model path.")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📷 Live Recognition", "📊 Testing Protocol", "📈 Results", 'Debug'])

with tab1:
    st.header("Live Camera Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Camera Feed")
        
        # Use st.camera_input instead of cv2.VideoCapture
        camera_photo = st.camera_input("Take a photo")
        
        if camera_photo is not None:
            # Convert to OpenCV format
            bytes_data = camera_photo.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame
            frame_display, predicted_class, confidence, top_5, hand_region = predict_sign(
                model, hands, mp_hands, mp_drawing, frame, use_segmentation
            )
            
            # Display
            st.image(frame_display, channels="RGB")
            
            # Show results
            if predicted_class:
                with col2:
                    st.markdown("### Prediction")
                    st.metric("Sign", predicted_class)
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    st.markdown("### Top 5 Predictions")
                    if top_5:
                        for i, (cls, conf) in enumerate(top_5, 1):
                            st.text(f"{i}. {cls:8s} {conf:.1%}")
    
    with col2:
        st.markdown("### Instructions")
        st.info("""
        - Position hand at arm's length
        - Face palm toward camera
        - Use good lighting
        - Plain background works best
        """)

with tab2:
    st.header("Structured Testing Protocol")
    st.markdown("""
    Test each letter systematically to evaluate model performance on webcam data.
    """)
    
    # Testing interface
    test_letter = st.selectbox("Select letter to test", CLASSES)
    num_attempts = st.number_input("Number of attempts", min_value=1, max_value=10, value=5)
    
    if test_letter not in st.session_state.test_results:
        st.session_state.test_results[test_letter] = []
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Correct"):
            st.session_state.test_results[test_letter].append(True)
            st.success(f"Recorded: {test_letter} - Correct")
    
    with col2:
        if st.button("❌ Incorrect"):
            st.session_state.test_results[test_letter].append(False)
            st.error(f"Recorded: {test_letter} - Incorrect")
    
    with col3:
        if st.button("🔄 Reset Letter"):
            st.session_state.test_results[test_letter] = []
            st.info(f"Reset results for {test_letter}")
    
    # Display current letter results
    if st.session_state.test_results[test_letter]:
        results = st.session_state.test_results[test_letter]
        correct = sum(results)
        total = len(results)
        accuracy = correct / total * 100
        
        st.markdown(f"### Results for {test_letter}")
        st.metric("Attempts", total)
        st.metric("Correct", correct)
        st.metric("Accuracy", f"{accuracy:.1f}%")
        st.progress(accuracy / 100)

with tab3:
    st.header("Testing Results Summary")
    
    if st.session_state.test_results:
        # Calculate overall statistics
        total_correct = 0
        total_attempts = 0
        
        results_data = []
        
        for letter in sorted(st.session_state.test_results.keys()):
            results = st.session_state.test_results[letter]
            if results:
                correct = sum(results)
                total = len(results)
                accuracy = correct / total * 100
                
                total_correct += correct
                total_attempts += total
                
                results_data.append({
                    'Letter': letter,
                    'Correct': correct,
                    'Total': total,
                    'Accuracy': f"{accuracy:.1f}%"
                })
        
        if results_data:
            import pandas as pd
            df = pd.DataFrame(results_data)
            
            st.dataframe(df, use_container_width=True)
            
            overall_accuracy = total_correct / total_attempts * 100 if total_attempts > 0 else 0
            
            st.markdown("### Overall Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Attempts", total_attempts)
            with col2:
                st.metric("Correct Predictions", total_correct)
            with col3:
                st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
            
            # Export results
            if st.button("📥 Export Results as CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="asl_webcam_test_results.csv",
                    mime="text/csv"
                )
    else:
        st.info("No test results yet. Use the Testing Protocol tab to record results.")
    
    # Reset all
    if st.button("🗑️ Clear All Results"):
        st.session_state.test_results = {}
        st.success("All results cleared!")
        st.rerun()

with tab4:
    st.header("Debug Model")
        
    uploaded = st.file_uploader("Upload training image (known letter)", type=['jpg', 'png'])
    
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(img, caption="Original", channels="BGR")
        
        with col2:
            # Show what gets fed to model
            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="After resize + RGB")
        
        with col3:
            # Preprocess and predict
            img_array = np.expand_dims(img_rgb, axis=0).astype('float32')
            img_preprocessed = preprocess_input(img_array)
            
            pred = model.predict(img_preprocessed, verbose=0)
            predicted_class = CLASSES[np.argmax(pred)]
            confidence = np.max(pred)
            
            st.metric("Prediction", predicted_class)
            st.metric("Confidence", f"{confidence:.1%}")
            
            st.write("Top 5:")
            top_5 = np.argsort(pred[0])[-5:][::-1]
            for i, idx in enumerate(top_5, 1):
                st.write(f"{i}. {CLASSES[idx]}: {pred[0][idx]:.1%}")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Bosky Atlani | University of San Diego | AAI-521 Computer Vision</p>
    <p>Project: Real-Time ASL Alphabet Recognition</p>
</div>
""", unsafe_allow_html=True)