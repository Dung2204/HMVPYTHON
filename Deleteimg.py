import os
import streamlit as st
from rembg import remove
from PIL import Image
import io
import cv2
import numpy as np

def run_DeleteBackground_app():
    # Function to process images
    def process_image(image):
        try:
            output_image = remove(image)
            if output_image.mode == "RGBA":
                output_image = output_image.convert("RGB")
            return output_image
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None

    # Function to process videos
    def process_video(video_path, output_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Unable to open video!")
                return False

            # Get video information
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                # Process background removal
                processed_frame = remove(pil_frame)
                if processed_frame.mode == "RGBA":
                    processed_frame = processed_frame.convert("RGB")
                # Convert back to numpy array
                processed_frame_np = np.array(processed_frame)
                processed_frame_bgr = cv2.cvtColor(processed_frame_np, cv2.COLOR_RGB2BGR)
                # Write frame to video
                out.write(processed_frame_bgr)

            cap.release()
            out.release()
            return True
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return False

    # Initialize session state
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'show_result' not in st.session_state:
        st.session_state.show_result = False

    # Custom CSS for professional styling
    st.markdown("""
        <style>
        .main-header {
            font-family: 'Arial', sans-serif;
            color: #1E3A8A;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .section-header {
            font-family: 'Arial', sans-serif;
            color: #3B82F6;
            font-size: 1.5em;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .card {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #3B82F6;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: 500;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #1E40AF;
        }
        .footer {
            text-align: center;
            color: #6B7280;
            font-size: 0.9em;
            margin-top: 30px;
        }
        .stFileUploader {
            border: 2px dashed #D1D5DB;
            border-radius: 5px;
            padding: 10px;
        }
        .stSuccess {
            background-color: #E7F3FE;
            color: #1E3A8A;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # App title
    st.markdown('<div class="main-header">🖼️ Background Removal Tool</div>', unsafe_allow_html=True)
    st.markdown("---")

    # If showing results
    if st.session_state.show_result:
        with st.container():
            with st.container():
                if st.session_state.get("image_processed", None) is not None:
                    st.image(st.session_state.image_processed, caption="Processed Image", use_container_width=True)
                    img_byte_arr = io.BytesIO()
                    st.session_state.image_processed.save(img_byte_arr, format="JPEG")
                    img_byte_arr = img_byte_arr.getvalue()
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=img_byte_arr,
                        file_name="output_image.jpg",
                        mime="image/jpeg",
                        key="download_image",
                        use_container_width=True
                    )
                elif st.session_state.get("video_processed", None) is not None:
                    with open(st.session_state.video_processed, "rb") as f:
                        st.video(f)
                    with open(st.session_state.video_processed, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="output_video.mp4",
                        mime="video/mp4",
                        key="download_video",
                        use_container_width=True
                    )
                    if os.path.exists(st.session_state.video_processed):
                        os.remove(st.session_state.video_processed)
                st.markdown('</div>', unsafe_allow_html=True)

            # Back button
            if st.button("🔙 Back", key="back_button", help="Return to upload another file"):
                st.session_state.show_result = False
                st.session_state.data_loaded = False
                st.session_state.uploaded_file = None
                st.session_state.image_processed = None
                st.session_state.video_processed = None
                st.rerun()

    # If not showing results, display upload and process interface
    else:
        # Upload section
        with st.container():
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload an image (PNG, JPG, JPEG, BMP, TIFF) or video (MP4, AVI)",
                    type=["png", "jpg", "jpeg", "bmp", "tiff", "mp4", "avi"],
                    help="Drag and drop your file here or click to browse"
                )
                if uploaded_file is not None:
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.data_loaded = True
                    file_type = uploaded_file.type
                    if file_type.startswith("image"):
                        st.image(uploaded_file, caption="Original Image", use_container_width=True)
                    elif file_type.startswith("video"):
                        st.video(uploaded_file)
                    st.success("Data uploaded successfully!")
                st.markdown('</div>', unsafe_allow_html=True)

        # Process section (only shown if data is uploaded)
        if st.session_state.data_loaded:
            with st.container():
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    if st.button("Remove Background", key="process_button", help="Click to remove the background"):
                        uploaded_file = st.session_state.uploaded_file
                        file_type = uploaded_file.type

                        # Process image
                        if file_type.startswith("image"):
                            image = Image.open(uploaded_file)
                            st.session_state.image_processed = process_image(image)
                            if st.session_state.image_processed:
                                st.session_state.show_result = True
                                st.rerun()
                            else:
                                st.error("Image processing failed!")

                        # Process video
                        elif file_type.startswith("video"):
                            temp_input_path = "temp_input_video." + uploaded_file.name.split('.')[-1]
                            temp_output_path = "temp_output_video.mp4"
                            with open(temp_input_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            if process_video(temp_input_path, temp_output_path):
                                st.session_state.video_processed = temp_output_path
                                st.session_state.show_result = True
                                st.rerun()
                            else:
                                st.error("Video processing failed!")
                            
                            if os.path.exists(temp_input_path):
                                os.remove(temp_input_path)
                    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="footer">Background Removal App - Powered by Dung</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    run_DeleteBackground_app()
