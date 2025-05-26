import subprocess
import os

def launch_streamlit():
    # Get the path to the streamlit_app.py file
    streamlit_app_path = os.path.join(os.getcwd(), "streamlit_app.py")
    # Launch the Streamlit app
    subprocess.Popen(["streamlit", "run", streamlit_app_path])

# Launch the Streamlit app when the Space is loaded
launch_streamlit()