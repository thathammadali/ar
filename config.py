"""Configuration constants for AR application."""

# ---------------- DETECTION PARAMETERS ----------------
# Adjust these values to fine-tune marker detection sensitivity
MIN_MATCH_COUNT = 15          # Minimum number of good matches required
MATCH_RATIO = 0.75            # Relaxed ratio test (0.75 from 0.7) to detect more matches
MIN_INLIER_RATIO = 0.4        # Slightly lower inlier ratio requirement
RANSAC_THRESHOLD = 5.0        # RANSAC reprojection threshold in pixels

# ---------------- PERFORMANCE PARAMETERS ----------------
CAMERA_WIDTH = 640            # Camera/display width (lower = faster)
CAMERA_HEIGHT = 480           # Camera/display height (lower = faster)
ORB_FEATURES = 2000           # Increased feature count (from 1000) for better detection
FRAME_SKIP = 1                # Process every Nth frame (higher = faster but less responsive)
MODEL_SCALE = 1.0             # Scale factor for 3D model (adjust if too big/small)
BASE_MODEL_SIZE = 0.25         # Model size relative to marker (0.5 = half marker size)
TEST_MODE = False             # Set to True to draw test cube at fixed position
POSITION_SMOOTHING = 0.2      # Position smoothing (0-1: lower = smoother, higher = more responsive)

# ---------------- FILE PATHS ----------------
# Define which model loads for which marker
MARKER_MAPPING = {
    "marker.jpeg": "Buddha.glb",
    "marker2.jpeg": "Stupid2.glb", 
    "marker3.jpeg": "staircase.glb", # Example second marker
     # Multiple markers can show same model
}

# Legacy single-file fallback (will be ignored by new app logic)
# MARKER_IMAGE = "marker.jpeg"
# MODEL_FILE = "Buddha.glb"
