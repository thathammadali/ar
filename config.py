"""Configuration constants for AR application."""

# ---------------- DETECTION PARAMETERS ----------------
# Adjust these values to fine-tune marker detection sensitivity
MIN_MATCH_COUNT = 20          # Minimum number of good matches required
MATCH_RATIO = 0.7             # Lowe's ratio test threshold (lower = stricter)
MIN_INLIER_RATIO = 0.5        # Minimum ratio of inliers from RANSAC (0.0-1.0)
RANSAC_THRESHOLD = 5.0        # RANSAC reprojection threshold in pixels

# ---------------- PERFORMANCE PARAMETERS ----------------
CAMERA_WIDTH = 640            # Camera/display width (lower = faster)
CAMERA_HEIGHT = 480           # Camera/display height (lower = faster)
ORB_FEATURES = 1000           # Number of ORB features (lower = faster)
FRAME_SKIP = 1                # Process every Nth frame (higher = faster but less responsive)
MODEL_SCALE = 1.0             # Scale factor for 3D model (adjust if too big/small)
BASE_MODEL_SIZE = 0.25         # Model size relative to marker (0.5 = half marker size)
TEST_MODE = False             # Set to True to draw test cube at fixed position
POSITION_SMOOTHING = 0.2      # Position smoothing (0-1: lower = smoother, higher = more responsive)

# ---------------- FILE PATHS ----------------
MARKER_IMAGE = "marker2.jpeg"  # Path to marker image
MODEL_FILE = "Stupid .glb"      # Path to 3D model (supports .obj, .gltf, .glb)
