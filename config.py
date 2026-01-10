"""Configuration constants for AR application."""

# ---------------- DETECTION PARAMETERS ----------------
# Adjust these values to fine-tune marker detection sensitivity
MIN_MATCH_COUNT = 8           # Ultra-sensitive (was 10)
MATCH_RATIO = 0.85            # Accepts weaker matches (was 0.8)
MIN_INLIER_RATIO = 0.15       # Minimal structural consistency (was 0.25)
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

# ============= TRACKING MODE PARAMETERS =============
TRACKING_MIN_MATCH_COUNT = 6
TRACKING_MATCH_RATIO = 0.8
TRACKING_MIN_INLIER_RATIO = 0.2
TRACKING_RANSAC_THRESHOLD = 8.0
TRACKING_MIN_QUALITY = 2           # Keep lock (was 4)

# ============= OPTICAL FLOW PARAMETERS =============
OPTICAL_FLOW_WIN_SIZE = 21
OPTICAL_FLOW_MAX_LEVEL = 3
OPTICAL_FLOW_MAX_CORNERS = 100
OPTICAL_FLOW_QUALITY = 0.01
OPTICAL_FLOW_MIN_DISTANCE = 10
OPTICAL_FLOW_MIN_POINTS = 10

# ============= LOCK-ON PARAMETERS =============
MAX_REPROJECTION_ERROR = 8.0       # Relaxed to 8.0 (was 3.0) for uncalibrated cameras

# ---------------- FILE PATHS ----------------
# Define which model loads for which marker
MARKER_MAPPING = {
    "marker3.jpeg": "Buddha.glb",
    # "marker2.jpeg": "Stupid2.glb", 
    # "marker.jpeg": "staircase.glb", # Example second marker
     # Multiple markers can show same model
}

# Legacy single-file fallback (will be ignored by new app logic)
# MARKER_IMAGE = "marker.jpeg"
# MODEL_FILE = "Buddha.glb"
