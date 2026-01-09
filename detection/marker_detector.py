"""Marker detection and pose estimation."""
import cv2
import numpy as np
from config import (
    MATCH_RATIO,
    MIN_MATCH_COUNT,
    MIN_INLIER_RATIO,
    RANSAC_THRESHOLD,
    ORB_FEATURES,
    MARKER_IMAGE,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)


class MarkerDetector:
    """Handles marker detection and pose estimation."""
    
    def __init__(self):
        """Initialize marker detector with ORB features."""
        # Load marker image
        self.marker = cv2.imread(MARKER_IMAGE, 0)
        if self.marker is None:
            raise FileNotFoundError(f"Marker image not found: {MARKER_IMAGE}")
        
        print(f"Marker loaded: {self.marker.shape[1]}x{self.marker.shape[0]} pixels")
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(ORB_FEATURES)
        self.kp_marker, self.des_marker = self.orb.detectAndCompute(self.marker, None)
        self.marker_height, self.marker_width = self.marker.shape
        
        print(f"Marker keypoints detected: {len(self.kp_marker)}")
        
        # Camera calibration matrix
        self.camera_matrix = np.array([
            [CAMERA_WIDTH, 0, CAMERA_WIDTH // 2],
            [0, CAMERA_WIDTH, CAMERA_HEIGHT // 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))
    
    def detect_marker(self, frame_gray):
        """Detect marker in frame and compute homography.
        
        Args:
            frame_gray: Grayscale frame from camera
        
        Returns:
            Homography matrix if marker detected, None otherwise
        """
        kp_frame, des_frame = self.orb.detectAndCompute(frame_gray, None)
        if des_frame is None:
            return None
        
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf_matcher.knnMatch(self.des_marker, des_frame, k=2)
        
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < MATCH_RATIO * n.distance:
                    good.append(m)
        
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [self.kp_marker[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD
            )
            
            if homography is not None:
                # Check if we have enough inliers from RANSAC
                matches_mask = mask.ravel().tolist()
                inliers = sum(matches_mask)
                inlier_ratio = inliers / len(good)
                
                # Only accept if we have a good inlier ratio
                if inlier_ratio > MIN_INLIER_RATIO:
                    return homography
        
        return None
    
    def get_pose(self, homography):
        """Estimate 3D pose from homography matrix.
        
        Args:
            homography: Homography matrix from marker detection
        
        Returns:
            Tuple of (rotation vector, translation vector)
        """
        # Define marker in 3D world coordinates (normalized size)
        # Define marker in 3D world coordinates (normalized size)
        # Width = 1.0, Height = derived from aspect ratio
        # Origin at center of marker
        aspect_ratio = self.marker_height / self.marker_width
        
        # Define object points relative to center (0,0,0)
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        # Matches the order of img_pts below
        obj_pts = np.array(
            [
                [-0.5, 0.5 * aspect_ratio, 0],
                [0.5, 0.5 * aspect_ratio, 0],
                [0.5, -0.5 * aspect_ratio, 0],
                [-0.5, -0.5 * aspect_ratio, 0]
            ],
            dtype=np.float32
        )
        
        img_pts = np.array(
            [
                [0, 0],
                [self.marker_width, 0],
                [self.marker_width, self.marker_height],
                [0, self.marker_height]
            ],
            dtype=np.float32
        ).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(img_pts, homography).reshape(-1, 2)
        
        _success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self.camera_matrix, self.dist_coeffs
        )
        
        return rvec, tvec
