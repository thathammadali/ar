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
        
        # Try to initialize CUDA
        self.use_cuda = False
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("CUDA device detected. Attempting to initialize CUDA ORB...")
                self.cuda_orb = cv2.cuda.ORB_create(ORB_FEATURES)
                self.cuda_matcher = cv2.cuda.DescriptorMatcher_createDFSMatcher(cv2.cuda.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
                
                # Upload marker to GPU for matching
                self.gpu_marker = cv2.cuda_GpuMat()
                self.gpu_marker.upload(self.marker)
                self.cuda_kp_marker, self.cuda_des_marker = self.cuda_orb.detectAndComputeAsync(self.gpu_marker, None)
                
                self.use_cuda = True
                print("✓ CUDA ORB initialized successfully!")
            else:
                print("CUDA not available (no device or module). Using CPU.")
        except AttributeError as e:
            print(f"CUDA module present but ORB_create missing: {e}. Using CPU.")
            self.use_cuda = False
        except Exception as e:
            print(f"Error initializing CUDA: {e}. Using CPU.")
            self.use_cuda = False

        # Camera calibration matrix
        self.camera_matrix = np.array([
            [CAMERA_WIDTH, 0, CAMERA_WIDTH // 2],
            [0, CAMERA_WIDTH, CAMERA_HEIGHT // 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        
        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.dist_coeffs = np.zeros((4, 1))
    
    def detect_marker(self, frame_gray):
        """Detect marker in frame and compute homography.
        
        Args:
            frame_gray: Grayscale frame from camera
        
        Returns:
            Homography matrix if marker detected, None otherwise
        """
        # Enhance contrast using CLAHE
        # This helps significantly with lighting variations and low-contrast markers
        frame_enhanced = self.clahe.apply(frame_gray)
        
        if self.use_cuda:
            return self._detect_marker_cuda(frame_enhanced)
        else:
            return self._detect_marker_cpu(frame_enhanced)

    def _detect_marker_cuda(self, frame_gray):
        """CUDA-accelerated detection implementation."""
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame_gray)
            
            # Detect features on GPU
            cuda_kp_frame, cuda_des_frame = self.cuda_orb.detectAndComputeAsync(gpu_frame, None)
            
            if cuda_des_frame is None or cuda_des_frame.empty():
                return None
                
            # Match descriptors on GPU
            matches = self.cuda_matcher.knnMatch(self.cuda_des_marker, cuda_des_frame, 2)
            
            # Download results to CPU for filtering / homography
            # Note: methods like download depend on OpenCV version, often returns list of lists on CPU directly
            # or we need to handle GpuMat matches. 
            # Usually knnMatch with generic matcher returns DMatches on CPU.
            
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < MATCH_RATIO * n.distance:
                        good.append(m)
            
            if len(good) > MIN_MATCH_COUNT:
                # Convert keypoints from GPU/Object format if needed
                # cv2.cuda.ORB returns keypoints differently? 
                # Actually detectAndComputeAsync returns (keypoints, descriptors)
                # Keypoints are usually on CPU, descriptors on GPU.
                # Let's verify API. keypoints are usually [cv2.KeyPoint] list on CPU.
                
                # However, the cuda_kp_frame might be GpuMat in some versions?
                # Standard python bindings: keypoints are returned as list of KeyPoint (CPU).
                # Descriptors are GpuMat.
                
                kp_frame_cpu = self.cuda_orb.convert(cuda_kp_frame) # Convert keypoints to CPU list
                
                src_pts = np.float32(
                    [self.kp_marker[m.queryIdx].pt for m in good] # Use CPU marker keypoints
                ).reshape(-1, 1, 2)
                
                dst_pts = np.float32(
                    [kp_frame_cpu[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                
                homography, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD
                )
                
                if homography is not None:
                    matches_mask = mask.ravel().tolist()
                    inliers = sum(matches_mask)
                    inlier_ratio = inliers / len(good)
                    
                    if inlier_ratio > MIN_INLIER_RATIO:
                        # Validate homography by checking if projected marker is convex
                        h, w = self.marker_height, self.marker_width
                        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts, homography)
                        
                        if cv2.isContourConvex(dst.astype(np.int32)):
                            return homography
                        else:
                            # print("Rejected non-convex homography")
                            pass
            return None
            
        except Exception as e:
            print(f"CUDA Error: {e}. Fallback to CPU.")
            self.use_cuda = False # Disable CUDA if runtime error
            return self._detect_marker_cpu(frame_gray)

    def _detect_marker_cpu(self, frame_gray):
        """Standard CPU detection implementation."""
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
                    # Validate homography by checking if projected marker is convex
                    h, w = self.marker_height, self.marker_width
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, homography)
                    
                    if cv2.isContourConvex(dst.astype(np.int32)):
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
        
        # DEBUG: Check shapes before solvePnP
        # print(f"DEBUG: obj_pts shape={obj_pts.shape}, img_pts shape={img_pts.shape}")
        
        _success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        
        return rvec, tvec
