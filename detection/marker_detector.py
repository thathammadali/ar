"""Marker detection and pose estimation."""
import cv2
import numpy as np
from config import (
    MATCH_RATIO,
    MIN_MATCH_COUNT,
    MIN_INLIER_RATIO,
    RANSAC_THRESHOLD,
    ORB_FEATURES,
    ORB_FEATURES,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)


class MarkerDetector:
    """Handles marker detection and pose estimation."""
    
    def __init__(self):
        """Initialize marker detector with ORB features."""
        from config import MARKER_MAPPING
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(ORB_FEATURES)
        
        # Camera calibration matrix
        self.camera_matrix = np.array([
            [CAMERA_WIDTH, 0, CAMERA_WIDTH // 2],
            [0, CAMERA_WIDTH, CAMERA_HEIGHT // 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.dist_coeffs = np.zeros((4, 1))
        
        # Load all markers
        self.markers = {}
        self.use_cuda = False
        
        # Check CUDA availability once
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("CUDA device detected. Initializing CUDA ORB...")
                self.cuda_orb = cv2.cuda.ORB_create(ORB_FEATURES)
                self.cuda_matcher = cv2.cuda.DescriptorMatcher_createDFSMatcher(cv2.cuda.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
                self.use_cuda = True
                print("✓ CUDA initialized!")
        except Exception as e:
            print(f"CUDA init failed: {e}. Using CPU.")

        print("Loading markers...")
        for image_path, _model_file in MARKER_MAPPING.items():
            try:
                # Load image
                img = cv2.imread(image_path, 0)
                if img is None:
                    print(f"Warning: Marker image not found: {image_path}")
                    continue
                
                h, w = img.shape
                # Compute features
                kp, des = self.orb.detectAndCompute(img, None)
                
                marker_data = {
                    "width": w,
                    "height": h,
                    "kp": kp,
                    "des": des,
                    "gpu_marker": None,
                    "cuda_kp": None,
                    "cuda_des": None
                }
                
                # Upload to GPU if available
                if self.use_cuda:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(img)
                    c_kp, c_des = self.cuda_orb.detectAndComputeAsync(gpu_img, None)
                    marker_data["gpu_marker"] = gpu_img
                    marker_data["cuda_kp"] = c_kp
                    marker_data["cuda_des"] = c_des
                
                self.markers[image_path] = marker_data
                print(f"  ✓ Loaded {image_path}: {w}x{h}, {len(kp)} features")
                
            except Exception as e:
                print(f"Error loading marker {image_path}: {e}")

    def detect_marker(self, frame_gray):
        """Detect best matching marker in frame.
        
        Args:
            frame_gray: Grayscale frame from camera
        
        Returns:
            Tuple (Homography matrix, marker_name) if detected, else (None, None)
        """
        # Enhance contrast
        frame_enhanced = self.clahe.apply(frame_gray)
        
        best_homography = None
        best_name = None
        max_inliers = 0
        
        # We need frame features once
        if self.use_cuda:
             gpu_frame = cv2.cuda_GpuMat()
             gpu_frame.upload(frame_enhanced)
             frame_kp_gpu, frame_des = self.cuda_orb.detectAndComputeAsync(gpu_frame, None)
             # Convert KP to CPU for findingHomography later? OR detectAndComputeAsync returns (kp, des)
             # frame_kp_gpu is keypoints handle?
        else:
             frame_kp, frame_des = self.orb.detectAndCompute(frame_enhanced, None)
        
        # Iterate all markers to find best match
        for name, data in self.markers.items():
            homography = None
            inliers = 0
            
            if self.use_cuda:
                homography, inliers = self._match_marker_cuda(data, frame_kp_gpu, frame_des)
            else:
                homography, inliers = self._match_marker_cpu(data, frame_kp, frame_des)
            
            if homography is not None and inliers > max_inliers:
                max_inliers = inliers
                best_homography = homography
                best_name = name
                
        return best_homography, best_name

    def _match_marker_cpu(self, marker_data, kp_frame, des_frame):
        if des_frame is None or marker_data["des"] is None:
            return None, 0
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(marker_data["des"], des_frame, k=2)
        
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < MATCH_RATIO * n.distance:
                    good.append(m)
        
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([marker_data["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
            
            if H is not None:
                inliers = np.sum(mask)
                if inliers / len(good) > MIN_INLIER_RATIO:
                    # Convexity check
                    h, w = marker_data["height"], marker_data["width"]
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, H)
                    if cv2.isContourConvex(dst.astype(np.int32)):
                        return H, inliers
        return None, 0

    def _match_marker_cuda(self, marker_data, kp_frame_gpu, des_frame_gpu):
        if des_frame_gpu is None or des_frame_gpu.empty():
            return None, 0
            
        matches = self.cuda_matcher.knnMatch(marker_data["cuda_des"], des_frame_gpu, 2)
        
        good = []
        # Download matches to CPU list
        # matches is list of list of DMatch on CPU usually?
        # cv2.cuda.DescriptorMatcher returns list of lists of DMatch
        
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < MATCH_RATIO * n.distance:
                    good.append(m)
                    
        if len(good) > MIN_MATCH_COUNT:
            # We need keypoints on CPU for findHomography
            # marker_data["kp"] is already CPU
            # kp_frame_gpu needs conversion
            kp_frame_cpu = self.cuda_orb.convert(kp_frame_gpu)
            
            src_pts = np.float32([marker_data["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame_cpu[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
             
            if H is not None:
                inliers = np.sum(mask)
                if inliers / len(good) > MIN_INLIER_RATIO:
                     # Convexity check
                    h, w = marker_data["height"], marker_data["width"]
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, H)
                    if cv2.isContourConvex(dst.astype(np.int32)):
                        return H, inliers
        return None, 0

    def get_pose(self, homography, marker_name):
        """Estimate 3D pose for specific marker."""
        if marker_name not in self.markers:
            return None, None
            
        data = self.markers[marker_name]
        w, h = data["width"], data["height"]
        aspect_ratio = h / w
        
        obj_pts = np.array([
            [-0.5, 0.5 * aspect_ratio, 0],
            [ 0.5, 0.5 * aspect_ratio, 0],
            [ 0.5, -0.5 * aspect_ratio, 0],
            [-0.5, -0.5 * aspect_ratio, 0]
        ], dtype=np.float32)
        
        img_pts = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        img_pts = cv2.perspectiveTransform(img_pts, homography).reshape(-1, 2)
        
        _success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        return rvec, tvec
