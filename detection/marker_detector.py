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

    def detect_specific_marker(self, frame_gray, marker_name):
        """Detect ONLY the specified marker - used in TRACKING mode."""
        from config import (TRACKING_MIN_MATCH_COUNT, TRACKING_MATCH_RATIO,
                           TRACKING_MIN_INLIER_RATIO, TRACKING_RANSAC_THRESHOLD,
                           TRACKING_MIN_QUALITY, MAX_REPROJECTION_ERROR)
        
        if marker_name not in self.markers:
            return None, 0
        
        # Enhance contrast
        frame_enhanced = self.clahe.apply(frame_gray)
        
        # Extract features
        frame_kp, frame_des = self.orb.detectAndCompute(frame_enhanced, None)
        
        if frame_des is None:
            return None, 0
        
        data = self.markers[marker_name]
        if data["des"] is None:
            return None, 0
        
        # Match features with RELAXED thresholds
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(data["des"], frame_des, k=2)
        
        good = []
        match_distances = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < TRACKING_MATCH_RATIO * n.distance:
                    good.append(m)
                    match_distances.append(m.distance)
        
        if len(good) > TRACKING_MIN_MATCH_COUNT:
            src_pts = np.float32([data["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, TRACKING_RANSAC_THRESHOLD)
            
            if H is not None:
                inliers = np.sum(mask)
                inlier_ratio = inliers / len(good)
                
                if inlier_ratio > TRACKING_MIN_INLIER_RATIO:
                    # Convexity check
                    h, w = data["height"], data["width"]
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, H)
                    
                    if cv2.isContourConvex(dst.astype(np.int32)):
                        # Calculate quality score
                        avg_match_dist = np.mean(match_distances) if match_distances else 100
                        
                        # Calculate reprojection error
                        inlier_src = src_pts[mask.ravel() == 1]
                        inlier_dst = dst_pts[mask.ravel() == 1]
                        projected = cv2.perspectiveTransform(inlier_src, H)
                        reproj_errors = np.linalg.norm(projected - inlier_dst, axis=2).flatten()
                        avg_reproj_error = np.mean(reproj_errors)
                        
                        if avg_reproj_error > MAX_REPROJECTION_ERROR:
                            return None, 0
                        
                        # Quality score
                        norm_dist = avg_match_dist / 100.0
                        norm_reproj = avg_reproj_error / MAX_REPROJECTION_ERROR
                        quality_score = (inliers * inlier_ratio) / (norm_dist + norm_reproj + 0.1)
                        
                        if quality_score >= TRACKING_MIN_QUALITY:
                            return H, quality_score
        
        return None, 0

    def init_optical_flow_tracking(self, frame_gray, homography, marker_name):
        """Initialize good features to track for optical flow."""
        from config import (OPTICAL_FLOW_MAX_CORNERS, OPTICAL_FLOW_QUALITY,
                           OPTICAL_FLOW_MIN_DISTANCE)
        
        if marker_name not in self.markers:
            return None
        
        data = self.markers[marker_name]
        w, h = data["width"], data["height"]
        
        # Get marker corners in current frame
        obj_corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        img_corners = cv2.perspectiveTransform(obj_corners, homography)
        
        # Define marker region
        mask = np.zeros(frame_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [img_corners.astype(np.int32)], 255)
        
        # Detect good features within marker region
        good_features = cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=OPTICAL_FLOW_MAX_CORNERS,
            qualityLevel=OPTICAL_FLOW_QUALITY,
            minDistance=OPTICAL_FLOW_MIN_DISTANCE,
            mask=mask
        )
        
        return good_features

    def track_with_optical_flow(self, prev_gray, curr_gray, prev_points, marker_name):
        """Track using Optical Flow (Fast & Smooth)."""
        from config import (OPTICAL_FLOW_WIN_SIZE, OPTICAL_FLOW_MAX_LEVEL,
                           OPTICAL_FLOW_MIN_POINTS)
        
        if prev_points is None or len(prev_points) < OPTICAL_FLOW_MIN_POINTS:
            return False, None, None
        
        # Lucas-Kanade optical flow
        lk_params = dict(
            winSize=(OPTICAL_FLOW_WIN_SIZE, OPTICAL_FLOW_WIN_SIZE),
            maxLevel=OPTICAL_FLOW_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **lk_params
        )
        
        if next_points is None or status is None:
            return False, None, None
        
        # Filter good points
        good_new = next_points[status.ravel() == 1]
        good_old = prev_points[status.ravel() == 1]
        
        if len(good_new) < OPTICAL_FLOW_MIN_POINTS:
            return False, None, None
        
        # Compute homography
        try:
            H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
            if H is None:
                return False, None, None
            
            # Convexity check on transformed marker bounds if needed (omitted for speed/H-validation covers it)
            return True, good_new.reshape(-1, 1, 2), H
            
        except Exception:
            return False, None, None

    def get_pose(self, homography, marker_name, prev_rvec=None, prev_tvec=None):
        """Estimate 3D pose for specific marker, using previous pose as guess for stability."""
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
        
        # Method: IPPE (Infinitesimal Plane-Based Pose Estimation)
        # This is the industry standard for flat square markers.
        # It is analytical (non-iterative) and extremely stable for finding the 4 corners.
        # It handles ambiguity better than EPnP or Iterative for planar targets.
        
        try:
            _success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE
            )
        except Exception:
            return None, None
                
        return rvec, tvec
