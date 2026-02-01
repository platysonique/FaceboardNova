#!/usr/bin/env python3
"""
Faceboard Nova - GUI Application
Applies face masks to video feed and outputs to virtual camera
"""

import os
import sys
import cv2
import numpy as np
import subprocess
from pathlib import Path
import mediapipe as mp
from PIL import Image, ImageSequence
import time
import traceback

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox,
                             QFileDialog, QGroupBox, QCheckBox,
                             QMessageBox, QDialog, QListWidget, QScrollArea,
                             QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QImage, QPixmap, QIcon

# Virtual camera support
try:
    import pyvirtualcam
    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

# MediaPipe face mesh landmarks
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362


class VideoProcessor(QThread):
    """Thread for processing video frames"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, masks_dir, max_faces=3, enable_vcam=True, vcam_device=None):
        super().__init__()
        self.masks_dir = masks_dir
        self.max_faces = max_faces
        self.mask_images = {}
        self.mask_names = {}  # Track mask names: {idx: filename}
        self.mask_gif_frames = {}  # For animated GIFs: {idx: [frames...]}
        self.mask_gif_timings = {}  # Frame timings for GIFs: {idx: [delays...]}
        self.mask_gif_current_frame = {}  # Current frame index: {idx: frame_num}
        self.mask_gif_last_update = {}  # Last frame update time: {idx: timestamp}
        self.mask_settings = {}  # {face_idx: {'scale': 1.2, 'opacity': 0.8, 'enabled': True, 'mask_idx': 1}}
        self.running = False
        self.camera_index = 0
        self.enable_vcam = enable_vcam
        self.vcam_device = vcam_device  # None = auto, or specific device path like "/dev/video10"
        
        # Initialize MediaPipe with high quality settings for stable tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # Use refined landmarks for better accuracy
            min_detection_confidence=0.3,  # Lower for faster detection with less face visible (was 0.5)
            min_tracking_confidence=0.3   # Lower for faster tracking with less face visible (was 0.5)
        )
        print(f"MediaPipe initialized for {max_faces} faces (fast detection + off-screen tracking mode)")
        
        # Face tracking for consistent mask assignment
        self.face_tracker = {}  # {face_id: {'center': (x,y), 'size': (w,h), 'mask_idx': 1, 'frames_seen': 0, 'assigned_face_num': None, 'locked': False}}
        self.next_face_id = 1
        self.tracking_threshold = 0.25  # Distance threshold for matching faces (lowered for tighter matching)
        
        # Face identity storage for persistent mask assignment
        # Maps face identity ID to stored face data
        self.face_identities = {}  # {identity_id: {'face_image': np.array, 'face_num': int, 'mask_idx': int, 'landmarks': list}}
        self.next_identity_id = 1
        self.identity_match_threshold = 0.4  # Threshold for matching faces to stored identities (lowered to reduce false matches between different people)
        self.pending_identity_capture = {}  # {face_num: mask_idx} - faces to capture on next frame
        
        self._frame_count = 0
    
    def load_masks(self):
        """Load mask images from directory (supports PNG, JPG, GIF)"""
        self.mask_images = {}
        self.mask_names = {}
        self.mask_gif_frames = {}
        self.mask_gif_timings = {}
        self.mask_gif_current_frame = {}
        self.mask_gif_last_update = {}
        
        if not os.path.exists(self.masks_dir):
            print(f"WARNING: Masks directory does not exist: {self.masks_dir}")
            return
        
        mask_files = sorted([f for f in os.listdir(self.masks_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
        
        print(f"Loading {len(mask_files)} masks from {self.masks_dir}")
        for i, mask_file in enumerate(mask_files, 1):
            mask_path = os.path.join(self.masks_dir, mask_file)
            
            # Check if it's a GIF
            if mask_file.lower().endswith('.gif'):
                try:
                    pil_img = Image.open(mask_path)
                    frames = []
                    delays = []
                    
                    # Check if animated
                    if hasattr(pil_img, 'is_animated') and pil_img.is_animated:
                        for frame in ImageSequence.Iterator(pil_img):
                            # Convert PIL frame to OpenCV format (BGR)
                            frame_cv = cv2.cvtColor(np.array(frame.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
                            frames.append(frame_cv)
                            # Get frame delay (default to 100ms if not specified)
                            delay = frame.info.get('duration', 100) / 1000.0  # Convert ms to seconds
                            delays.append(delay)
                        
                        if frames:
                            self.mask_gif_frames[i] = frames
                            self.mask_gif_timings[i] = delays
                            self.mask_gif_current_frame[i] = 0
                            self.mask_gif_last_update[i] = time.time()
                            # Use first frame as initial image
                            self.mask_images[i] = frames[0]
                            self.mask_names[i] = mask_file
                            print(f"  Loaded animated GIF {i}: {mask_file} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
                        else:
                            print(f"  Failed to load GIF frames: {mask_file}")
                    else:
                        # Static GIF - treat as regular image
                        mask_img = cv2.cvtColor(np.array(pil_img.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
                        self.mask_images[i] = mask_img
                        self.mask_names[i] = mask_file
                        print(f"  Loaded static GIF {i}: {mask_file} ({mask_img.shape[1]}x{mask_img.shape[0]})")
                except (OSError, FileNotFoundError, ValueError, AttributeError) as e:
                    # Handle file I/O errors, missing files, invalid image data, or missing attributes
                    # PIL.UnidentifiedImageError is a subclass of ValueError, so caught by ValueError
                    print(f"  Failed to load GIF {mask_file}: {e}")
            else:
                # Regular image (PNG, JPG)
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_img is not None:
                    self.mask_images[i] = mask_img
                    self.mask_names[i] = mask_file
                    print(f"  Loaded mask {i}: {mask_file} ({mask_img.shape[1]}x{mask_img.shape[0]})")
                else:
                    print(f"  Failed to load: {mask_file}")
        
        if not self.mask_images:
            print("WARNING: No masks loaded!")
    
    def get_face_bounds(self, landmarks, width, height):
        """Get bounding box for a face"""
        x_coords = [landmarks.landmark[i].x * width for i in FACE_OVAL]
        y_coords = [landmarks.landmark[i].y * height for i in FACE_OVAL]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        center_x = (x_min + x_max) / 2.0  # Use float for better precision
        center_y = (y_min + y_max) / 2.0
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        return center_x, center_y, face_width, face_height
    
    def is_face_partially_visible(self, landmarks, width, height, min_visible_ratio=0.1):
        """Check if face is at least partially visible (enough landmarks are on-screen)"""
        # Check key facial landmarks to determine visibility
        key_landmarks = [33, 7, 263, 249, 61, 291, 168, 6]  # Eyes, nose, mouth corners
        visible_count = 0
        
        for idx in key_landmarks:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                # Check if landmark is within frame bounds (with some margin)
                if 0 <= lm.x <= 1.0 and 0 <= lm.y <= 1.0:
                    visible_count += 1
        
        # Face is considered visible if at least min_visible_ratio of key landmarks are on-screen
        # Lower threshold (0.1 = 10%) allows detection with very little face visible
        return visible_count >= len(key_landmarks) * min_visible_ratio
    
    def calculate_head_rotation(self, landmarks, width, height):
        """Calculate head rotation angle (roll) from side-to-side tilt"""
        try:
            # Get eye positions
            left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
            left_eye_inner = landmarks.landmark[LEFT_EYE_INNER]
            right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
            right_eye_inner = landmarks.landmark[RIGHT_EYE_INNER]
            
            # Calculate eye centers
            left_eye_center_x = (left_eye_outer.x + left_eye_inner.x) / 2.0 * width
            left_eye_center_y = (left_eye_outer.y + left_eye_inner.y) / 2.0 * height
            
            right_eye_center_x = (right_eye_outer.x + right_eye_inner.x) / 2.0 * width
            right_eye_center_y = (right_eye_outer.y + right_eye_inner.y) / 2.0 * height
            
            # Calculate angle between eyes (roll rotation)
            dx = right_eye_center_x - left_eye_center_x
            dy = right_eye_center_y - left_eye_center_y
            
            # Angle in degrees (negative because y increases downward)
            angle = -np.degrees(np.arctan2(dy, dx))
            return angle
        except (IndexError, AttributeError):
            # Return 0.0 if landmark access fails (face may be partially visible)
            # This is expected when face is partially off-screen or landmarks are incomplete
            return 0.0
    
    def calculate_face_distance(self, center1, size1, center2, size2):
        """Calculate distance between two faces"""
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        distance = np.sqrt(dx*dx + dy*dy)
        # Normalize by average face size
        avg_size = (size1[0] + size1[1] + size2[0] + size2[1]) / 4.0
        if avg_size > 0:
            distance /= avg_size
        return distance
    
    def match_face_to_tracker(self, center, size):
        """Match a detected face to an existing tracked face"""
        best_match = None
        best_distance = float('inf')
        
        for face_id, tracked in self.face_tracker.items():
            tracked_center = tracked['center']
            tracked_size = tracked['size']
            distance = self.calculate_face_distance(center, size, tracked_center, tracked_size)
            
            if distance < self.tracking_threshold and distance < best_distance:
                best_distance = distance
                best_match = face_id
        
        return best_match
    
    def update_face_tracker(self, face_id, center, size):
        """Update face tracker with exponential smoothing - LOCK assigned_face_num once set"""
        if face_id not in self.face_tracker:
            self.face_tracker[face_id] = {
                'center': center,
                'size': size,
                'mask_idx': 1,
                'frames_seen': 0,
                'assigned_face_num': None,
                'locked': False  # Lock assigned_face_num once it's set
            }
        else:
            # Exponential smoothing (alpha = 0.2 for more stability)
            alpha = 0.2  # Lower alpha = more smoothing = more stable
            old_center = self.face_tracker[face_id]['center']
            old_size = self.face_tracker[face_id]['size']
            
            self.face_tracker[face_id]['center'] = (
                alpha * center[0] + (1 - alpha) * old_center[0],
                alpha * center[1] + (1 - alpha) * old_center[1]
            )
            self.face_tracker[face_id]['size'] = (
                alpha * size[0] + (1 - alpha) * old_size[0],
                alpha * size[1] + (1 - alpha) * old_size[1]
            )
        
        self.face_tracker[face_id]['frames_seen'] += 1
        
        # Face number assignment is now handled in process_frame() 
        # to ensure sequential ordering based on currently detected faces
        # This method just updates the tracker position/size
    
    def extract_face_features(self, landmarks, width, height):
        """Extract normalized face features from landmarks for identity matching"""
        # Use more facial landmarks for better recognition robustness
        # Key points: nose tip, eye corners, mouth corners, jawline points
        # Using more landmarks makes matching more robust to angle/lighting changes
        key_indices = [
            1,      # Nose tip
            33, 133, 263, 362,  # Eye corners (left outer, left inner, right outer, right inner)
            61, 291,  # Mouth corners
            10, 151,  # Jawline points for face shape
            168, 6,   # More facial structure points
        ]
        
        features = []
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                # Normalize coordinates (already normalized by MediaPipe)
                features.extend([lm.x, lm.y])
            else:
                # Pad with zeros if landmark doesn't exist
                features.extend([0.0, 0.0])
        
        # Normalize feature vector to unit length for better matching
        features_array = np.array(features)
        norm = np.linalg.norm(features_array)
        if norm > 0:
            features_array = features_array / norm
        
        return features_array
    
    def match_face_to_identity(self, landmarks, width, height):
        """Match detected face to stored identity"""
        if not self.face_identities:
            return None
        
        current_features = self.extract_face_features(landmarks, width, height)
        
        best_match = None
        best_distance = float('inf')
        
        for identity_id, identity_data in self.face_identities.items():
            stored_features = identity_data.get('features')
            if stored_features is None:
                continue
            
            # Calculate cosine distance (1 - cosine similarity) for normalized vectors
            # This is more robust than Euclidean distance for normalized feature vectors
            dot_product = np.dot(current_features, stored_features)
            # Clamp to [-1, 1] to avoid numerical errors
            dot_product = np.clip(dot_product, -1.0, 1.0)
            cosine_similarity = dot_product
            cosine_distance = 1.0 - cosine_similarity
            
            if cosine_distance < self.identity_match_threshold and cosine_distance < best_distance:
                best_distance = cosine_distance
                best_match = identity_id
        
        if best_match:
            print(f"Matched face to identity {best_match} (distance: {best_distance:.3f})")
        
        return best_match if best_match else None
    
    def capture_face_identity(self, face_num, mask_idx, frame, landmarks, width, height):
        """Capture and store face identity when mask is assigned"""
        # Extract face region from frame
        center_x, center_y, face_width, face_height = self.get_face_bounds(landmarks, width, height)
        
        # Expand bounding box slightly for better capture
        padding = 0.2
        x1 = max(0, int(center_x - face_width * (0.5 + padding)))
        y1 = max(0, int(center_y - face_height * (0.5 + padding)))
        x2 = min(width, int(center_x + face_width * (0.5 + padding)))
        y2 = min(height, int(center_y + face_height * (0.5 + padding)))
        
        # Extract face image (convert BGR to RGB for storage)
        face_image_bgr = frame[y1:y2, x1:x2].copy()
        face_image_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        
        # Extract features
        features = self.extract_face_features(landmarks, width, height)
        
        # Check if identity already exists for this face_num and update it, or create new
        identity_id = None
        for existing_id, identity_data in self.face_identities.items():
            if identity_data['face_num'] == face_num:
                identity_id = existing_id
                break
        
        # If no identity found for this face_num, check if this face matches an existing identity
        # BUT: Only reuse an identity if it's NOT currently assigned to a different visible face
        # This prevents two different people from sharing the same identity/mask
        if identity_id is None:
            # Get currently visible face numbers from face_tracker (more reliable than mask_settings)
            # Any face_id with an assigned_face_num is currently visible
            active_face_nums = set()
            for tracked in self.face_tracker.values():
                assigned_num = tracked.get('assigned_face_num')
                if assigned_num is not None:
                    active_face_nums.add(assigned_num)
            
            # Try matching to see if this is actually a known person
            matched_identity = self.match_face_to_identity(landmarks, width, height)
            if matched_identity:
                # Check if the matched identity is currently assigned to a DIFFERENT face_num
                matched_face_num = self.face_identities[matched_identity]['face_num']
                
                # Only reuse identity if:
                # 1. The matched identity's face_num is NOT currently visible (person left and came back)
                # 2. OR the matched identity's face_num IS the same as this face_num (shouldn't happen, but be safe)
                if matched_face_num not in active_face_nums or matched_face_num == face_num:
                    # Safe to reuse - person left and came back, or same face
                    identity_id = matched_identity
                    print(f"Updating existing identity {identity_id} (was Face {matched_face_num}, now Face {face_num})")
                else:
                    # Matched identity is assigned to a DIFFERENT currently visible face
                    # This means two different people are being confused - create NEW identity
                    identity_id = self.next_identity_id
                    self.next_identity_id += 1
                    print(f"Face {face_num} matched identity {matched_identity} (Face {matched_face_num}), but that identity is in use. Creating new identity {identity_id}.")
            else:
                # New identity - no match found
                identity_id = self.next_identity_id
                self.next_identity_id += 1
        
        # Update or create identity with current features and mask
        self.face_identities[identity_id] = {
            'face_image': face_image_rgb,
            'face_num': face_num,
            'mask_idx': mask_idx,
            'features': features,
            'landmarks': landmarks
        }
        
        print(f"Captured/updated face identity {identity_id} for Face {face_num} with mask {mask_idx} (features shape: {features.shape})")
        return identity_id
    
    def apply_mask_to_face(self, frame, landmarks, settings, width, height):
        """Apply a mask image to a detected face"""
        if not settings.get('enabled', True):
            return frame
        
        mask_idx = settings.get('mask_idx', 1)
        
        # Validate mask exists
        if mask_idx not in self.mask_images:
            # Try to reload masks if not found
            if not self.mask_images:
                self.load_masks()
            if mask_idx not in self.mask_images:
                # Fallback to first available mask
                if self.mask_images:
                    mask_idx = min(self.mask_images.keys())
                    settings['mask_idx'] = mask_idx
                else:
                    return frame
        
        try:
            # Handle animated GIFs - update frame if needed
            if mask_idx in self.mask_gif_frames:
                current_time = time.time()
                last_update = self.mask_gif_last_update.get(mask_idx, current_time)
                current_frame_idx = self.mask_gif_current_frame.get(mask_idx, 0)
                frames = self.mask_gif_frames[mask_idx]
                delays = self.mask_gif_timings[mask_idx]
                
                # Check if it's time to advance to next frame
                if current_time - last_update >= delays[current_frame_idx]:
                    current_frame_idx = (current_frame_idx + 1) % len(frames)
                    self.mask_gif_current_frame[mask_idx] = current_frame_idx
                    self.mask_gif_last_update[mask_idx] = current_time
                    # Update the current mask image
                    self.mask_images[mask_idx] = frames[current_frame_idx]
            
            mask_img = self.mask_images[mask_idx].copy()  # Make a copy to avoid issues
            center_x, center_y, face_width, face_height = self.get_face_bounds(landmarks, width, height)
            
            # Calculate head rotation angle (roll - side to side tilt)
            rotation_angle = self.calculate_head_rotation(landmarks, width, height)
            
            # Apply scale
            scale = settings.get('scale', 1.2)
            mask_width = int(face_width * scale)
            mask_height = int(face_height * scale)
            
            # Resize mask
            mask_resized = cv2.resize(mask_img, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)
            
            # Rotate mask if head is tilted
            if abs(rotation_angle) > 2.0:  # Only rotate if tilt is significant
                center_mask = (mask_width // 2, mask_height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center_mask, rotation_angle, 1.0)
                mask_resized = cv2.warpAffine(mask_resized, rotation_matrix, (mask_width, mask_height),
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            
            # Calculate position (center mask on face) - allow extending beyond frame boundaries
            x1_full = int(center_x - mask_width // 2)
            y1_full = int(center_y - mask_height // 2)
            x2_full = x1_full + mask_width
            y2_full = y1_full + mask_height
            
            # Calculate the portion that's actually within the frame
            x1_frame = max(0, x1_full)
            y1_frame = max(0, y1_full)
            x2_frame = min(width, x2_full)
            y2_frame = min(height, y2_full)
            
            # Calculate offsets for mask extraction (how much is off-screen)
            mask_x_offset = max(0, -x1_full)  # How much mask extends left of frame
            mask_y_offset = max(0, -y1_full)  # How much mask extends above frame
            
            # Extract ROI from frame (only the visible portion)
            roi_height = y2_frame - y1_frame
            roi_width = x2_frame - x1_frame
            
            if roi_height > 0 and roi_width > 0:
                roi = frame[y1_frame:y2_frame, x1_frame:x2_frame].copy()
                
                # Extract corresponding portion from mask
                mask_y_end = mask_y_offset + roi_height
                mask_x_end = mask_x_offset + roi_width
                mask_roi = mask_resized[mask_y_offset:mask_y_end, mask_x_offset:mask_x_end]
                
                # Handle alpha channel
                if mask_roi.shape[2] == 4:
                    # Extract alpha channel and convert to 2D array (h, w) not (h, w, 1)
                    alpha = mask_roi[:, :, 3] / 255.0  # Shape: (h, w)
                    opacity = settings.get('opacity', 0.8)
                    alpha = alpha * opacity  # Shape: (h, w)
                    
                    # Blend each color channel
                    for c in range(3):
                        # Ensure alpha is broadcastable: (h, w) * (h, w) works fine
                        roi[:, :, c] = (alpha * mask_roi[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
                else:
                    # No alpha channel, use opacity
                    opacity = settings.get('opacity', 0.8)
                    roi = cv2.addWeighted(roi, 1 - opacity, mask_roi[:, :, :3], opacity, 0)
                
                # Put ROI back (only the visible portion)
                frame[y1_frame:y2_frame, x1_frame:x2_frame] = roi
            
        except (cv2.error, ValueError, IndexError, AttributeError) as e:
            # Handle OpenCV errors, invalid array operations, or missing attributes
            print(f"Error applying mask (idx={mask_idx}): {e}")
            traceback.print_exc()
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame"""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        self._frame_count += 1
        
        # Track which face IDs are seen this frame
        seen_face_ids = set()
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            num_faces = len(results.multi_face_landmarks)
            # Only print debug every 30 frames to reduce spam
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                print(f"DEBUG: Detected {num_faces} faces, max_faces={self.max_faces}, mask_images={len(self.mask_images)}")
            
            # First pass: match detected faces to tracked faces
            # Also try to match tracked faces that went off-screen to newly detected faces
            detected_faces = []
            # Process ALL detected faces, not just up to max_faces for matching
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Check if face is at least partially visible (allows partial detection)
                # Use very lenient threshold (0.1 = 10% of key landmarks visible) for faster detection
                if not self.is_face_partially_visible(face_landmarks, width, height, min_visible_ratio=0.1):
                    # Face is too far off-screen - still process it but with lower priority
                    # This allows tracking to continue even when face is mostly off-screen
                    pass  # Continue processing - MediaPipe already filtered it
                
                center_x, center_y, face_width, face_height = self.get_face_bounds(face_landmarks, width, height)
                center = (center_x, center_y)
                size = (face_width, face_height)
                
                # Try to match to existing tracked face (including ones that went off-screen)
                face_id = self.match_face_to_tracker(center, size)
                
                if face_id is None:
                    # New face detected - assign new ID
                    face_id = self.next_face_id
                    self.next_face_id += 1
                    # Wrap around if we exceed max_faces, but keep tracking
                    if self.next_face_id > 100:  # Prevent infinite growth
                        self.next_face_id = self.max_faces + 1
                
                # Update tracker (even if face was off-screen, this brings it back)
                self.update_face_tracker(face_id, center, size)
                seen_face_ids.add(face_id)
                
                detected_faces.append({
                    'id': face_id,
                    'landmarks': face_landmarks,
                    'center': center,
                    'size': size
                })
            
            
            # Remove faces that haven't been seen for a while
            # But keep at least the faces we're currently processing
            # Extended timeout to persist faces that go off-screen (3 seconds at 30fps = 90 frames)
            faces_to_remove = []
            active_face_ids = {f['id'] for f in detected_faces[:self.max_faces]}
            
            for face_id in list(self.face_tracker.keys()):
                if face_id not in seen_face_ids:
                    self.face_tracker[face_id]['frames_seen'] -= 1
                    # Extended timeout: 90 frames (3 seconds) to allow faces to go off-screen and come back
                    # Only remove if face hasn't been seen AND it's not in the active set
                    tracked = self.face_tracker[face_id]
                    assigned_num = tracked.get('assigned_face_num')
                    frames_seen = tracked['frames_seen']
                    
                    # Keep faces with assigned numbers longer (6 seconds = 180 frames)
                    # Unassigned faces removed after 3 seconds (90 frames)
                    removal_threshold = -180 if assigned_num is not None else -90
                    
                    if frames_seen < removal_threshold:
                        if face_id not in active_face_ids:
                            faces_to_remove.append(face_id)
                else:
                    # Face was seen - reset frames_seen counter to prevent removal
                    # This allows faces to persist even if they're partially off-screen
                    if self.face_tracker[face_id]['frames_seen'] < 0:
                        self.face_tracker[face_id]['frames_seen'] = 0
            
            for face_id in faces_to_remove:
                tracked = self.face_tracker.get(face_id)
                if tracked:
                    frames_gone = abs(tracked.get('frames_seen', 0))
                    del self.face_tracker[face_id]
                    print(f"Removed face_id {face_id} from tracker (gone for {frames_gone} frames)")
            
            # Match detected faces to stored identities first
            # Use a greedy matching algorithm: find best match for each face, ensuring no identity is shared
            identity_matches = {}  # {face_id: identity_id}
            
            # Build list of (face_id, landmarks, best_identity, best_distance) for all faces
            face_matches = []
            for face_data in detected_faces[:self.max_faces]:
                face_id = face_data['id']
                landmarks = face_data['landmarks']
                
                # Find best matching identity for this face
                best_identity = None
                best_distance = float('inf')
                
                if self.face_identities:
                    current_features = self.extract_face_features(landmarks, width, height)
                    for identity_id, identity_data in self.face_identities.items():
                        stored_features = identity_data.get('features')
                        if stored_features is None:
                            continue
                        
                        # Calculate cosine distance
                        dot_product = np.dot(current_features, stored_features)
                        dot_product = np.clip(dot_product, -1.0, 1.0)
                        cosine_similarity = dot_product
                        cosine_distance = 1.0 - cosine_similarity
                        
                        if cosine_distance < self.identity_match_threshold and cosine_distance < best_distance:
                            best_distance = cosine_distance
                            best_identity = identity_id
                
                if best_identity:
                    face_matches.append((face_id, landmarks, best_identity, best_distance))
            
            # Sort by distance (best matches first) to prioritize better matches
            face_matches.sort(key=lambda x: x[3])
            
            # Assign identities greedily, ensuring each identity is only used once
            used_identities = set()
            for face_id, landmarks, matched_identity, distance in face_matches:
                if matched_identity not in used_identities:
                    identity_matches[face_id] = matched_identity
                    used_identities.add(matched_identity)
                    # Restore mask assignment from identity to tracker immediately
                    identity_data = self.face_identities[matched_identity]
                    if face_id in self.face_tracker:
                        self.face_tracker[face_id]['mask_idx'] = identity_data['mask_idx']
                        print(f"Restored mask_idx {identity_data['mask_idx']} to tracker for face_id {face_id} (identity {matched_identity})")
                else:
                    print(f"Face {face_id} matched identity {matched_identity} (distance: {distance:.3f}), but identity already assigned to another face. Skipping.")
            
            # Reassign face numbers sequentially based on currently detected faces (only up to max_faces)
            # Sort detected faces by their current assigned number (if any) to maintain stability
            active_faces = detected_faces[:self.max_faces]
            # Sort by assigned_face_num if available, otherwise by face_id for stability
            active_faces.sort(key=lambda f: (
                self.face_tracker.get(f['id'], {}).get('assigned_face_num') or 999,
                f['id']
            ))
            
            # Reassign sequentially - ensure all active faces have assigned numbers
            for idx, face_data in enumerate(active_faces, 1):
                face_id = face_data['id']
                if face_id not in self.face_tracker:
                    # Shouldn't happen, but be safe
                    continue
                tracked = self.face_tracker[face_id]
                # Assign sequential number (1, 2, 3...)
                tracked['assigned_face_num'] = idx
                tracked['locked'] = True
                
                # If this face matches a stored identity, restore mask to NEW assigned face_num
                if face_id in identity_matches:
                    identity_data = self.face_identities[identity_matches[face_id]]
                    # Restore mask_idx to the NEW assigned face_num (not the old stored one)
                    if idx not in self.mask_settings:
                        self.mask_settings[idx] = {
                            'enabled': True,
                            'mask_idx': identity_data['mask_idx'],
                            'scale': 1.2,
                            'opacity': 0.8
                        }
                    else:
                        self.mask_settings[idx]['mask_idx'] = identity_data['mask_idx']
                    # Update identity's face_num to current assignment
                    identity_data['face_num'] = idx
                else:
                    # No identity match - ensure default settings exist (but don't enable mask yet)
                    # This prevents the "no settings found" warning
                    if idx not in self.mask_settings:
                        # Get mask_idx from tracker if available, otherwise use default
                        mask_idx = tracked.get('mask_idx', 1)
                        self.mask_settings[idx] = {
                            'enabled': False,  # Disabled until user assigns a mask
                            'mask_idx': mask_idx,
                            'scale': 1.2,
                            'opacity': 0.8
                        }
            
            # Capture face identities if requested (after reassignment so we have correct face_num)
            for face_data in active_faces:
                face_id = face_data['id']
                tracked = self.face_tracker.get(face_id)
                if tracked:
                    assigned_face_num = tracked.get('assigned_face_num')
                    if assigned_face_num and assigned_face_num in self.pending_identity_capture:
                        # Capture this face's identity
                        mask_idx = self.pending_identity_capture.pop(assigned_face_num)
                        self.capture_face_identity(
                            assigned_face_num, 
                            mask_idx, 
                            frame, 
                            face_data['landmarks'], 
                            width, 
                            height
                        )
            
            # Second pass: apply masks to faces (only up to max_faces)
            for face_data in active_faces:
                face_id = face_data['id']
                tracked = self.face_tracker.get(face_id)
                
                if tracked:
                    assigned_face_num = tracked.get('assigned_face_num')
                    if assigned_face_num and assigned_face_num in self.mask_settings:
                        # Get settings and make a copy to avoid modifying original
                        settings = self.mask_settings[assigned_face_num].copy()
                        # Use mask_idx from tracker if available (may be restored from identity), otherwise from settings
                        mask_idx_from_tracker = tracked.get('mask_idx')
                        mask_idx_from_settings = settings.get('mask_idx', 1)
                        # Prefer tracker's mask_idx (restored from identity) over settings
                        mask_idx = mask_idx_from_tracker if mask_idx_from_tracker is not None else mask_idx_from_settings
                        settings['mask_idx'] = mask_idx
                        
                        # Ensure mask_settings is updated with the mask_idx from tracker/identity
                        if self.mask_settings[assigned_face_num].get('mask_idx') != mask_idx:
                            self.mask_settings[assigned_face_num]['mask_idx'] = mask_idx
                            if mask_idx_from_tracker is not None:
                                print(f"Updated mask_settings[{assigned_face_num}]['mask_idx'] to {mask_idx} (from identity)")
                        
                        # Ensure scale and opacity are in settings (use defaults if missing)
                        if 'scale' not in settings:
                            settings['scale'] = 1.2
                        if 'opacity' not in settings:
                            settings['opacity'] = 0.8
                        
                        if settings.get('enabled', True):
                            try:
                                frame = self.apply_mask_to_face(
                                    frame, 
                                    face_data['landmarks'], 
                                    settings, 
                                    width, 
                                    height
                                )
                            except (cv2.error, ValueError, IndexError, AttributeError) as e:
                                # Handle OpenCV errors, invalid array operations, or missing attributes
                                print(f"Error applying mask {mask_idx} to face {face_id}: {e}")
                                traceback.print_exc()
                        else:
                            if self._frame_count % 60 == 0:  # Only warn occasionally
                                print(f"WARNING: Face {face_id} is disabled")
                    else:
                        if self._frame_count % 60 == 0:  # Only warn occasionally
                            print(f"WARNING: Face {face_id} has assigned_face_num {assigned_face_num} but no settings found")
        
        return frame
    
    def run(self):
        """Main processing loop"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {self.camera_index}")
            return
        
        # Set camera to high quality if possible
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try 1920x1080
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        self.running = True
        
        self.virtual_cam = None
        self.v4l2_writer = None  # Direct v4l2loopback writer
        self.v4l2_device = None  # Device path being used
        self.v4l2_format = 'BGR24'  # Format being used
        
        # Skip virtual camera initialization if disabled
        if not self.enable_vcam:
            print("Virtual camera output disabled - preview only mode")
            print("  (OBS can use Window Capture or you can enable VCam output)")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            
            # Store these for frame output
            self.output_width = width
            self.output_height = height
            self.output_fps = fps
            
            # DEFINITIVE FIX based on research:
            # OBS Virtual Camera ALWAYS uses the FIRST v4l2loopback device it finds
            # OBS cannot be configured to use a different device (no GUI option)
            # Solution: Exclude video9 (first device for OBS) and use video10+ for our app
            EXCLUDED_DEVICES = {'/dev/video9', '/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3', 
                               '/dev/video4', '/dev/video5', '/dev/video6', '/dev/video7', '/dev/video8'}
            
            # Check which device OBS is using (it uses the FIRST v4l2loopback device)
            obs_device = None
            if os.path.exists('/dev/video9'):
                try:
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', '/dev/video9', '--info'],
                        capture_output=True, text=True, timeout=1
                    )
                    if 'v4l2loopback' in result.stdout.lower():
                        obs_device = '/dev/video9'
                        print(f"  Research-based fix: OBS uses {obs_device} (first v4l2loopback device)")
                        print(f"  Our app will use video10+ to avoid conflicts")
                except (subprocess.SubprocessError, FileNotFoundError, OSError):
                    # v4l2-ctl may not be available or command may fail - continue without device info
                    pass
            
            available_devices = []
            devices_in_use = set()
            
            # Check which devices are currently in use by other processes (like OBS)
            try:
                lsof_result = subprocess.run(
                    ['lsof', '/dev/video*'],
                    capture_output=True, text=True, timeout=2
                )
                if lsof_result.returncode == 0:
                    for line in lsof_result.stdout.split('\n'):
                        if '/dev/video' in line:
                            # Extract device path (e.g., /dev/video9)
                            parts = line.split()
                            for part in parts:
                                if '/dev/video' in part:
                                    devices_in_use.add(part)
                                    print(f"  Device {part} is in use by another process")
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                # lsof might not be available or command may fail - continue anyway
                pass
            
            # Add excluded devices to the in-use set
            devices_in_use.update(EXCLUDED_DEVICES)
            print(f"  DEFINITIVE FIX: Excluding devices: {', '.join(sorted(EXCLUDED_DEVICES))}")
            print(f"  This ensures /dev/video9 is NEVER used (reserved for OBS)")
            
            # Check for v4l2loopback devices - START FROM video10 to avoid conflicts
            # NEVER check video9 or lower - they are permanently excluded
            for device_idx in range(10, 20):  # Start at 10, NEVER use 9
                device_path = f'/dev/video{device_idx}'
                
                # DOUBLE CHECK: Never add excluded devices
                if device_path in EXCLUDED_DEVICES:
                    print(f"  SKIPPING {device_path} - permanently excluded")
                    continue
                
                if os.path.exists(device_path):
                    # Check if it's a v4l2loopback device
                    is_v4l2loopback = False
                    try:
                        result = subprocess.run(
                            ['v4l2-ctl', '--device', device_path, '--info'],
                            capture_output=True, text=True, timeout=1
                        )
                        if 'v4l2loopback' in result.stdout.lower():
                            is_v4l2loopback = True
                    except (subprocess.SubprocessError, FileNotFoundError, OSError):
                        # If v4l2-ctl fails, still try the device (might be v4l2loopback)
                        # But only if it's not explicitly excluded
                        if device_path not in EXCLUDED_DEVICES:
                            is_v4l2loopback = True
                    
                    if is_v4l2loopback:
                        # Skip devices that are in use OR explicitly excluded
                        if device_path not in devices_in_use:
                            available_devices.append((device_idx, device_path))
                            print(f"  ✓ Found available device: {device_path}")
                        else:
                            print(f"  ✗ Skipping {device_path} - excluded or in use")
            
            # Sort devices by index DESCENDING (prefer highest numbers first)
            # This ensures we use video13, video12, video11, video10 in that order
            available_devices.sort(key=lambda x: x[0], reverse=True)
            
            if available_devices:
                print(f"  ✓ Found {len(available_devices)} available device(s): {[d[1] for d in available_devices]}")
            else:
                print("⚠ CRITICAL: No available v4l2loopback devices found!")
                print("  Only /dev/video9 exists and it's excluded (reserved for OBS)")
                print("")
                print("  Attempting to create devices automatically...")
                
                # Try to create devices automatically
                try:
                    # Check if module needs to be reloaded
                    need_reload = False
                    for i in range(10, 14):
                        if not os.path.exists(f'/dev/video{i}'):
                            need_reload = True
                            break
                    
                    if need_reload:
                        # CRITICAL: OBS uses the FIRST v4l2loopback device it finds
                        # We need to reload with video9 FIRST (for OBS), then add more devices
                        # Check if video9 exists and is v4l2loopback
                        video9_is_v4l2 = False
                        if os.path.exists('/dev/video9'):
                            try:
                                result = subprocess.run(
                                    ['v4l2-ctl', '--device', '/dev/video9', '--info'],
                                    capture_output=True, text=True, timeout=1
                                )
                                if 'v4l2loopback' in result.stdout.lower():
                                    video9_is_v4l2 = True
                            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                                # Device check may fail - continue
                                pass
                        
                        # Try to unload and reload module
                        try:
                            print("  Unloading existing v4l2loopback module...")
                            print("  (OBS Virtual Camera must be stopped first!)")
                            unload_result = subprocess.run(
                                ['sudo', 'modprobe', '-r', 'v4l2loopback'],
                                capture_output=True, text=True, timeout=5
                            )
                            if unload_result.returncode == 0:
                                print("  ✓ Module unloaded")
                                import time
                                time.sleep(1)
                            else:
                                print(f"  ⚠ Could not unload: {unload_result.stderr}")
                                print("  SOLUTION: Stop OBS Virtual Camera, then restart this app")
                                self.enable_vcam = False
                                return
                        except (subprocess.SubprocessError, OSError, PermissionError) as e:
                            # Handle subprocess errors, OS errors, or permission issues
                            print(f"  ⚠ Could not unload module: {e}")
                            self.enable_vcam = False
                            return
                        
                        # Try to load with ALL devices: video9 (for OBS) + video10-13 (for us)
                        try:
                            # Research shows: OBS uses FIRST v4l2loopback device
                            # Load with video9 FIRST (for OBS), then video10-13 (for us)
                            print("  Loading v4l2loopback with 5 devices...")
                            print("  - video9: Reserved for OBS (labeled 'cumcam', OBS auto-selects first device)")
                            print("  - video10-13: For Faceboard Nova")
                            load_result = subprocess.run([
                                'sudo', 'modprobe', 'v4l2loopback',
                                'devices=5', 'video_nr=9,10,11,12,13',
                                'card_label=cumcam,VCam-1,VCam-2,VCam-3,VCam-4',
                                'exclusive_caps=1'
                            ], capture_output=True, text=True, timeout=5)
                            
                            if load_result.returncode == 0:
                                print("  ✓ Module loaded")
                                import time
                                time.sleep(1)
                                
                                # Re-check for devices
                                for device_idx in range(10, 14):
                                    device_path = f'/dev/video{device_idx}'
                                    if os.path.exists(device_path):
                                        try:
                                            result = subprocess.run(
                                                ['v4l2-ctl', '--device', device_path, '--info'],
                                                capture_output=True, text=True, timeout=1
                                            )
                                            if 'v4l2loopback' in result.stdout.lower():
                                                if device_path not in devices_in_use:
                                                    available_devices.append((device_idx, device_path))
                                                    print(f"  ✓ Created and found device: {device_path}")
                                        except (OSError, ValueError, subprocess.SubprocessError):
                                            # Device creation check may fail - continue
                                            pass
                            else:
                                print(f"  ⚠ Failed to load module: {load_result.stderr}")
                        except (subprocess.SubprocessError, OSError, PermissionError) as e:
                            # Handle subprocess errors, OS errors, or permission issues
                            print(f"  ⚠ Could not load module: {e}")
                    
                    if not available_devices:
                        print("")
                        print("  ⚠ Automatic device creation failed (requires sudo password)")
                        print("  SOLUTION: Run manually in terminal:")
                        print("    sudo modprobe -r v4l2loopback")
                        print("    sudo modprobe v4l2loopback devices=4 video_nr=10,11,12,13")
                        print("  Then restart this app.")
                        self.enable_vcam = False
                except (subprocess.SubprocessError, OSError, PermissionError) as e:
                    # Handle subprocess errors, OS errors, or permission issues during device creation
                    print(f"  ⚠ Error during automatic device creation: {e}")
                    print("  Please run: bash fix_vcam_conflict.sh")
                    self.enable_vcam = False
            
            # If specific device requested, prioritize it
            if self.vcam_device and self.vcam_device != "Auto (find available)":
                if os.path.exists(self.vcam_device):
                    available_devices.insert(0, (int(self.vcam_device.split('video')[-1]), self.vcam_device))
                else:
                    print(f"⚠ Warning: Requested device {self.vcam_device} not found, using auto-detection")
            
            # Try direct v4l2loopback output using OpenCV VideoWriter
            # Try devices in order (already sorted highest to lowest)
            device_tried = []
            if not available_devices:
                print("  ⚠ SKIPPING: No devices available to try")
                print("  Virtual camera initialization ABORTED - no devices found")
                self.v4l2_writer = None
                self.virtual_cam = None
            else:
                print(f"  Attempting to initialize virtual camera on devices: {[d[1] for d in available_devices]}")
                for device_idx, device_path in available_devices:
                    # ABSOLUTE SAFETY CHECK: Never use video9 or excluded devices
                    if device_path in EXCLUDED_DEVICES or device_idx < 10:
                        print(f"  ERROR: Attempted to use excluded device {device_path} - SKIPPING!")
                        continue
                    
                    try:
                        # Setup device format - let OpenCV handle it, but try to set compatible format
                        try:
                            # Try RGB24 (BGR24) first, then YUYV
                            subprocess.run([
                                'v4l2-ctl', '--device', device_path,
                                '--set-fmt-video', f'width={width},height={height},pixelformat=RGB3'
                            ], check=True, timeout=2, capture_output=True)
                        except (subprocess.SubprocessError, FileNotFoundError, OSError):
                            try:
                                # Fallback to YUYV
                                subprocess.run([
                                    'v4l2-ctl', '--device', device_path,
                                    '--set-fmt-video', f'width={width},height={height},pixelformat=YUYV'
                                ], check=True, timeout=2, capture_output=True)
                            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                                # Continue even if setup fails - device may work without explicit format setting
                                pass
                        
                        # Try to create VideoWriter for v4l2loopback
                        # v4l2loopback needs specific format - try multiple approaches
                        v4l2_writer = None
                        formats_to_try = [
                            (0, 'auto-detect'),
                            (cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'), 'YUYV'),
                            (cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 'MJPEG'),
                            (cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 'XVID'),
                        ]
                        
                        for fourcc_code, format_name in formats_to_try:
                            try:
                                if fourcc_code == 0:
                                    v4l2_writer = cv2.VideoWriter(device_path, 0, fps, (width, height))
                                else:
                                    v4l2_writer = cv2.VideoWriter(device_path, fourcc_code, fps, (width, height))
                                
                                if v4l2_writer and v4l2_writer.isOpened():
                                    # Test write
                                    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                                    test_success = v4l2_writer.write(test_frame)
                                    if test_success:
                                        self.v4l2_writer = v4l2_writer
                                        self.v4l2_device = device_path
                                        print(f"✓ Direct v4l2loopback output started on {device_path}: {width}x{height} @ {fps} FPS")
                                        print(f"  Using OpenCV VideoWriter with {format_name} format")
                                        print(f"  Frame format: BGR, Size: {width}x{height}, FPS: {fps}")
                                        print(f"  Test write: SUCCESS")
                                        print(f"  This device is reserved for Faceboard Nova")
                                        print(f"  OBS Virtual Camera should use /dev/video9 (cumcam)")
                                        break
                                    else:
                                        v4l2_writer.release()
                                        v4l2_writer = None
                                        print(f"  {format_name} format: Test write returned False, trying next...")
                                else:
                                    if v4l2_writer:
                                        v4l2_writer.release()
                                    v4l2_writer = None
                            except (cv2.error, OSError, IOError) as e:
                                # Handle OpenCV format errors or device I/O errors
                                if v4l2_writer:
                                    try:
                                        v4l2_writer.release()
                                    except Exception:
                                        pass  # Ignore errors during cleanup
                                v4l2_writer = None
                                print(f"  {format_name} format failed: {e}")
                                continue
                        
                        if self.v4l2_writer:
                            break
                        else:
                            print(f"  All VideoWriter formats failed for {device_path}")
                            device_tried.append((device_path, "all formats failed"))
                    except (cv2.error, OSError, IOError) as e:
                        # Handle OpenCV VideoWriter creation errors or device I/O errors
                        # Clean up any partially created VideoWriter
                        if 'v4l2_writer' in locals() and v4l2_writer is not None:
                            try:
                                v4l2_writer.release()
                            except Exception:
                                pass  # Ignore errors during cleanup
                        # Also clean up instance variable if it exists
                        if hasattr(self, 'v4l2_writer') and self.v4l2_writer is not None:
                            try:
                                self.v4l2_writer.release()
                            except Exception:
                                pass  # Ignore errors during cleanup
                            self.v4l2_writer = None
                        print(f"  Failed to create VideoWriter for {device_path}: {e}")
                        traceback.print_exc()
                        device_tried.append((device_path, str(e)))
                        # Continue to next device
                        error_str = str(e).lower()
                        if 'busy' in error_str or 'in use' in error_str or 'permission' in error_str:
                            print(f"  Device {device_path} is busy (likely used by OBS or another app) - trying next device...")
                            continue
            
            # Fallback to pyvirtualcam if direct v4l2loopback didn't work
            if self.v4l2_writer is None and HAS_VIRTUAL_CAM and available_devices:
                print("  Direct v4l2loopback failed, trying pyvirtualcam...")
                for device_idx, device_path in available_devices:
                    # ABSOLUTE SAFETY CHECK: Never use video9 or excluded devices
                    if device_path in EXCLUDED_DEVICES or device_idx < 10:
                        print(f"  ERROR: Attempted to use excluded device {device_path} - SKIPPING!")
                        continue
                    
                    try:
                        self.virtual_cam = pyvirtualcam.Camera(
                            width=width, 
                            height=height, 
                            fps=fps,
                            fmt=pyvirtualcam.PixelFormat.BGR,
                            device=device_path
                        )
                        print(f"✓ Virtual camera started on {device_path}: {width}x{height} @ {fps} FPS")
                        if hasattr(self.virtual_cam, 'device'):
                            print(f"  Device: {self.virtual_cam.device}")
                        break
                    except (OSError, RuntimeError, ValueError) as e:
                        # pyvirtualcam may raise OSError (device busy), RuntimeError (initialization), or ValueError (invalid params)
                        error_str = str(e).lower()
                        if 'busy' in error_str or 'in use' in error_str or 'permission' in error_str:
                            continue
                        if device_idx == available_devices[-1][0]:
                            print(f"  All devices failed: {e}")
                
                # DO NOT use auto-detect as fallback - it might pick video9
                # Only use explicitly selected devices
                if self.virtual_cam is None:
                    print("⚠ Could not start virtual camera on any available device")
                    print("  Preview will still work - you can use Window Capture in OBS")
                    print("  To fix: Create more v4l2loopback devices:")
                    print("    sudo modprobe v4l2loopback devices=4 video_nr=10,11,12,13")
                
                if self.virtual_cam is None:
                    print("⚠ WARNING: Could not start virtual camera")
                    print("  The preview will still work, but OBS won't receive the video")
                    print("  Try:")
                    print("    1. Create more v4l2loopback devices: sudo modprobe v4l2loopback devices=4")
                    print("    2. Or use Window Capture in OBS")
                    print("    3. Or disable Virtual Camera Output in settings")
        
        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (frame is already BGR from OpenCV)
            processed_frame = self.process_frame(frame.copy())
            
            # Output to virtual camera (only if enabled)
            if self.enable_vcam:
                # Ensure frame is in correct format and size
                output_frame = processed_frame.copy()
                
                # Make sure frame matches expected dimensions
                if hasattr(self, 'output_width') and hasattr(self, 'output_height'):
                    if output_frame.shape[:2] != (self.output_height, self.output_width):
                        output_frame = cv2.resize(output_frame, (self.output_width, self.output_height))
                elif output_frame.shape[:2] != (height, width):
                    output_frame = cv2.resize(output_frame, (width, height))
                
                # Ensure frame is BGR (3 channels)
                if len(output_frame.shape) != 3 or output_frame.shape[2] != 3:
                    if len(output_frame.shape) == 2:
                        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
                    elif output_frame.shape[2] == 4:
                        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGRA2BGR)
                
                if hasattr(self, 'v4l2_writer') and self.v4l2_writer:
                    # Check if writer is still opened
                    if not self.v4l2_writer.isOpened():
                        print(f"⚠ WARNING: v4l2_writer is no longer opened!")
                        self.v4l2_writer = None
                    else:
                        # Direct v4l2loopback output
                        try:
                            # Ensure frame is correct size and type
                            if output_frame.shape[:2] != (self.output_height, self.output_width):
                                output_frame = cv2.resize(output_frame, (self.output_width, self.output_height))
                            
                            # Ensure BGR format
                            if len(output_frame.shape) != 3 or output_frame.shape[2] != 3:
                                if len(output_frame.shape) == 2:
                                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
                                elif output_frame.shape[2] == 4:
                                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGRA2BGR)
                            
                            success = self.v4l2_writer.write(output_frame)
                            frame_count += 1
                            if frame_count == 1:
                                print(f"✓ Started writing frames to v4l2loopback")
                                print(f"  Device: {getattr(self, 'v4l2_device', 'unknown')}")
                                print(f"  Frame size: {output_frame.shape}, dtype: {output_frame.dtype}")
                            if frame_count % 60 == 0:  # Debug every 60 frames (~2 seconds)
                                print(f"  Writing frames... ({frame_count} frames, write_success={success})")
                            if not success and frame_count % 30 == 0:
                                print(f"⚠ Warning: VideoWriter.write() returned False - frames may not be reaching device")
                        except (cv2.error, OSError, IOError) as e:
                            # Handle OpenCV write errors or device I/O errors
                            print(f"⚠ Error writing to v4l2loopback: {e}")
                            traceback.print_exc()
                elif hasattr(self, 'virtual_cam') and self.virtual_cam:
                    # pyvirtualcam output - expects BGR
                    try:
                        self.virtual_cam.send(output_frame)
                        self.virtual_cam.sleep_until_next_frame()
                        frame_count += 1
                        if frame_count % 30 == 0:  # Debug every 30 frames
                            print(f"  Sending frames to pyvirtualcam... ({frame_count} frames)")
                    except (OSError, RuntimeError, AttributeError) as e:
                        # Handle pyvirtualcam errors (device issues, not initialized, etc.)
                        print(f"⚠ Error sending to pyvirtualcam: {e}")
                        traceback.print_exc()
                else:
                    if frame_count == 0:
                        print("⚠ ERROR: Virtual camera not initialized!")
                        print("  Reason: No v4l2loopback devices available (only /dev/video9 exists)")
                        print("  Fix: Run 'bash fix_vcam_conflict.sh' to create more devices")
                        print("  Then restart this application")
                    elif frame_count % 300 == 0:  # Remind every 5 seconds at 60fps
                        print("  ⚠ Virtual camera still not initialized - frames not being sent to OBS")
            
            # Emit for preview (preserve quality)
            self.frame_ready.emit(processed_frame)
        
        cap.release()
        if hasattr(self, 'v4l2_writer') and self.v4l2_writer:
            self.v4l2_writer.release()
        if hasattr(self, 'virtual_cam') and self.virtual_cam:
            self.virtual_cam.close()
    
    def stop(self):
        """Stop processing"""
        self.running = False


class FixedComboBox(QComboBox):
    """QComboBox with fixed dropdown positioning for frameless windows"""
    def showPopup(self):
        """Override to fix dropdown position"""
        super().showPopup()
        # Wait a moment for popup to be created, then reposition
        QTimer.singleShot(10, self._reposition_popup)
    
    def _reposition_popup(self):
        """Reposition the popup dropdown"""
        # Get the view and its window (the popup)
        view = self.view()
        if view:
            popup = view.window()
            if popup and popup.isVisible():
                # Get combo box position in global coordinates
                combo_rect = self.rect()
                combo_global_pos = self.mapToGlobal(combo_rect.bottomLeft())
                # Position dropdown directly below combo box
                popup.move(combo_global_pos.x(), combo_global_pos.y() + 2)


class TitleBar(QWidget):
    """Custom titlebar with vaporwave styling - handles window dragging directly"""
    def __init__(self, parent, title_text=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(40)  # Increased height to accommodate larger buttons (28px + margins)
        self.drag_position = QPoint()  # Store drag offset
        self.dragging = False  # Track dragging state
        
        # Set object name for debugging
        self.setObjectName("TitleBar")
        
        # Ensure widget can receive mouse events
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        self.setStyleSheet("""
            QWidget#TitleBar {
                background-color: #1a1a2e;
                border-bottom: 2px solid #9d00ff;
            }
        """)
        
        # Logging: initialization
        parent_class = parent.__class__.__name__ if parent else "None"
        print(f"[TitleBar.__init__] Created TitleBar for parent={parent_class}, objectName={self.objectName()}, height={self.height()}")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 6, 5, 6)  # Vertical margins to center 28px buttons in 40px height
        layout.setSpacing(5)
        
        # Title
        title = title_text if title_text else parent.windowTitle()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #00ffff; font-weight: bold; font-size: 11pt;")
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Window buttons - increased size to show symbols clearly
        self.minimize_btn = QPushButton("−")
        self.minimize_btn.setFixedSize(28, 28)  # Larger buttons for better symbol visibility
        self.minimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a3e;
                color: #e0e0e0;
                border: 2px solid #9d00ff;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16pt;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #9d00ff;
                border-color: #00ffff;
            }
        """)
        self.minimize_btn.clicked.connect(self.parent.showMinimized)
        
        self.maximize_btn = QPushButton("□")
        self.maximize_btn.setFixedSize(28, 28)  # Larger buttons for better symbol visibility
        self.maximize_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a3e;
                color: #e0e0e0;
                border: 2px solid #9d00ff;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14pt;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #9d00ff;
                border-color: #00ffff;
            }
        """)
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(28, 28)  # Larger buttons for better symbol visibility
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a3e;
                color: #e0e0e0;
                border: 2px solid #ff1493;
                border-radius: 4px;
                font-weight: bold;
                font-size: 18pt;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #ff1493;
                border-color: #00ffff;
                color: #ffffff;
            }
        """)
        self.close_btn.clicked.connect(self.parent.close)
        
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)
        
        self.setLayout(layout)
        
        # Logging: after layout setup
        print(f"[TitleBar.__init__] Layout complete, final height={self.height()}, parent window pos={parent.pos() if parent else 'N/A'}")
    
    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()
    
    def enterEvent(self, event):
        """Log when mouse enters titlebar"""
        print(f"[TitleBar.enterEvent] Mouse entered TitleBar (parent={self.parent.__class__.__name__ if self.parent else 'None'})")
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Log when mouse leaves titlebar"""
        print(f"[TitleBar.leaveEvent] Mouse left TitleBar")
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press - start dragging window if not clicking buttons"""
        # Logging: mouse press
        print(f"[TitleBar.mousePressEvent] Button={event.button()}, pos={event.pos()}, globalPos={event.globalPos()}")
        if self.parent:
            print(f"  Parent window pos={self.parent.pos()}, frameGeometry.topLeft={self.parent.frameGeometry().topLeft()}")
        
        if event.button() == Qt.LeftButton:
            # Check if click is on a button
            click_pos = event.pos()
            clicked_on_button = False
            clicked_button_name = None
            
            for btn_name, btn in [("minimize", self.minimize_btn), ("maximize", self.maximize_btn), ("close", self.close_btn)]:
                btn_rect = btn.geometry()
                if btn_rect.contains(click_pos):
                    clicked_on_button = True
                    clicked_button_name = btn_name
                    print(f"  Clicked on button: {btn_name}")
                    break
            
            # If NOT clicking a button, start dragging the parent window
            if not clicked_on_button:
                # Check if we're on Wayland - if so, use startSystemMove() for proper compositor integration
                is_wayland = os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland'
                
                if is_wayland:
                    # On Wayland, use startSystemMove() which delegates to the compositor
                    # This is the only reliable way to move frameless windows on Wayland
                    # Access via windowHandle() - the underlying QWindow object
                    print(f"  Starting drag on Wayland - using startSystemMove()")
                    try:
                        window_handle = self.parent.windowHandle()
                        if window_handle and hasattr(window_handle, 'startSystemMove'):
                            window_handle.startSystemMove()
                            print(f"  Called windowHandle().startSystemMove() - compositor will handle movement")
                            # Don't set dragging=True - compositor handles it
                            event.accept()
                            return
                        else:
                            print(f"  WARNING: windowHandle() or startSystemMove() not available, falling back to manual move")
                            # Fall through to manual move code
                    except (AttributeError, RuntimeError, OSError) as e:
                        # Handle missing windowHandle, system move failures, or OS-level errors
                        print(f"  ERROR: startSystemMove() failed: {e}, falling back to manual move")
                        traceback.print_exc()
                        # Fall through to manual move code
                
                # Manual move() approach (works on X11, may work on some Wayland compositors)
                parent_frame_top_left = self.parent.frameGeometry().topLeft()
                mouse_global = event.globalPos()
                self.drag_position = QPoint(mouse_global.x() - parent_frame_top_left.x(), 
                                            mouse_global.y() - parent_frame_top_left.y())
                self.dragging = True
                session_type = os.environ.get('XDG_SESSION_TYPE', 'unknown')
                print(f"  Starting drag ({session_type}): mouse_global={mouse_global}, parent_frame_top_left={parent_frame_top_left}, drag_position={self.drag_position}")
                event.accept()
                return
        
        # Button click - let default handling
        print(f"  Passing to super().mousePressEvent")
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move - drag parent window if dragging"""
        # Logging: mouse move
        if self.dragging:
            print(f"[TitleBar.mouseMoveEvent] DRAGGING: pos={event.pos()}, globalPos={event.globalPos()}, buttons={event.buttons()}")
        
        if self.dragging and event.buttons() & Qt.LeftButton:
            # Move the parent window directly
            # Formula: new_window_pos = current_mouse_global_pos - drag_offset
            # This keeps the mouse at the same relative position in the window during drag
            mouse_global = event.globalPos()
            new_x = mouse_global.x() - self.drag_position.x()
            new_y = mouse_global.y() - self.drag_position.y()
            print(f"  Moving parent to: ({new_x}, {new_y}) (mouse_global={mouse_global}, drag_position={self.drag_position})")
            # Use move() with explicit x,y coordinates
            # On Wayland: Some compositors restrict client-side window movement, but most allow it for frameless windows
            # On X11: This works reliably
            self.parent.move(new_x, new_y)
            # Verify the move actually happened
            actual_pos = self.parent.pos()
            if actual_pos.x() != new_x or actual_pos.y() != new_y:
                print(f"  WARNING: Move failed! Requested ({new_x}, {new_y}) but window is at {actual_pos}")
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - stop dragging"""
        # Logging: mouse release
        print(f"[TitleBar.mouseReleaseEvent] Button={event.button()}, was_dragging={self.dragging}, pos={event.pos()}, globalPos={event.globalPos()}")
        if event.button() == Qt.LeftButton:
            if self.dragging:
                print(f"  Stopping drag")
            self.dragging = False
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎭 Faceboard Nova")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set frameless window
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        script_dir = Path(__file__).parent
        self.masks_dir = script_dir / "masks"
        os.makedirs(self.masks_dir, exist_ok=True)
        
        self.processor = None
        self.faceboard_window = None
        # Dragging is now handled by TitleBar widget itself, no need for window-level dragging
        self.apply_vaporwave_theme()
        self.setup_ui()
        # Populate virtual camera devices after UI is set up
        self.populate_vcam_devices()
    
    def apply_vaporwave_theme(self):
        """Apply vaporwave/retrowave color theme"""
        # Color palette: Dark bg (#1a1a2e), Hot Pink (#ff1493), Cyan (#00ffff), Purple (#9d00ff), Light text (#e0e0e0)
        theme = """
        QMainWindow {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QMainWindow::title {
            background-color: #1a1a2e;
            color: #00ffff;
            font-weight: bold;
        }
        QWidget {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QGroupBox {
            border: 2px solid #9d00ff;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            color: #00ffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            background-color: #1a1a2e;
        }
        QPushButton {
            background-color: #9d00ff;
            color: #e0e0e0;
            border: 2px solid #ff1493;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #ff1493;
            border-color: #00ffff;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #00ffff;
            border-color: #ff1493;
            color: #1a1a2e;
        }
        QLabel {
            color: #e0e0e0;
        }
        QSlider::groove:horizontal {
            border: 1px solid #9d00ff;
            height: 8px;
            background: #2a2a3e;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #ff1493;
            border: 2px solid #00ffff;
            width: 18px;
            height: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #00ffff;
            border-color: #ff1493;
        }
        QComboBox {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 2px solid #9d00ff;
            border-radius: 4px;
            padding: 5px;
            min-width: 100px;
        }
        QComboBox:hover {
            border-color: #ff1493;
        }
        QComboBox::drop-down {
            border: none;
            background-color: #9d00ff;
        }
        QComboBox QAbstractItemView {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 2px solid #ff1493;
            selection-background-color: #ff1493;
            selection-color: #ffffff;
            /* Fix dropdown positioning for frameless windows */
            position: absolute;
        }
        QSpinBox {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 2px solid #9d00ff;
            border-radius: 4px;
            padding: 5px;
        }
        QSpinBox:hover {
            border-color: #ff1493;
        }
        QCheckBox {
            color: #e0e0e0;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #9d00ff;
            border-radius: 3px;
            background-color: #2a2a3e;
        }
        QCheckBox::indicator:hover {
            border-color: #ff1493;
        }
        QCheckBox::indicator:checked {
            background-color: #ff1493;
            border-color: #00ffff;
        }
        QListWidget {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 2px solid #9d00ff;
            border-radius: 4px;
        }
        QListWidget::item {
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #ff1493;
            color: #ffffff;
        }
        QListWidget::item:hover {
            background-color: #9d00ff;
        }
        QDialog {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QDialog::title {
            background-color: #1a1a2e;
            color: #00ffff;
            font-weight: bold;
        }
        QScrollArea {
            background-color: #1a1a2e;
            border: 2px solid #9d00ff;
            border-radius: 4px;
        }
        QRadioButton {
            color: #e0e0e0;
            spacing: 5px;
        }
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #9d00ff;
            border-radius: 9px;
            background-color: #2a2a3e;
        }
        QRadioButton::indicator:hover {
            border-color: #ff1493;
        }
        QRadioButton::indicator:checked {
            background-color: #ff1493;
            border-color: #00ffff;
        }
        QRadioButton:disabled {
            color: #666666;
        }
        QRadioButton:disabled::indicator {
            border-color: #666666;
            background-color: #1a1a2e;
        }
        """
        self.setStyleSheet(theme)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create custom titlebar
        self.titlebar = TitleBar(self)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with titlebar
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.titlebar)
        
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        central_widget.setLayout(main_layout)
        
        # Left side: Preview
        preview_group = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Click 'Start' to begin")
        self.preview_label.setStyleSheet("background-color: #0a0a1a; border: 2px solid #9d00ff; border-radius: 4px; color: #e0e0e0;")
        preview_layout.addWidget(self.preview_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        self.import_btn = QPushButton("Import Mask Image")
        self.import_btn.clicked.connect(self.import_mask_image)
        
        self.delete_btn = QPushButton("Delete Mask")
        self.delete_btn.clicked.connect(self.delete_mask_image)
        
        self.faceboard_btn = QPushButton("Faceboard")
        self.faceboard_btn.clicked.connect(self.open_faceboard)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.faceboard_btn)
        preview_layout.addLayout(button_layout)
        
        preview_group.setLayout(preview_layout)
        content_layout.addWidget(preview_group, 2)
        
        # Right side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        
        # Global settings
        global_group = QGroupBox("Global Settings")
        global_layout = QVBoxLayout()
        
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = FixedComboBox()
        self.populate_cameras()  # Populate available cameras
        camera_layout.addWidget(self.camera_combo)
        global_layout.addLayout(camera_layout)
        
        max_faces_layout = QHBoxLayout()
        max_faces_layout.addWidget(QLabel("Max Faces:"))
        self.max_faces_group = QButtonGroup(self)
        self.max_faces_radios = []
        for i in range(1, 6):
            radio = QRadioButton(str(i))
            radio.setObjectName(f"max_faces_radio_{i}")
            self.max_faces_group.addButton(radio, i)
            self.max_faces_radios.append(radio)
            max_faces_layout.addWidget(radio)
        # Set fifth radio (max_faces=5) as default
        self.max_faces_radios[4].setChecked(True)
        # Connect signal to update max faces
        self.max_faces_group.buttonClicked.connect(lambda btn: self.on_max_faces_changed(self.max_faces_group.checkedId()))
        global_layout.addLayout(max_faces_layout)
        
        # Virtual camera output toggle
        vcam_layout = QHBoxLayout()
        self.vcam_enable_check = QCheckBox("Enable Virtual Camera Output")
        self.vcam_enable_check.setChecked(True)
        self.vcam_enable_check.setToolTip("Disable this if OBS Virtual Camera conflicts. You can use Window Capture instead.")
        vcam_layout.addWidget(self.vcam_enable_check)
        global_layout.addLayout(vcam_layout)
        
        # Virtual camera device selection
        vcam_device_layout = QHBoxLayout()
        vcam_device_layout.addWidget(QLabel("VCam Device:"))
        self.vcam_device_combo = FixedComboBox()
        self.vcam_device_combo.addItem("Auto (find available)")
        # Will be populated when starting
        vcam_device_layout.addWidget(self.vcam_device_combo)
        global_layout.addLayout(vcam_device_layout)
        
        global_group.setLayout(global_layout)
        controls_layout.addWidget(global_group)
        
        controls_widget.setLayout(controls_layout)
        content_layout.addWidget(controls_widget, 1)
        
        # Face-specific controls - REMOVED (now handled by Faceboard)
        # Face controls are obsolete - all face management is done through the Faceboard
        self.face_groups = []  # Empty list - kept for compatibility but no longer used
        
        controls_widget.setLayout(controls_layout)
        content_layout.addWidget(controls_widget, 1)
        
        # Timer for preview updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
    
    def populate_cameras(self):
        """Populate camera combo box with available cameras"""
        self.camera_combo.clear()
        
        # Method 1: Check /dev/video* devices on Linux
        available_cameras = []
        
        # Check /dev/video* devices
        if os.path.exists('/dev'):
            video_devices = []
            for dev in os.listdir('/dev'):
                if dev.startswith('video') and dev[5:].isdigit():
                    try:
                        dev_num = int(dev[5:])
                        video_devices.append((dev_num, f'/dev/{dev}'))
                    except (ValueError, IndexError):
                        # Invalid device name format - skip
                        pass
            
            video_devices.sort(key=lambda x: x[0])
            
            # Test each device to see if it's a real camera (not v4l2loopback)
            import subprocess
            for dev_num, dev_path in video_devices:
                try:
                    # Check if it's v4l2loopback (skip those)
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', dev_path, '--info'],
                        capture_output=True, text=True, timeout=1
                    )
                    if 'v4l2loopback' in result.stdout.lower():
                        continue  # Skip virtual cameras
                    
                    # Try to open with OpenCV to verify it's a real camera
                    test_cap = cv2.VideoCapture(dev_num)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            # Get camera name if possible
                            camera_name = f"Camera {dev_num}"
                            try:
                                # Try to get more info
                                cap_prop = test_cap.get(cv2.CAP_PROP_BACKEND)
                                camera_name = f"Camera {dev_num} ({dev_path})"
                            except (AttributeError, cv2.error):
                                # Camera property access may fail - use default name
                                pass
                            available_cameras.append((dev_num, camera_name))
                        test_cap.release()
                except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
                    # If v4l2-ctl fails or device doesn't exist, try OpenCV directly
                    try:
                        test_cap = cv2.VideoCapture(dev_num)
                        if test_cap.isOpened():
                            ret, frame = test_cap.read()
                            if ret:
                                available_cameras.append((dev_num, f"Camera {dev_num} ({dev_path})"))
                            test_cap.release()
                    except (cv2.error, OSError, AttributeError):
                        # Camera test may fail - continue to next device
                        pass
        
        # If no cameras found via /dev/video*, try sequential indices
        if not available_cameras:
            for idx in range(10):  # Check first 10 indices
                try:
                    test_cap = cv2.VideoCapture(idx)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            available_cameras.append((idx, f"Camera {idx}"))
                        test_cap.release()
                except (cv2.error, OSError, AttributeError):
                    # Sequential camera enumeration may fail - continue
                    pass
        
        # Populate combo box
        if available_cameras:
            for idx, name in available_cameras:
                self.camera_combo.addItem(name, idx)
        else:
            # Fallback: add default camera 0
            self.camera_combo.addItem("Camera 0 (default)", 0)
        
        # Set default selection
        self.camera_combo.setCurrentIndex(0)
    
    def on_max_faces_changed(self, value):
        """Handle max faces change - disable faces above the limit"""
        if not self.processor:
            return
        
        # Update processor max_faces
        self.processor.max_faces = value
        
        # Disable faces above the limit in mask_settings
        for face_id in range(1, 6):  # Faces 1-5
            if face_id > value:
                # Disable this face
                if face_id in self.processor.mask_settings:
                    self.processor.mask_settings[face_id]['enabled'] = False
            else:
                # Ensure face is enabled if it exists
                if face_id in self.processor.mask_settings:
                    self.processor.mask_settings[face_id]['enabled'] = True
        
        # Update faceboard if open - this will grey out unused face ID radio buttons
        if self.faceboard_window:
            # Update radio button states and refresh faceboard
            self.faceboard_window.update_face_id_radios()
            # If currently selected face ID is now disabled, switch to first enabled one
            current_face_id = self.faceboard_window.face_id_group.checkedId()
            if current_face_id > value:
                # Current selection is now disabled, switch to first enabled
                for radio in self.faceboard_window.face_id_radios[:value]:
                    if radio.isEnabled():
                        radio.setChecked(True)
                        break
            self.faceboard_window.on_face_id_changed()
    
    def populate_vcam_devices(self):
        """Populate virtual camera device combo box with available devices"""
        self.vcam_device_combo.clear()
        self.vcam_device_combo.addItem("Auto (find available)")
        
        import subprocess
        
        # Check for v4l2loopback devices (start from 10, exclude 9)
        for device_idx in range(10, 20):
            device_path = f'/dev/video{device_idx}'
            if os.path.exists(device_path):
                try:
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', device_path, '--info'],
                        capture_output=True, text=True, timeout=1
                    )
                    if 'v4l2loopback' in result.stdout.lower():
                        # Extract device name if available
                        device_name = f"video{device_idx}"
                        self.vcam_device_combo.addItem(f"{device_name} ({device_path})", device_path)
                except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
                    # Device info retrieval may fail - skip this device
                    pass
    
    def update_mask_combo(self, combo):
        """Update mask combo box with available masks"""
        combo.clear()
        if os.path.exists(self.masks_dir):
            masks = sorted([f for f in os.listdir(self.masks_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
            combo.addItems(masks)
    
    def on_mask_changed(self, face_num, idx):
        """Handle mask selection change"""
        if self.processor:
            # Reload masks if needed
            if not self.processor.mask_images:
                self.processor.load_masks()
            
            # Get mask files
            mask_files = sorted([f for f in os.listdir(self.masks_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
            
            if idx >= 0 and idx < len(mask_files):
                mask_name = mask_files[idx]
                # Find mask_idx by name
                mask_idx = None
                for mask_id, name in self.processor.mask_names.items():
                    if name == mask_name:
                        mask_idx = mask_id
                        break
                
                if mask_idx:
                    # Update settings for this face number
                    if face_num in self.processor.mask_settings:
                        self.processor.mask_settings[face_num]['mask_idx'] = mask_idx
                        print(f"Face {face_num} mask changed to: {mask_name} (idx: {mask_idx})")
                    
                    # Also update any tracked faces assigned to this UI face number
                    for face_id, tracked in self.processor.face_tracker.items():
                        if tracked.get('assigned_face_num') == face_num:
                            tracked['mask_idx'] = mask_idx
                            print(f"  Updated tracked face {face_id} to use mask {mask_idx}")
                else:
                    # Fallback: use index + 1
                    fallback_idx = idx + 1
                    if fallback_idx in self.processor.mask_images:
                        if face_num in self.processor.mask_settings:
                            self.processor.mask_settings[face_num]['mask_idx'] = fallback_idx
                        print(f"Face {face_num} mask changed (fallback) to idx: {fallback_idx}")
    
    def update_mask_setting(self, face_num, setting, value):
        """Update mask setting for a specific face"""
        if self.processor:
            # Update settings if they exist
            if face_num in self.processor.mask_settings:
                self.processor.mask_settings[face_num][setting] = value
                print(f"Updated Face {face_num} {setting} to {value}")
            else:
                # Initialize settings if they don't exist
                if not hasattr(self.processor, 'mask_settings'):
                    self.processor.mask_settings = {}
                self.processor.mask_settings[face_num] = {
                    'enabled': True,
                    'mask_idx': 1,
                    'scale': 1.2,
                    'opacity': 0.8
                }
                self.processor.mask_settings[face_num][setting] = value
                print(f"Initialized and updated Face {face_num} {setting} to {value}")
            
            # Also update tracked faces that are assigned to this UI face number
            if hasattr(self.processor, 'face_tracker'):
                for face_id, tracked in self.processor.face_tracker.items():
                    if tracked.get('assigned_face_num') == face_num:
                        # Update the tracked face's settings too
                        if face_num in self.processor.mask_settings:
                            tracked[setting] = self.processor.mask_settings[face_num][setting]
    
    def import_mask_image(self):
        """Import a custom mask image (PNG, JPG, GIF)"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Mask Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.gif *.PNG *.JPG *.JPEG *.GIF)"
        )
        
        if filepath:
            import shutil
            name = os.path.basename(filepath)
            dest_path = os.path.join(self.masks_dir, name)
            
            # Copy file to masks directory
            shutil.copy(filepath, dest_path)
            
            # Reload masks if processor exists
            if self.processor:
                self.processor.load_masks()
            
            # Face groups removed - masks are managed through Faceboard now
            # No need to update combo boxes
            
            # Update faceboard if open
            if self.faceboard_window:
                self.faceboard_window.refresh_buttons()
            
            print(f"Imported mask: {name}")
    
    def delete_mask_image(self):
        """Delete a mask image"""
        if not os.path.exists(self.masks_dir):
            return
        
        # Get list of masks
        mask_files = sorted([f for f in os.listdir(self.masks_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
        
        if not mask_files:
            QMessageBox.information(self, "No Masks", "No masks available to delete.")
            return
        
        # Create dialog to select mask to delete
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Delete Mask")
        dialog.setModal(True)
        layout = QVBoxLayout()
        
        list_widget = QListWidget()
        list_widget.addItems(mask_files)
        layout.addWidget(list_widget)
        
        btn_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(delete_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        
        def do_delete():
            selected_items = list_widget.selectedItems()
            if not selected_items:
                QMessageBox.warning(dialog, "No Selection", "Please select a mask to delete.")
                return
            
            mask_name = selected_items[0].text()
            mask_path = os.path.join(self.masks_dir, mask_name)
            
            # Confirm deletion
            reply = QMessageBox.question(
                dialog, "Confirm Delete", 
                f"Are you sure you want to delete '{mask_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    os.remove(mask_path)
                    print(f"Deleted mask: {mask_name}")
                    
                    # Reload masks if processor exists
                    if self.processor:
                        self.processor.load_masks()
                    
                    # Face groups removed - masks are managed through Faceboard now
                    # Update faceboard if open
                    if self.faceboard_window:
                        self.faceboard_window.refresh_buttons()
                    
                    # Remove from list
                    list_widget.takeItem(list_widget.row(selected_items[0]))
                    
                    QMessageBox.information(dialog, "Success", f"Deleted '{mask_name}'")
                    
                    # Close if no more masks
                    if list_widget.count() == 0:
                        dialog.accept()
                except (OSError, PermissionError, FileNotFoundError) as e:
                    # Handle file system errors when deleting mask files
                    QMessageBox.critical(dialog, "Error", f"Failed to delete mask: {e}")
        
        delete_btn.clicked.connect(do_delete)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()
    
    
    def start_processing(self):
        """Start video processing"""
        if self.processor:
            self.stop_processing()
        
        max_faces = self.max_faces_group.checkedId()
        # Get camera index from combo box
        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            # Fallback to first item's data or default to 0
            camera_index = self.camera_combo.itemData(0)
            if camera_index is None:
                camera_index = 0
        enable_vcam = self.vcam_enable_check.isChecked()
        vcam_device = None
        if self.vcam_device_combo.currentIndex() > 0:
            vcam_device = self.vcam_device_combo.itemData(self.vcam_device_combo.currentIndex())
        
        self.processor = VideoProcessor(
            str(self.masks_dir), 
            max_faces,
            enable_vcam=enable_vcam,
            vcam_device=vcam_device
        )
        self.processor.camera_index = camera_index
        
        # Reload masks to ensure we have latest
        self.processor.load_masks()
        
        # Reset face tracker when starting
        self.processor.face_tracker = {}
        self.processor.next_face_id = 1
        
        # Face settings are now managed entirely through the Faceboard
        # No need to initialize from UI groups (they've been removed)
        # Settings will be created/updated when masks are assigned via Faceboard
        # Initialize default settings for faces 1-5 if they don't exist
        for face_id in range(1, max_faces + 1):
            if face_id not in self.processor.mask_settings:
                self.processor.mask_settings[face_id] = {
                    'enabled': True,
                    'mask_idx': 1,
                    'scale': 1.2,
                    'opacity': 0.8
                }
        
        # Connect frame signal
        self.processor.frame_ready.connect(self.display_frame)
        
        # Start processing
        self.processor.start()
        
        # Enable/disable buttons
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def stop_processing(self):
        """Stop video processing"""
        if self.processor:
            self.processor.stop()
            self.processor.wait()
            self.processor = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_label.setText("Stopped")
    
    def display_frame(self, frame):
        """Display processed frame in preview"""
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit preview while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
    
    def update_preview(self):
        """Update preview (called by timer)"""
        pass  # Frame updates come from signal
    
    
    def open_faceboard(self):
        """Open the faceboard window"""
        if not self.processor:
            QMessageBox.warning(self, "Not Started", "Please start processing first.")
            return
        
        try:
            if self.faceboard_window is None:
                self.faceboard_window = FaceboardWindow(
                    parent=None,  # Independent window
                    processor=self.processor,
                    masks_dir=self.masks_dir
                )
            
            # Position window next to main window
            main_geometry = self.geometry()
            faceboard_x = main_geometry.right() + 20
            faceboard_y = main_geometry.top()
            self.faceboard_window.move(faceboard_x, faceboard_y)
            
            self.faceboard_window.refresh_buttons()
            self.faceboard_window.show()  # Use show() instead of exec_() for non-modal
            self.faceboard_window.raise_()  # Bring to front
            self.faceboard_window.activateWindow()  # Focus
        except (AttributeError, RuntimeError, ImportError) as e:
            # Handle missing attributes, runtime errors, or import issues when opening faceboard
            QMessageBox.critical(self, "Error", f"Failed to open faceboard: {str(e)}")
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle window close"""
        self.stop_processing()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


class ImageButton(QPushButton):
    """Custom button that displays an image thumbnail"""
    def __init__(self, mask_idx=None, mask_name=None, parent=None):
        super().__init__(parent)
        self.mask_idx = mask_idx  # None if empty
        self.mask_name = mask_name
        self.setMinimumSize(100, 100)
        self.setMaximumSize(100, 100)
        self.setIconSize(QPixmap(100, 100).size())
        # Timer for 2-second hold detection
        self.hold_timer = QTimer()
        self.hold_timer.setSingleShot(True)
        self.hold_timer.timeout.connect(self._on_hold_complete)
        self.holding = False
        # Track clicks for double-click detection
        self.last_click_time = 0
        self.click_count = 0
        self.update_display()
    
    def mousePressEvent(self, event):
        """Start hold timer on mouse press, track clicks for double-click"""
        if event.button() == Qt.LeftButton:
            self.holding = True
            # Start 1-second timer
            self.hold_timer.start(1000)  # 1000ms = 1 second
            
            # Track clicks for double-click detection
            current_time = time.time()
            if current_time - self.last_click_time < 0.5:  # 500ms double-click window
                self.click_count += 1
            else:
                self.click_count = 1
            self.last_click_time = current_time
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Cancel hold timer on mouse release, handle single click"""
        if event.button() == Qt.LeftButton:
            if self.holding:
                self.holding = False
                self.hold_timer.stop()
                
                # If it was a single click (not double-click), emit clicked signal
                # Double-click will be handled separately
                if self.click_count == 1:
                    # Small delay to check if this becomes a double-click
                    QTimer.singleShot(300, self._check_single_click)
        super().mouseReleaseEvent(event)
    
    def _check_single_click(self):
        """Check if this was a single click (not part of a double-click)"""
        if self.click_count == 1:
            # It's a single click - emit clicked signal
            self.clicked.emit()
        self.click_count = 0
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click for swap mode"""
        if event.button() == Qt.LeftButton:
            # Cancel hold timer if active
            if self.hold_timer.isActive():
                self.hold_timer.stop()
                self.holding = False
            
            # Reset click count
            self.click_count = 0
            
            # Find button index and emit double-click signal
            parent_window = self.parent()
            while parent_window and not isinstance(parent_window, FaceboardWindow):
                parent_window = parent_window.parent()
            
            if parent_window and hasattr(parent_window, 'button_grid'):
                # Find this button's index
                for row_idx, row in enumerate(parent_window.button_grid):
                    if self in row:
                        col_idx = row.index(self)
                        btn_idx = row_idx * 5 + col_idx
                        parent_window.on_button_double_clicked(btn_idx)
                        return
        super().mouseDoubleClickEvent(event)
    
    def _on_hold_complete(self):
        """Called when 2-second hold completes - trigger image swap"""
        if self.holding:
            self.holding = False
            # Find the button index in the grid
            parent_window = self.parent()
            while parent_window and not isinstance(parent_window, FaceboardWindow):
                parent_window = parent_window.parent()
            
            if parent_window and hasattr(parent_window, 'button_grid'):
                # Find this button's index
                for row_idx, row in enumerate(parent_window.button_grid):
                    if self in row:
                        col_idx = row.index(self)
                        btn_idx = row_idx * 5 + col_idx
                        parent_window.open_image_swap_popup(btn_idx)
                        return
    
    def update_display(self):
        """Update button display with image or placeholder"""
        if self.mask_idx is not None and self.mask_name:
            # Try to load thumbnail from masks directory
            script_dir = Path(__file__).parent
            masks_dir = script_dir / "masks"
            mask_path = os.path.join(masks_dir, self.mask_name)
            
            if os.path.exists(mask_path):
                try:
                    # Load image
                    if self.mask_name.lower().endswith('.gif'):
                        from PIL import Image as PILImage
                        pil_img = PILImage.open(mask_path)
                        if hasattr(pil_img, 'is_animated') and pil_img.is_animated:
                            pil_img.seek(0)  # Get first frame
                        img_array = np.array(pil_img.convert('RGBA'))
                        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
                    else:
                        img_cv = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    
                    if img_cv is not None:
                        # Resize to thumbnail
                        height, width = img_cv.shape[:2]
                        if height > width:
                            new_height = 90
                            new_width = int(width * (90 / height))
                        else:
                            new_width = 90
                            new_height = int(height * (90 / width))
                        
                        thumbnail = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        
                        # Convert to QPixmap
                        if len(thumbnail.shape) == 3:
                            if thumbnail.shape[2] == 4:  # BGRA
                                rgb_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGRA2RGB)
                                height, width, channel = rgb_thumbnail.shape
                                bytes_per_line = 3 * width
                                q_image = QImage(rgb_thumbnail.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            else:  # BGR
                                rgb_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                                height, width, channel = rgb_thumbnail.shape
                                bytes_per_line = 3 * width
                                q_image = QImage(rgb_thumbnail.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            
                            pixmap = QPixmap.fromImage(q_image)
                            self.setIcon(QIcon(pixmap))
                            self.setToolTip(self.mask_name)
                            # Style will be set by update_display, but ensure it's correct
                            if not self.styleSheet() or "border: 3px solid #0ff" not in self.styleSheet():
                                self.setStyleSheet("background-color: #2a2a3e; border: 2px solid #9d00ff; border-radius: 4px;")
                            return
                except (OSError, FileNotFoundError, ValueError, AttributeError) as e:
                    # Handle file I/O errors, missing files, invalid image data, or missing attributes
                    print(f"Error loading thumbnail for {self.mask_name}: {e}")
        
        # Empty button - show placeholder
        self.setIcon(QIcon())
        self.setText("")
        self.setToolTip("Empty slot")
        self.setStyleSheet("background-color: #2a2a3e; border: 2px solid #9d00ff; border-radius: 4px;")


class FaceboardWindow(QDialog):
    """Faceboard interface for quick mask selection"""
    def __init__(self, parent=None, processor=None, masks_dir=None):
        super().__init__(None)  # No parent - independent window
        
        # Validate required parameters BEFORE creating UI
        if processor is None:
            raise ValueError("FaceboardWindow requires a processor")
        if masks_dir is None:
            raise ValueError("FaceboardWindow requires a masks_dir")
        
        self.processor = processor
        self.masks_dir = masks_dir
        self.setWindowTitle("🎭 Mask Faceboard")
        self.setMinimumSize(800, 700)
        # Set frameless window - independent of parent
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        
        # Apply vaporwave theme to match main window
        self.apply_vaporwave_theme()
        
        # Button assignments: list of (mask_idx, mask_name) or None for empty slots
        # 30 buttons total
        self.button_assignments = [None] * 30
        
        # Initialize button_grid as empty list BEFORE setup_ui to prevent AttributeError
        self.button_grid = []
        
        self.setup_ui()
        self.load_button_assignments()
        # Only refresh buttons if button_grid was successfully created
        if self.button_grid:
            self.refresh_buttons()
        
        # Update face ID radio button states based on max_faces
        # Delay to ensure processor is available
        QTimer.singleShot(100, self.update_face_id_radios)
    
    def apply_vaporwave_theme(self):
        """Apply vaporwave/retrowave color theme - same as MainWindow"""
        # Color palette: Dark bg (#1a1a2e), Hot Pink (#ff1493), Cyan (#00ffff), Purple (#9d00ff), Light text (#e0e0e0)
        theme = """
        QDialog {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QWidget {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QPushButton {
            background-color: #9d00ff;
            color: #e0e0e0;
            border: 2px solid #ff1493;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #ff1493;
            border-color: #00ffff;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #00ffff;
            border-color: #ff1493;
            color: #1a1a2e;
        }
        QLabel {
            color: #e0e0e0;
        }
        QSlider::groove:horizontal {
            border: 1px solid #9d00ff;
            height: 8px;
            background: #2a2a3e;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #ff1493;
            border: 2px solid #00ffff;
            width: 18px;
            height: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #00ffff;
            border-color: #ff1493;
        }
        QComboBox {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 2px solid #9d00ff;
            border-radius: 4px;
            padding: 5px;
        }
        QComboBox:hover {
            border-color: #ff1493;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #00ffff;
            width: 0;
            height: 0;
        }
        QComboBox QAbstractItemView {
            background-color: #2a2a3e;
            color: #e0e0e0;
            border: 2px solid #ff1493;
            selection-background-color: #ff1493;
            selection-color: #ffffff;
        }
        QRadioButton {
            color: #e0e0e0;
            spacing: 5px;
        }
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #9d00ff;
            border-radius: 9px;
            background-color: #2a2a3e;
        }
        QRadioButton::indicator:hover {
            border-color: #ff1493;
        }
        QRadioButton::indicator:checked {
            background-color: #ff1493;
            border-color: #00ffff;
        }
        QRadioButton:disabled {
            color: #666666;
        }
        QRadioButton:disabled::indicator {
            border-color: #666666;
            background-color: #1a1a2e;
        }
        """
        self.setStyleSheet(theme)
    
    def setup_ui(self):
        """Setup the faceboard UI"""
        # Create custom titlebar
        self.titlebar = TitleBar(self, "🎭 Mask Faceboard")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.titlebar)
        
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(content_layout)
        
        self.setLayout(layout)
        
        # Top: Scale slider, opacity slider, and Face ID radio buttons
        top_layout = QVBoxLayout()
        
        face_layout = QHBoxLayout()
        face_layout.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(50)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setValue(120)
        self.scale_slider.setMinimumWidth(150)
        self.scale_label = QLabel("1.20")
        self.scale_slider.valueChanged.connect(self.on_scale_changed)
        face_layout.addWidget(self.scale_slider)
        face_layout.addWidget(self.scale_label)
        
        face_layout.addSpacing(20)
        face_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setMinimumWidth(150)
        self.opacity_label = QLabel("0.80")
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        face_layout.addWidget(self.opacity_slider)
        face_layout.addWidget(self.opacity_label)
        
        # Face ID radio buttons to the right of sliders
        face_layout.addSpacing(30)
        face_layout.addWidget(QLabel("Face ID:"))
        self.face_id_group = QButtonGroup(self)
        self.face_id_radios = []
        for i in range(1, 6):
            radio = QRadioButton(str(i))
            radio.setObjectName(f"face_id_radio_{i}")
            self.face_id_group.addButton(radio, i)
            self.face_id_radios.append(radio)
            face_layout.addWidget(radio)
        # Set first radio as default
        self.face_id_radios[0].setChecked(True)
        # Connect signal AFTER UI is fully set up to avoid crashes
        self.face_id_group.buttonClicked.connect(lambda btn: self.on_face_id_changed())
        
        face_layout.addStretch()
        top_layout.addLayout(face_layout)
        
        # Instructions
        instructions = QLabel("Click a button to assign its mask to selected Face ID | Double-click two buttons to swap their positions | Hold for 2 seconds to swap image")
        instructions.setStyleSheet("color: #00ffff; font-size: 9pt; padding: 5px; background-color: #2a2a3e; border: 1px solid #9d00ff; border-radius: 4px;")
        top_layout.addWidget(instructions)
        
        content_layout.addLayout(top_layout)
        
        # Grid of buttons: 6 rows x 5 columns
        # button_grid already initialized in __init__, but ensure it's empty here
        if not hasattr(self, 'button_grid'):
            self.button_grid = []
        grid_layout = QVBoxLayout()
        
        for row in range(6):
            row_layout = QHBoxLayout()
            row_buttons = []
            for col in range(5):
                btn_idx = row * 5 + col
                btn = ImageButton()
                # Use a closure to properly capture btn_idx
                def make_click_handler(idx):
                    return lambda checked: self.on_button_clicked(idx)
                # Single click (on release) = assign mask
                btn.clicked.connect(make_click_handler(btn_idx))
                # Double click is handled via mouseDoubleClickEvent in ImageButton
                row_buttons.append(btn)
                row_layout.addWidget(btn)
            self.button_grid.append(row_buttons)
            grid_layout.addLayout(row_layout)
        
        content_layout.addLayout(grid_layout)
        
        # Bottom: Close button
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)
        
        content_layout.addLayout(bottom_layout)
        
        # Track which button was clicked first for swapping (double-click mode)
        self.swap_mode_active = False
        self.first_swap_idx = None
        
        # Ensure button_grid is initialized before refresh_buttons is called
        # This prevents AttributeError if refresh_buttons is called before setup_ui completes
    
    
    def load_button_assignments(self):
        """Load button assignments from available masks - filter out disabled masks"""
        if not self.processor or not self.processor.mask_images:
            return
        
        # Get all masks sorted by name, but filter out disabled ones
        mask_items = []
        for idx, name in self.processor.mask_names.items():
            # Check if this mask is disabled for ALL faces (if so, exclude it)
            # A mask is "available" if at least one face has it enabled
            is_available = False
            for face_num, settings in self.processor.mask_settings.items():
                if settings.get('mask_idx') == idx and settings.get('enabled', True):
                    is_available = True
                    break
            
            # If no face uses this mask, it's still available
            mask_in_use = any(s.get('mask_idx') == idx for s in self.processor.mask_settings.values())
            if not mask_in_use:
                is_available = True
            
            # Only include available masks
            if is_available:
                mask_items.append((idx, name))
        
        mask_items.sort(key=lambda x: x[1])  # Sort by name
        
        # Assign masks to buttons (first 30)
        for i, (mask_idx, mask_name) in enumerate(mask_items[:30]):
            self.button_assignments[i] = (mask_idx, mask_name)
    
    def refresh_buttons(self):
        """Refresh all button displays"""
        # Ensure button_grid exists
        if not hasattr(self, 'button_grid') or not self.button_grid:
            return
        
        if self.processor:
            # Reload masks if needed (only if load_masks method exists)
            if not self.processor.mask_images and hasattr(self.processor, 'load_masks'):
                self.processor.load_masks()
            
            # Reload assignments
            self.load_button_assignments()
        
        # Update button displays - track indices properly instead of using .index()
        btn_idx = 0
        for row_idx, row in enumerate(self.button_grid):
            for col_idx, btn in enumerate(row):
                if btn_idx < len(self.button_assignments):
                    assignment = self.button_assignments[btn_idx]
                    
                    if assignment:
                        mask_idx, mask_name = assignment
                        btn.mask_idx = mask_idx
                        btn.mask_name = mask_name
                    else:
                        btn.mask_idx = None
                        btn.mask_name = None
                    
                    btn.update_display()
                btn_idx += 1
    
    def on_button_clicked(self, btn_idx):
        """Handle button click - single click assigns mask, double click enables swap mode"""
        # This is called on mouse release, so single click = assign mask
        assignment = self.button_assignments[btn_idx]
        
        if assignment:
            mask_idx, mask_name = assignment
            face_id = self.face_id_group.checkedId()
            self.assign_mask_to_face(face_id, mask_idx)
            
            # Visual feedback - cyan flash
            row = btn_idx // 5
            col = btn_idx % 5
            btn = self.button_grid[row][col]
            btn.setStyleSheet("background-color: #00ffff; border: 3px solid #ff1493; border-radius: 4px;")
            QTimer.singleShot(300, lambda: self._highlight_assigned_mask(face_id))
    
    def on_button_double_clicked(self, btn_idx):
        """Handle button double-click - enable swap mode or swap positions"""
        if not self.swap_mode_active:
            # First double-click - enable swap mode
            self.swap_mode_active = True
            self.first_swap_idx = btn_idx
            # Highlight button
            row = btn_idx // 5
            col = btn_idx % 5
            self.button_grid[row][col].setStyleSheet("background-color: #2a2a3e; border: 3px solid #00ffff; border-radius: 4px;")
        else:
            # Second double-click - swap positions
            if self.first_swap_idx != btn_idx:
                self.swap_buttons(self.first_swap_idx, btn_idx)
            
            # Reset swap mode
            row = self.first_swap_idx // 5
            col = self.first_swap_idx % 5
            self.button_grid[row][col].update_display()
            self.swap_mode_active = False
            self.first_swap_idx = None
    
    def swap_buttons(self, idx1, idx2):
        """Swap assignments between two buttons"""
        self.button_assignments[idx1], self.button_assignments[idx2] = \
            self.button_assignments[idx2], self.button_assignments[idx1]
        
        # Update displays
        row1, col1 = idx1 // 5, idx1 % 5
        row2, col2 = idx2 // 5, idx2 % 5
        
        btn1 = self.button_grid[row1][col1]
        btn2 = self.button_grid[row2][col2]
        
        assignment1 = self.button_assignments[idx1]
        assignment2 = self.button_assignments[idx2]
        
        if assignment1:
            btn1.mask_idx, btn1.mask_name = assignment1
        else:
            btn1.mask_idx, btn1.mask_name = None, None
        
        if assignment2:
            btn2.mask_idx, btn2.mask_name = assignment2
        else:
            btn2.mask_idx, btn2.mask_name = None, None
        
        btn1.update_display()
        btn2.update_display()
        
        print(f"Swapped buttons {idx1} and {idx2}")
    
    def assign_mask_to_face(self, face_id, mask_idx):
        """Assign a mask to a face and capture face identity for persistent recognition"""
        if not self.processor:
            return
        
        # Update processor settings
        if face_id not in self.processor.mask_settings:
            self.processor.mask_settings[face_id] = {
                'enabled': True,
                'mask_idx': mask_idx,
                'scale': 1.2,
                'opacity': 0.8
            }
        else:
            self.processor.mask_settings[face_id]['mask_idx'] = mask_idx
        
        # Also update tracked faces assigned to this UI face number
        # And capture face identity for persistent recognition
        if hasattr(self.processor, 'face_tracker'):
            for face_track_id, tracked in self.processor.face_tracker.items():
                if tracked.get('assigned_face_num') == face_id:
                    tracked['mask_idx'] = mask_idx
                    # Capture face identity if we have the current frame and landmarks
                    # We'll need to capture this in process_frame when the mask is applied
        
        mask_name = self.processor.mask_names.get(mask_idx, f"mask_{mask_idx}")
        print(f"Assigned mask '{mask_name}' (idx: {mask_idx}) to Face {face_id}")
        
        # Signal processor to capture face identity on next frame
        if hasattr(self.processor, 'pending_identity_capture'):
            self.processor.pending_identity_capture[face_id] = mask_idx
        
        # Signal processor to capture face identity on next frame
        if hasattr(self.processor, 'pending_identity_capture'):
            self.processor.pending_identity_capture[face_id] = mask_idx
    
    def update_face_id_radios(self):
        """Update face ID radio button enabled/disabled states based on max_faces"""
        if not hasattr(self, 'face_id_radios'):
            return
        
        # Get max_faces from processor or parent window
        max_faces = 5
        if self.processor and hasattr(self.processor, 'max_faces'):
            max_faces = self.processor.max_faces
        elif hasattr(self.parent(), 'max_faces_group'):
            max_faces = self.parent().max_faces_group.checkedId()
        
        for idx, radio in enumerate(self.face_id_radios):
            face_id = idx + 1
            if face_id > max_faces:
                # Grey out unused face IDs
                radio.setEnabled(False)
                radio.setStyleSheet("color: #666666;")
                # If this radio was selected and it's being disabled, switch to first enabled one
                if radio.isChecked():
                    for r in self.face_id_radios[:max_faces]:
                        if r.isEnabled():
                            r.setChecked(True)
                            break
            else:
                # Enable used face IDs
                radio.setEnabled(True)
                radio.setStyleSheet("")  # Use default theme styling
    
    def on_face_id_changed(self):
        """Update scale and opacity sliders when Face ID changes, and highlight the assigned mask button"""
        # Guard against being called before UI is fully initialized
        if not hasattr(self, 'face_id_group') or not hasattr(self, 'scale_slider'):
            return
        
        try:
            face_id = self.face_id_group.checkedId()
            if self.processor and face_id in self.processor.mask_settings:
                scale = self.processor.mask_settings[face_id].get('scale', 1.2)
                self.scale_slider.setValue(int(scale * 100))
                self.scale_label.setText(f"{scale:.2f}")
                
                opacity = self.processor.mask_settings[face_id].get('opacity', 0.8)
                self.opacity_slider.setValue(int(opacity * 100))
                self.opacity_label.setText(f"{opacity:.2f}")
            else:
                # Default values
                self.scale_slider.setValue(120)
                self.scale_label.setText("1.20")
                self.opacity_slider.setValue(80)
                self.opacity_label.setText("0.80")
            
            # Update button highlights to show which mask is assigned to this face
            self._highlight_assigned_mask(face_id)
        except (AttributeError, ValueError):
            # Handle errors during initialization (UI elements may not be ready yet)
            pass
    
    def _highlight_assigned_mask(self, face_id):
        """Highlight the button that has the mask assigned to the given face ID"""
        if not hasattr(self, 'button_grid') or not self.button_grid:
            return
        
        if not self.processor or face_id not in self.processor.mask_settings:
            # No mask assigned - clear all highlights
            for row in self.button_grid:
                for btn in row:
                    btn.update_display()  # This will reset the style
            return
        
        # Get the mask_idx assigned to this face
        assigned_mask_idx = self.processor.mask_settings[face_id].get('mask_idx')
        if assigned_mask_idx is None:
            # No mask assigned - clear all highlights
            for row in self.button_grid:
                for btn in row:
                    btn.update_display()
            return
        
        # Find which button has this mask_idx and highlight it
        btn_idx = 0
        for row_idx, row in enumerate(self.button_grid):
            for col_idx, btn in enumerate(row):
                if btn_idx < len(self.button_assignments):
                    assignment = self.button_assignments[btn_idx]
                    if assignment:
                        mask_idx, mask_name = assignment
                        if mask_idx == assigned_mask_idx:
                            # This is the assigned mask - highlight it with cyan border
                            btn.setStyleSheet("background-color: #2a2a3e; border: 3px solid #00ffff; border-radius: 4px;")
                        else:
                            # Not the assigned mask - use normal style
                            btn.update_display()
                    else:
                        btn.update_display()
                btn_idx += 1
    
    def on_scale_changed(self, value):
        """Update scale for selected Face ID"""
        # Guard against being called before UI is fully initialized
        if not hasattr(self, 'scale_label') or not hasattr(self, 'face_id_group'):
            return
        
        scale_val = value / 100.0
        self.scale_label.setText(f"{scale_val:.2f}")
        
        if not self.processor:
            return
        
        try:
            face_id = self.face_id_group.checkedId()
            if face_id not in self.processor.mask_settings:
                opacity = self.opacity_slider.value() / 100.0
                self.processor.mask_settings[face_id] = {
                    'enabled': True,
                    'mask_idx': 1,
                    'scale': scale_val,
                    'opacity': opacity
                }
            else:
                self.processor.mask_settings[face_id]['scale'] = scale_val
        except (AttributeError, ValueError):
            # Settings may not be initialized yet or face_id invalid - ignore during UI updates
            pass
    
    def on_opacity_changed(self, value):
        """Update opacity for selected Face ID"""
        # Guard against being called before UI is fully initialized
        if not hasattr(self, 'opacity_label') or not hasattr(self, 'face_id_group'):
            return
        
        opacity_val = value / 100.0
        self.opacity_label.setText(f"{opacity_val:.2f}")
        
        if not self.processor:
            return
        
        try:
            face_id = self.face_id_group.checkedId()
            if face_id not in self.processor.mask_settings:
                scale = self.scale_slider.value() / 100.0
                self.processor.mask_settings[face_id] = {
                    'enabled': True,
                    'mask_idx': 1,
                    'scale': scale,
                    'opacity': opacity_val
                }
            else:
                self.processor.mask_settings[face_id]['opacity'] = opacity_val
        except (AttributeError, ValueError):
            # Settings may not be initialized yet - ignore
            pass
    
    def open_image_swap_popup(self, btn_idx):
        """Open popup to swap button image with another from library"""
        if not self.processor:
            return
        
        if not hasattr(self, 'button_grid') or not self.button_grid:
            return
        
        swap_window = ImageSwapWindow(
            parent=None,  # Independent window
            processor=self.processor,
            masks_dir=self.masks_dir,
            target_button_idx=btn_idx,
            current_assignment=self.button_assignments[btn_idx] if btn_idx < len(self.button_assignments) else None
        )
        
        if swap_window.exec_() == QDialog.Accepted:
            # Swap was successful - update button
            new_assignment = swap_window.selected_assignment
            if new_assignment:
                if btn_idx < len(self.button_assignments):
                    self.button_assignments[btn_idx] = new_assignment
                row = btn_idx // 5
                col = btn_idx % 5
                if row < len(self.button_grid) and col < len(self.button_grid[row]):
                    btn = self.button_grid[row][col]
                    if new_assignment:
                        btn.mask_idx, btn.mask_name = new_assignment
                    else:
                        btn.mask_idx, btn.mask_name = None, None
                    btn.update_display()
                    print(f"Swapped button {btn_idx} to {new_assignment}")


class ImageSwapWindow(QDialog):
    """Popup window for swapping button images - visual library selection"""
    def __init__(self, parent=None, processor=None, masks_dir=None, target_button_idx=None, current_assignment=None):
        super().__init__(None)  # No parent - independent window
        self.processor = processor
        self.masks_dir = masks_dir
        self.target_button_idx = target_button_idx
        self.current_assignment = current_assignment
        self.selected_assignment = None
        
        self.setWindowTitle("🔄 Swap Image - Select Replacement")
        self.setMinimumSize(900, 700)
        # Set frameless window - independent of parent
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        
        # Apply vaporwave theme to match main window
        self.apply_vaporwave_theme()
        
        # Dragging is now handled by TitleBar widget itself, no need for window-level dragging
        
        self.setup_ui()
        self.load_all_masks()
    
    def apply_vaporwave_theme(self):
        """Apply vaporwave/retrowave color theme - same as MainWindow"""
        # Color palette: Dark bg (#1a1a2e), Hot Pink (#ff1493), Cyan (#00ffff), Purple (#9d00ff), Light text (#e0e0e0)
        theme = """
        QDialog {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QWidget {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QPushButton {
            background-color: #9d00ff;
            color: #e0e0e0;
            border: 2px solid #ff1493;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
            min-width: 100px;
        }
        QPushButton:hover {
            background-color: #ff1493;
            border-color: #00ffff;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #00ffff;
            border-color: #ff1493;
            color: #1a1a2e;
        }
        QLabel {
            color: #e0e0e0;
        }
        QScrollArea {
            background-color: #1a1a2e;
            border: 2px solid #9d00ff;
            border-radius: 4px;
        }
        """
        self.setStyleSheet(theme)
    
    def setup_ui(self):
        """Setup the image swap UI"""
        # Create custom titlebar
        self.titlebar = TitleBar(self, "🔄 Swap Image")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.titlebar)
        
        content_layout = QVBoxLayout()
        layout.addLayout(content_layout)
        
        self.setLayout(layout)
        
        # Instructions
        info_label = QLabel("Double-click an image to swap it with the selected button")
        info_label.setStyleSheet("color: #00ffff; font-size: 10pt; padding: 10px; background-color: #2a2a3e; border: 2px solid #9d00ff; border-radius: 4px; font-weight: bold;")
        content_layout.addWidget(info_label)
        
        # Scroll area for image grid
        scroll = QWidget()
        scroll_layout = QVBoxLayout()
        scroll.setLayout(scroll_layout)
        
        # Grid of image buttons - show all available masks
        self.image_grid = []
        grid_layout = QVBoxLayout()
        
        # Calculate grid size (5 columns)
        masks_list = self.get_available_masks()
        num_masks = len(masks_list)
        num_rows = (num_masks + 4) // 5  # Round up
        
        for row in range(num_rows):
            row_layout = QHBoxLayout()
            row_buttons = []
            for col in range(5):
                idx = row * 5 + col
                if idx < num_masks:
                    mask_idx, mask_name = masks_list[idx]
                    btn = ImageButton(mask_idx=mask_idx, mask_name=mask_name)
                    btn.clicked.connect(lambda checked, assignment=(mask_idx, mask_name): self.on_image_selected(assignment))
                    row_buttons.append(btn)
                    row_layout.addWidget(btn)
                else:
                    # Empty slot
                    spacer = QWidget()
                    spacer.setMinimumSize(100, 100)
                    row_layout.addWidget(spacer)
            self.image_grid.append(row_buttons)
            grid_layout.addLayout(row_layout)
        
        scroll_layout.addLayout(grid_layout)
        scroll_layout.addStretch()
        
        # Add scroll area
        scroll_area = QWidget()
        scroll_area_layout = QVBoxLayout()
        scroll_area_layout.addWidget(scroll)
        scroll_area.setLayout(scroll_area_layout)
        
        # Use QScrollArea
        scroll_widget = QScrollArea()
        scroll_widget.setWidget(scroll)
        scroll_widget.setWidgetResizable(True)
        content_layout.addWidget(scroll_widget)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)
        
        content_layout.addLayout(bottom_layout)
        
        self.selected_image = None
    
    
    def get_available_masks(self):
        """Get all available masks (including disabled ones for swap)"""
        if not self.processor or not self.processor.mask_images:
            return []
        
        mask_items = []
        for idx, name in self.processor.mask_names.items():
            mask_items.append((idx, name))
        
        mask_items.sort(key=lambda x: x[1])  # Sort by name
        return mask_items
    
    def load_all_masks(self):
        """Load all masks into the grid"""
        # Already done in setup_ui - this method kept for API compatibility
        pass
    
    def on_image_selected(self, assignment):
        """Handle image selection - single click to confirm"""
        # Single click selects the image
        self.selected_assignment = assignment
        self.accept()


if __name__ == "__main__":
    main()
