import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
import time
import ast
import operator as op
import math

# Safe math operations
def safe_eval(expr):
    """Safely evaluate mathematical expressions"""
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return _eval(node.left) + _eval(node.right)
            elif isinstance(node.op, ast.Sub):
                return _eval(node.left) - _eval(node.right)
            elif isinstance(node.op, ast.Mult):
                return _eval(node.left) * _eval(node.right)
            elif isinstance(node.op, ast.Div):
                return _eval(node.left) / _eval(node.right)
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")
    
    try:
        tree = ast.parse(expr, mode='eval')
        return _eval(tree.body)
    except:
        return "Error"

def euclidean(pt1, pt2):
    """Calculate Euclidean distance between two points"""
    d = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return d

class HandCalculator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Calculator display settings
        self.display_text = ""
        self.result = None
        self.operators = ["+", "-", "*", "/", "C", "="]
        self.current_operator = None
        self.waiting_for_second_number = False  # New flag to track input state
        
        # Frame settings
        self.frame_size = (800, 1200)
        
        # ROI settings for number input
        self.roi_left = (50, 250, 450, 750)
        self.roi_right = (750, 250, 1150, 750)
        
        # Gesture recognition settings
        self.last_number_time = 0
        self.number_cooldown = 1.0  # Increased from 0.2 to 1.0 second
        self.last_operator_time = 0
        self.operator_cooldown = 0.1
        
        # Gesture shortcuts
        self.gesture_shortcuts = {
            "draw": False,      # Index finger extended
            "navigate": False,  # Two fingers extended
            "submit": False     # Pinky extended
        }
        
        # Initialize operator panel
        self.create_operator_panel()
        
    def create_operator_panel(self):
        """Create the operator panel at the top of the frame"""
        self.op_panel = np.zeros((150, self.frame_size[1], 3), dtype=np.uint8)
        cols = np.linspace(0, self.frame_size[1]-1, len(self.operators)+1).astype(np.int64)
        cols = [(cols[i], cols[i+1]) for i in range(len(cols)-1)]
        
        for i, col in enumerate(cols):
            # Highlight selected operator
            if self.operators[i] == self.current_operator:
                self.op_panel[:, col[0]:col[1]] = [100, 100, 100]
            else:
                self.op_panel[:, col[0]:col[1]] = [50, 50, 50]
            
            self.op_panel[:, col[0]-2:col[0]+2] = [200, 200, 200]
            self.op_panel[:, col[1]-2:col[1]+2] = [200, 200, 200]
            cv2.putText(self.op_panel, self.operators[i], 
                       (int((col[0]+col[1])/2), 90), 
                       cv2.FONT_HERSHEY_COMPLEX, 1.5, (200, 100, 200), 3)
    
    def check_gesture_shortcuts(self, landmarks, frame_shape):
        """Check for gesture shortcuts"""
        # Get finger landmarks
        index_tip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[8].x, landmarks.landmark[8].y,
            frame_shape[1], frame_shape[0])
        index_pip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[6].x, landmarks.landmark[6].y,
            frame_shape[1], frame_shape[0])
        
        middle_tip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[12].x, landmarks.landmark[12].y,
            frame_shape[1], frame_shape[0])
        middle_pip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[10].x, landmarks.landmark[10].y,
            frame_shape[1], frame_shape[0])
        
        pinky_tip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[20].x, landmarks.landmark[20].y,
            frame_shape[1], frame_shape[0])
        pinky_pip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[18].x, landmarks.landmark[18].y,
            frame_shape[1], frame_shape[0])
        
        # Check for draw mode (index finger extended)
        if index_tip and index_pip:
            self.gesture_shortcuts["draw"] = index_tip[1] < index_pip[1]
        
        # Check for navigate mode (two fingers extended)
        if index_tip and index_pip and middle_tip and middle_pip:
            self.gesture_shortcuts["navigate"] = (
                index_tip[1] < index_pip[1] and
                middle_tip[1] < middle_pip[1]
            )
        
        # Check for submit mode (pinky extended)
        if pinky_tip and pinky_pip:
            self.gesture_shortcuts["submit"] = pinky_tip[1] < pinky_pip[1]
    
    def count_fingers(self, landmarks, frame_shape):
        """Count the number of fingers that are up with improved accuracy and support for numbers 6-9"""
        finger_count = 0
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]  # PIP joints
        finger_mcps = [5, 9, 13, 17]   # MCP joints
        
        # Convert landmarks to pixel coordinates
        tips = []
        pips = []
        mcps = []
        
        for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
            tip_coords = self.mp_draw._normalized_to_pixel_coordinates(
                landmarks.landmark[tip].x,
                landmarks.landmark[tip].y,
                frame_shape[1], frame_shape[0])
            pip_coords = self.mp_draw._normalized_to_pixel_coordinates(
                landmarks.landmark[pip].x,
                landmarks.landmark[pip].y,
                frame_shape[1], frame_shape[0])
            mcp_coords = self.mp_draw._normalized_to_pixel_coordinates(
                landmarks.landmark[mcp].x,
                landmarks.landmark[mcp].y,
                frame_shape[1], frame_shape[0])
            
            if tip_coords and pip_coords and mcp_coords:
                tips.append(tip_coords)
                pips.append(pip_coords)
                mcps.append(mcp_coords)
        
        # Get thumb coordinates
        thumb_tip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[4].x,
            landmarks.landmark[4].y,
            frame_shape[1], frame_shape[0])
        thumb_ip = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[3].x,
            landmarks.landmark[3].y,
            frame_shape[1], frame_shape[0])
        thumb_mcp = self.mp_draw._normalized_to_pixel_coordinates(
            landmarks.landmark[2].x,
            landmarks.landmark[2].y,
            frame_shape[1], frame_shape[0])

        # Basic finger counting (0-5)
        basic_count = 0
        for tip, pip, mcp in zip(tips, pips, mcps):
            is_extended = (
                tip[1] < pip[1] and  # Tip is above PIP
                tip[1] < mcp[1] and  # Tip is above MCP
                euclidean(tip, pip) > euclidean(pip, mcp) * 0.5  # Finger is extended enough
            )
            if is_extended:
                basic_count += 1

        # Check thumb
        if thumb_tip and thumb_ip and thumb_mcp:
            is_thumb_extended = (
                thumb_tip[0] > thumb_ip[0] and  # Thumb is to the right
                thumb_tip[0] > thumb_mcp[0] and  # Thumb is to the right of MCP
                euclidean(thumb_tip, thumb_ip) > euclidean(thumb_ip, thumb_mcp) * 0.5  # Thumb is extended enough
            )
            if is_thumb_extended:
                basic_count += 1

        # Extended gesture recognition for numbers 6-9
        if basic_count == 5:  # All fingers extended
            # Check for number 6: All fingers extended + thumb touching index finger
            if thumb_tip and tips[0]:  # tips[0] is index finger
                if euclidean(thumb_tip, tips[0]) < 50:  # Close enough to be touching
                    return 6
            
            # Check for number 7: All fingers extended + thumb touching middle finger
            if thumb_tip and tips[1]:  # tips[1] is middle finger
                if euclidean(thumb_tip, tips[1]) < 50:
                    return 7
            
            # Check for number 8: All fingers extended + thumb touching ring finger
            if thumb_tip and tips[2]:  # tips[2] is ring finger
                if euclidean(thumb_tip, tips[2]) < 50:
                    return 8
            
            # Check for number 9: All fingers extended + thumb touching pinky
            if thumb_tip and tips[3]:  # tips[3] is pinky
                if euclidean(thumb_tip, tips[3]) < 50:
                    return 9

        return basic_count
    
    def process_frame(self, frame):
        """Process each frame and update calculator state"""
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, self.frame_size[::-1])
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # Draw ROIs with labels
        roi_color = (0, 0, 255)  # Default red
        if self.waiting_for_second_number:
            roi_color = (255, 165, 0)  # Orange to indicate waiting for second number
            
        cv2.rectangle(frame, self.roi_left[:2], self.roi_left[2:], roi_color, 2)
        cv2.rectangle(frame, self.roi_right[:2], self.roi_right[2:], roi_color, 2)
        cv2.putText(frame, "Left Number Input", (self.roi_left[0], self.roi_left[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        cv2.putText(frame, "Right Number Input", (self.roi_right[0], self.roi_right[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
        
        current_time = time.time()
        values = []
        move = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check gesture shortcuts
                self.check_gesture_shortcuts(hand_landmarks, frame.shape)
                
                # Get landmarks
                h, w = frame.shape[:-1]
                index_tip = self.mp_draw._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[8].x,
                    hand_landmarks.landmark[8].y,
                    w, h)
                index_pip = self.mp_draw._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[6].x,
                    hand_landmarks.landmark[6].y,
                    w, h)
                thumb_tip = self.mp_draw._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[4].x,
                    hand_landmarks.landmark[4].y,
                    w, h)
                wrist = self.mp_draw._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[0].x,
                    hand_landmarks.landmark[0].y,
                    w, h)
                
                if index_tip and index_pip and thumb_tip:
                    # Check for operator selection
                    if index_tip[1] < 150 and current_time - self.last_operator_time > self.operator_cooldown:
                        cols = np.linspace(0, self.frame_size[1]-1, len(self.operators)+1).astype(np.int64)
                        cols = [(cols[i], cols[i+1]) for i in range(len(cols)-1)]
                        for i, col in enumerate(cols):
                            if col[0] < index_tip[0] < col[1]:
                                self.current_operator = self.operators[i]
                                self.last_operator_time = current_time
                                self.create_operator_panel()
                                break
                    
                    # Check for move mode
                    if euclidean(index_pip, thumb_tip) < 120:
                        move = True
                
                # Count fingers if wrist is in ROI
                if wrist:
                    # Draw wrist position indicator
                    cv2.circle(frame, (wrist[0], wrist[1]), 10, (255, 0, 0), -1)
                    
                    # Check if wrist is in ROI
                    in_left_roi = (self.roi_left[0] < wrist[0] < self.roi_left[2] and 
                                 self.roi_left[1] < wrist[1] < self.roi_left[3])
                    in_right_roi = (self.roi_right[0] < wrist[0] < self.roi_right[2] and 
                                  self.roi_right[1] < wrist[1] < self.roi_right[3])
                    
                    if in_left_roi or in_right_roi:
                        # Highlight the active ROI
                        active_roi_color = (0, 255, 0)  # Green for active ROI
                        if in_left_roi:
                            cv2.rectangle(frame, self.roi_left[:2], self.roi_left[2:], active_roi_color, 2)
                        if in_right_roi:
                            cv2.rectangle(frame, self.roi_right[:2], self.roi_right[2:], active_roi_color, 2)
                        
                        finger_count = self.count_fingers(hand_landmarks, frame.shape)
                        if not move and current_time - self.last_number_time > self.number_cooldown:
                            values.append(finger_count)
                            self.last_number_time = current_time
                            # Draw feedback circle
                            cv2.circle(frame, (wrist[0], wrist[1]), 20, (0, 255, 0), -1)
                            # Show detected number
                            cv2.putText(frame, f"Detected: {finger_count}", 
                                      (wrist[0]-30, wrist[1]-30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process operator
        if self.current_operator and current_time - self.last_operator_time > self.operator_cooldown:
            if self.current_operator == "C":
                self.display_text = ""
                self.result = None
                self.waiting_for_second_number = False
            elif self.current_operator == "=":
                try:
                    self.result = safe_eval(self.display_text)
                    self.display_text = str(self.result)
                    self.waiting_for_second_number = False
                except:
                    self.display_text = "Error"
            else:
                if self.display_text and self.display_text[-1] not in self.operators:
                    self.display_text += self.current_operator
                    self.waiting_for_second_number = True  # Set flag when operator is added
            self.current_operator = None
            self.create_operator_panel()
        
        # Process number input
        if values and not move:
            d = dict(Counter(values))
            pred = sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0]
            if pred > 0:
                # If we have a result and no operator, clear the display
                if self.result is not None and not self.waiting_for_second_number:
                    self.display_text = ""
                    self.result = None
                
                # Add the number to the display
                self.display_text += str(pred)
                
                # If we were waiting for second number, reset the flag
                if self.waiting_for_second_number:
                    self.waiting_for_second_number = False
        
        # Add operator panel
        frame[0:150] = self.op_panel
        
        # Display current text and result
        cv2.putText(frame, "Current Input:", (100, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, self.display_text, (100, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Display input state
        if self.waiting_for_second_number:
            cv2.putText(frame, "Waiting for second number...", (100, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Display gesture shortcuts status
        shortcut_y = 350
        for gesture, active in self.gesture_shortcuts.items():
            color = (0, 255, 0) if active else (0, 0, 255)
            cv2.putText(frame, f"{gesture}: {'Active' if active else 'Inactive'}", 
                       (100, shortcut_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            shortcut_y += 30
        
        return frame

def main():
    calculator = HandCalculator()
    cap = cv2.VideoCapture(0)
    
    # Set window size
    cv2.namedWindow('Hand Calculator', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Calculator', calculator.frame_size[1], calculator.frame_size[0])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = calculator.process_frame(frame)
        cv2.imshow('Hand Calculator', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 