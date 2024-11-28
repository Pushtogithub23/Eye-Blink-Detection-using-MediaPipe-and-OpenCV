import cv2 as cv
import numpy as np
from typing import Tuple, Union, Optional

class DrawingUtils:
    """Utility class for OpenCV drawing operations with enhanced error handling."""

    @staticmethod
    def draw_overlay(
        frame: np.ndarray, 
        pt1: Tuple[int, int], 
        pt2: Tuple[int, int], 
        alpha: float = 0.25, 
        color: Tuple[int, int, int] = (51, 68, 255), 
        filled: bool = True
    ) -> None:
        """
        Draw a semi-transparent overlay on the frame.

        Args:
            frame (np.ndarray): Input image
            pt1 (tuple): Top-left corner coordinates
            pt2 (tuple): Bottom-right corner coordinates
            alpha (float): Transparency level (0-1)
            color (tuple): Rectangle color (B,G,R)
            filled (bool): Whether to fill the rectangle

        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input types are incorrect
        """
        # Input validation
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy array")
        
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
        
        try:
            overlay = frame.copy()
            rect_color = color if filled else (0, 0, 0)
            cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
            cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        except Exception as e:
            raise RuntimeError(f"Error drawing overlay: {str(e)}")

    @staticmethod
    def draw_rounded_rect(
        img: np.ndarray, 
        bbox: Union[Tuple[int, int, int, int], list], 
        line_color: Tuple[int, int, int] = (255, 255, 255), 
        ellipse_color: Tuple[int, int, int] = (0, 0, 255), 
        line_thickness: int = 2,
        ellipse_thickness: int = 3, 
        radius: int = 15
    ) -> None:
        """
        Draw a rectangle with rounded corners.

        Args:
            img (np.ndarray): Input image
            bbox (tuple/list): Bounding box coordinates (x1, y1, x2, y2)
            line_color (tuple): Color for straight lines
            ellipse_color (tuple): Color for corner ellipses
            line_thickness (int): Thickness of straight lines
            ellipse_thickness (int): Thickness of corner ellipses
            radius (int): Radius of corner rounding

        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input types are incorrect
        """
        # Input validation
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if len(bbox) != 4:
            raise ValueError("Bounding box must contain 4 coordinates")
        
        x1, y1, x2, y2 = bbox

        try:
            # Draw straight lines
            cv.line(img, (x1 + radius, y1), (x2 - radius, y1), line_color, line_thickness)
            cv.line(img, (x1 + radius, y2), (x2 - radius, y2), line_color, line_thickness)
            cv.line(img, (x1, y1 + radius), (x1, y2 - radius), line_color, line_thickness)
            cv.line(img, (x2, y1 + radius), (x2, y2 - radius), line_color, line_thickness)

            # Draw corner ellipses
            corner_points = [
                ((x1 + radius, y1 + radius), 180),
                ((x2 - radius, y1 + radius), 270),
                ((x1 + radius, y2 - radius), 90),
                ((x2 - radius, y2 - radius), 0)
            ]

            for (center, angle) in corner_points:
                cv.ellipse(img, center, (radius, radius), angle, 0, 90, ellipse_color, ellipse_thickness)
        
        except Exception as e:
            raise RuntimeError(f"Error drawing rounded rectangle: {str(e)}")

    @staticmethod
    def draw_text_with_bg(
        frame: np.ndarray, 
        text: str, 
        pos: Tuple[int, int], 
        font: int = cv.FONT_HERSHEY_SIMPLEX, 
        font_scale: float = 0.3, 
        thickness: int = 1, 
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> None:
        """
        Draw text with a background rectangle.

        Args:
            frame (np.ndarray): Input image
            text (str): Text to draw
            pos (tuple): Starting position of text
            font (int): OpenCV font type
            font_scale (float): Font scale factor
            thickness (int): Line thickness
            bg_color (tuple): Background rectangle color
            text_color (tuple): Text color

        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input types are incorrect
        """
        # Input validation
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy array")
        
        if not text:
            raise ValueError("Text cannot be empty")

        try:
            # Calculate text size
            (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
            x, y = pos

            # Draw background rectangle
            cv.rectangle(
                frame, 
                (x, y - text_height - baseline), 
                (x + text_width, y + baseline), 
                bg_color, 
                cv.FILLED
            )

            # Draw text
            cv.putText(
                frame, 
                text, 
                (x, y), 
                font, 
                font_scale, 
                text_color, 
                thickness, 
                lineType=cv.LINE_AA
            )
        
        except Exception as e:
            raise RuntimeError(f"Error drawing text with background: {str(e)}")

# Example usage
def main():
    try:
        # Create a sample image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Demonstrate drawing methods with error handling
        DrawingUtils.draw_overlay(img, (50, 50), (200, 200))
        DrawingUtils.draw_rounded_rect(img, (100, 100, 250, 250))
        DrawingUtils.draw_text_with_bg(img, "Hello, OpenCV!", (50, 50))
        
        cv.imshow('Drawing Utilities Demo', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()