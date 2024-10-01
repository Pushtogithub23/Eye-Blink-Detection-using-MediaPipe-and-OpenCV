import cv2 as cv



def draw_overlay(frame, pt1, pt2, alpha=0.25, color=(51, 68, 255), filled=True):
    overlay = frame.copy()
    rect_color = color if filled else (0, 0, 0)
    cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_rounded_rect(img, bbox, line_color=(255, 255, 255), ellipse_color=(0, 0, 255), line_thickness=2,
                      ellipse_thickness=3, radius=15):
    """ Draw a rectangle with rounded corners, allowing separate colors and thicknesses for lines and ellipses. """
    x1, y1, x2, y2 = bbox

    # Draw straight lines between corner circles (rectangular part)
    cv.line(img, (x1 + radius, y1), (x2 - radius, y1), line_color, line_thickness)
    cv.line(img, (x1 + radius, y2), (x2 - radius, y2), line_color, line_thickness)
    cv.line(img, (x1, y1 + radius), (x1, y2 - radius), line_color, line_thickness)
    cv.line(img, (x2, y1 + radius), (x2, y2 - radius), line_color, line_thickness)

    # Draw arcs at the corners (ellipse part)
    cv.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, ellipse_color, ellipse_thickness)  # Top-left corner
    cv.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, ellipse_color, ellipse_thickness)  # Top-right corner
    cv.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, ellipse_color, ellipse_thickness)   # Bottom-left corner
    cv.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, ellipse_color, ellipse_thickness)     # Bottom-right corner


def draw_text_with_bg(frame, text, pos, font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.3, thickness=1, bg_color=(255, 255, 255),
                      text_color=(0, 0, 0)):
    """ Draws text with a background rectangle. """
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, cv.FILLED)
    cv.putText(frame, text, (x, y), font, font_scale, text_color, thickness, lineType=cv.LINE_AA)

