
from PIL import Image
import cv2, numpy as np

class RoiLBC:
    config = None

    @staticmethod
    def main_method(roi_image):

        edges = RoiLBC.apply_canny_edge_detection(roi_image, 2, 100)
        final_box = RoiLBC.find_bounding_boxes(edges,roi_image)

        # Uncomment the following lines to save the output image with the bounding box drawn
        # basename = os.path.splitext(os.path.basename(input_path))[0]
        # output_path = f"{basename}_output.jpg"
        # original = Image.open(input_path).convert('RGBA')
        # draw = ImageDraw.Draw(original)
        # draw.rectangle([final_box.left, final_box.top, final_box.right, final_box.bottom], outline="green", width=3)
        # original.save(output_path)

        return final_box

    def apply_canny_edge_detection(image, low_threshold, high_threshold):

        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(image_np, low_threshold, high_threshold)

        return Image.fromarray(edges)



    @staticmethod
    def find_bounding_boxes(edge_image, original_image):
 
        edge_np = np.array(edge_image)
 
        original_np = np.array(original_image)
        height, width = original_np.shape[:2]
 
        contours, _ = cv2.findContours(edge_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # boxes = [cv2.boundingRect(contour) for contour in contours]
 
        bounding_boxes = []
 
        border_threshold_x = 50
        border_threshold_y = 150
 
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
 
            if (x < border_threshold_x or y < border_threshold_y or
                x + w > width - border_threshold_x or y + h > height - border_threshold_y):
                continue  # 경계 근처면 무시
 
            bounding_boxes.append((x, y, w, h))
 
 
        if bounding_boxes:
            x_min = min(x for x, y, w, h in bounding_boxes)
            y_min = min(y for x, y, w, h in bounding_boxes)
            x_max = max(x + w for x, y, w, h in bounding_boxes)
            y_max = max(y + h for x, y, w, h in bounding_boxes)
 
            cv2.rectangle(original_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
 
            final_box = (x_min, y_min, x_max - x_min, y_max - y_min)
 
        return final_box

