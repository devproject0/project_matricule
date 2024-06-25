from django.http import JsonResponse
import cv2
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import statistics
from django.views.decorators.csrf import csrf_exempt
from detectron2 import model_zoo
# Load the OCR models (license plate detector and character OCR)
# Current file directory
current_directory = os.path.dirname(__file__)
license_plate_detector_path = os.path.join(current_directory, 'license_plate_detector.pt')
import os

# Locate the current file's directory
current_directory = os.path.dirname(__file__)

# Define the path to the 'model_final.pth' file
#model_file_path = os.path.join(current_directory, 'weights', 'plate_ocr', 'model_final.pth')
model_file_path = os.path.join(current_directory, 'weights', 'ocr_lp', 'model_final.pth')
print("Model file path:", model_file_path)

# Initialize YOLO model with the correct file path
license_plate_detector = YOLO(license_plate_detector_path)
class PlateOCR:
    def __init__(self):
        #create a predictor
        self._cfg = get_cfg()
        self._predictor = self._makePredictor()
        #self._characters = ["0","1","2","3","4","5","6","7","8","9", "a","b","h","w","d","p","waw","j","m","m"]
        self._characters = ["0","1","2","3","4","5","6","7","8","9","a","b","h","w","d","ch","waw","t"]
        self._class = MetadataCatalog.get("characters").set(thing_classes=self._characters)
    
    def _makePredictor(self):
        self._cfg.MODEL.DEVICE = "cpu"
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.SOLVER.IMS_PER_BATCH = 2
        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
        self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18
        self._cfg.MODEL.WEIGHTS = model_file_path
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        return DefaultPredictor(self._cfg)
    
    def predict(self, image):
        return self._predictor(image)
    
    def characterBoxes(self, output):
        boxes = output['instances'].pred_boxes.tensor.cpu().numpy().tolist() 
        scores = output['instances'].scores.numpy().tolist()
        classes = output['instances'].pred_classes.to('cpu').tolist()
        characters = {}
        if len(scores) > 0:
            characters = {i: {"character": self._characters[classes[i]], "score": scores[i], "boxes": boxes[i]} for i in range(0, len(scores))}
        return characters

    def postProcess(self, image, output):
        if len(output.keys())<=0:
            plate_ocr_string = {'plate':image[:-4],'plate_string':''}
        else :
            y_mins = []
            for character in list(output.items()):
                y_mins.append(character[1]['boxes'][1])
            median_y_mins = statistics.median(y_mins)
            top_characters =  dict()
            bottom_characters = dict()
            for key,value in output.items():
                if (value['boxes'][3] <= median_y_mins ):
                    top_characters[key] = value
                else :
                    bottom_characters[key] = value
            sorted_top_characters = sorted(top_characters.items(), key=lambda e: e[1]['boxes'][0])
            sorted_bottom_characters = sorted(bottom_characters.items(), key=lambda e: e[1]['boxes'][0])
            top_plate_ocr = [item[1]['character'] for item in sorted_top_characters]
            bottom_plate_ocr = [item[1]['character'] for item in sorted_bottom_characters]
            plate_ocr = bottom_plate_ocr+top_plate_ocr
            plate_ocr = "".join(str(x) for x in plate_ocr)
            plate_ocr_string = {'plate':image[:-4],'plate_string':plate_ocr}
        return(plate_ocr_string)

@csrf_exempt
def process_license_plate(request):
    # Extract the 'path' parameter from the request's GET parameters
    image_path = request.GET.get('path', None)
    
    if image_path is None:
        # If 'path' parameter is not provided, return an error response
        error_response = {'error': 'Image path not provided in URL parameters.'}
        return JsonResponse(error_response, status=400)
    
    try:
        # Load the image
        frame = cv2.imread(image_path)
        
        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        
        # Initialize OCR
        plate_ocr = PlateOCR()
        
        # Process each license plate and store results
        plate_results = []
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            # Perform OCR
            output = plate_ocr.predict(license_plate_crop)
            characters = plate_ocr.characterBoxes(output)
            plate_ocr_string = plate_ocr.postProcess(image_path, characters)
            
            # Add the result to the list
            #plate_results.append({
                #'Matricule': plate_ocr_string['plate_string']
            #})
        #return JsonResponse({'plates': plate_results}, safe=False)
        # Return the JSON response with the results
        return JsonResponse({'Matricule': plate_ocr_string['plate_string']}, safe=False)
    
    except Exception as e:
        # Handle any exceptions that may occur during processing
        error_response = {'error': f'An error occurred: {str(e)}'}
        return JsonResponse(error_response, status=500)
