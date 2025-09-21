import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

# --- CONFIGURATION ---
ANSWER_KEY = {0: 1, 1: 3, 2: 0, 3: 2, 4: 1, 5: 0, 6: 3, 7: 1, 8: 2, 9: 0,
              10: 1, 11: 3, 12: 0, 13: 2, 14: 1, 15: 0, 16: 3, 17: 1, 18: 2, 19: 0,
              20: 1, 21: 3, 22: 0, 23: 2, 24: 1, 25: 0, 26: 3, 27: 1, 28: 2, 29: 0,
              30: 1, 31: 3, 32: 0, 33: 2, 34: 1, 35: 0, 36: 3, 37: 1, 38: 2, 39: 0,
              40: 1, 41: 3, 42: 0, 43: 2, 44: 1, 45: 0, 46: 3, 47: 1, 48: 2, 49: 0,
              50: 1, 51: 3, 52: 0, 53: 2, 54: 1, 55: 0, 56: 3, 57: 1, 58: 2, 59: 0,
              60: 1, 61: 3, 62: 0, 63: 2, 64: 1, 65: 0, 66: 3, 67: 1, 68: 2, 69: 0,
              70: 1, 71: 3, 72: 0, 73: 2, 74: 1, 75: 0, 76: 3, 77: 1, 78: 2, 79: 0,
              80: 1, 81: 3, 82: 0, 83: 2, 84: 1, 85: 0, 86: 3, 87: 1, 88: 2, 89: 0,
              90: 1, 91: 3, 92: 0, 93: 2, 94: 1, 95: 0, 96: 3, 97: 1, 98: 2, 99: 0}
              
SUBJECT_MAP = {
    "Subject 1": range(0, 20),
    "Subject 2": range(20, 40),
    "Subject 3": range(40, 60),
    "Subject 4": range(60, 80),
    "Subject 5": range(80, 100)
}

def process_omr_sheet(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    doc_cnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
    
    if doc_cnt is None:
        raise ValueError("Could not find the OMR sheet's 4 corners. Please use a clearer image.")

    paper = four_point_transform(image, doc_cnt.reshape(4, 2))
    warped = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    question_cnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            question_cnts.append(c)

    # Safety check: If no bubbles are found, stop and raise an error.
    if not question_cnts:
        raise ValueError("No bubbles were found. Please use a clearer image with better lighting.")
    
    question_cnts = contours.sort_contours(question_cnts, method="top-to-bottom")[0]

    correct = 0
    subject_scores = {subject: 0 for subject in SUBJECT_MAP}
    
    # Assuming 4 options per question
    for (q, i) in enumerate(np.arange(0, len(question_cnts), 4)):
        cnts = contours.sort_contours(question_cnts[i:i + 4], method="left-to-right")[0]
        
        # Safety check: Only process if a full set of options is found for a question.
        if len(cnts) == 4:
            bubbled = None
            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)
            
            color = (0, 0, 255)
            k = ANSWER_KEY.get(q)

            if k is not None and bubbled is not None and k == bubbled[1]:
                color = (0, 255, 0)
                correct += 1
                for subject, q_range in SUBJECT_MAP.items():
                    if q in q_range:
                        subject_scores[subject] += 1
            
            if k is not None:
                 cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    score = (correct / 100.0) * 100
    cv2.putText(paper, f"Score: {score:.2f}% ({correct}/100)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    result_image_path = image_path.replace('uploads', 'processed')
    cv2.imwrite(result_image_path, paper)
    
    return score, subject_scores, result_image_path