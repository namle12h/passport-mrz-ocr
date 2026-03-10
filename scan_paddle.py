import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from datetime import datetime
import cv2
import numpy as np
import os
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init OCR
ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
)

os.makedirs("debug", exist_ok=True)


# rotate image nếu bị nghiêng
def deskew(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def format_date(date_str, is_expiry=False):

    if len(date_str) != 6:
        return None

    yy = int(date_str[0:2])
    mm = int(date_str[2:4])
    dd = int(date_str[4:6])

    if mm < 1 or mm > 12 or dd < 1 or dd > 31:
        return None

    current_year = datetime.now().year % 100

    if is_expiry:
        year = 2000 + yy
    else:
        if yy > current_year:
            year = 1900 + yy
        else:
            year = 2000 + yy

    return f"{year}-{mm:02d}-{dd:02d}"


def parse_mrz(lines):

    if len(lines) < 2:
        return None

    l1 = lines[0]
    l2 = lines[1]

    try:

        passport_number = l2[0:9].replace("<", "")
        nationality = l2[10:13]

        dob_raw = l2[13:19]
        gender = l2[20]
        expiry_raw = l2[21:27]

        dob = format_date(dob_raw)
        expiry = format_date(expiry_raw, True)

        name_raw = l1[5:]
        names = name_raw.split("<<")

        last_name = names[0].replace("<", " ")
        first_name = ""

        if len(names) > 1:
            first_name = names[1].replace("<", " ")

        name = (last_name + " " + first_name).strip()

        return {
            "name": name,
            "passportNumber": passport_number,
            "nationality": nationality,
            "dob": dob,
            "gender": gender,
            "expiry": expiry
        }

    except:
        return None


@app.post("/api/scan-passport")
async def scan_passport(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = deskew(img)

    h, w = img.shape[:2]

    # crop vùng MRZ (30% dưới)
    mrz = img[int(h * 0.6):h, 0:w]

    gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
    
    # tự động cân bằng sáng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
  

    # save debug image
    cv2.imwrite("debug/mrz_crop.jpg", mrz)
    cv2.imwrite("debug/img_goc.jpg", img)
    cv2.imwrite("debug/mrz_thresh.jpg", thresh)

    # OCR bằng Paddle
    result = ocr.ocr(thresh)

    text = ""

    for line in result:
        for word in line:
            text += word[1][0] + "\n"

    print("===== RAW OCR =====")
    print(text)

    lines = text.split("\n")

    mrz_lines = []

    # for l in lines:

    #     l = l.strip().replace(" ", "")

    #     if "<" in l and len(l) > 20:
    #         mrz_lines.append(l)
    
    mrz_lines = []

    for l in lines:

        l = l.strip().replace(" ", "")
        l = l.upper()

        # sửa lỗi OCR
        # l = l.replace("K", "<")
        l = l.replace("(", "<")
        l = l.replace("|", "<")

        # nếu có P< nhưng có ký tự rác phía trước
        if "P<" in l:
            l = l[l.index("P<"):]   # cắt bỏ ký tự trước P<

        # nếu là dòng MRZ đầu
        if l.startswith("P<"):
            l = l.ljust(44, "<")
            mrz_lines.insert(0, l)

        # dòng MRZ thứ hai
        elif len(l) > 30 and "<" in l:
            mrz_lines.append(l)

    print("MRZ lines:", mrz_lines)

    data = parse_mrz(mrz_lines)

    success = data is not None

    return {
        "success": success,
        "data": data,
        "mrz_lines": mrz_lines,
        "raw_text": text,
        "message": "Scan thành công" if success else "Không đọc được MRZ"
    }

@app.get("/")
def home():
    return {"status": "passport OCR API running"}