import pytesseract
from PIL import Image
import requests
from io import BytesIO

url = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
img = Image.open(BytesIO(requests.get(url).content))

text = pytesseract.image_to_string(img)
print(text)
