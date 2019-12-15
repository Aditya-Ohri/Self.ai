import cv2
import time
import boto3


ACCESS_ID = "XXXX"
ACCESS_KEY = "XXXXX"

# Create SDK clients for rekognition
rekog_client = boto3.client("rekognition", region_name='us-east-2',
                                 aws_access_key_id=ACCESS_ID,
                                 aws_secret_access_key=ACCESS_KEY)


def capture_image():
    for i in range(3):
        camera = cv2.VideoCapture(0)
        return_value, image = camera.read()
        cv2.imwrite('C:/Users/MAIN/PycharmProjects/Self.ai/tmp/imgopencv'+str(i)+'.png', image)
    camera.release()
    cv2.destroyAllWindows()


def bytes_to_img(imageBytes):
    with open(imageBytes, "rb") as image:
        f = image.read()
        byte_array = bytearray(f)
    return byte_array


def analyze_image(img_path):
    im = bytes_to_img(img_path)
    response = rekog_client.detect_faces(
        Image={'Bytes': im },
        Attributes=['ALL']
    )
    emotions_dict = {}
    response = response['FaceDetails'][0]['Emotions']
    for emotion in response:
        emotions_dict.update({emotion['Type']: emotion['Confidence']})
    print(emotions_dict)
    happiness = int(emotions_dict['HAPPY'] + 1)
    calm = int(emotions_dict['CALM'] + 1)
    sadness = int(emotions_dict['SAD'] + 1)
    anger = int(emotions_dict['ANGRY'] + 1)
    fear = int(emotions_dict['FEAR'] + 1)
    emotions_summary = [happiness, calm, sadness, anger, fear]
    return emotions_summary


#print(analyze_image())

