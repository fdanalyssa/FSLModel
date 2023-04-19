import cv2
import os


def main():
  folder = "test"

  newResizedFolder = "done/"

  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))

  #  if img is not None:
      # Use Flip code 0 to flip vertically
     # img = cv2.flip(img, 1)
    newImage = cv2.resize(img, (400, 400), fx = 0.75, fy = 0.75, interpolation=cv2.INTER_AREA)
    newImgPath = newResizedFolder + filename
    cv2.imwrite(newImgPath, newImage)

if __name__ == "__main__":
  main()