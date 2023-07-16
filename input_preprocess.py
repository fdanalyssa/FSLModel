import cv2
import os
import splitfolders

def splitdata():

  input_folder = "folder"
  output_folder = "output_folder"

  splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .2), group_prefix=None) # default values

  return None


def resized():
  LETTER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

  for i in range(len(LETTER)):
    
    folder = f"Dataset/{LETTER[i]}"

    newResizedFolder = f"Resized_224/{LETTER[i]}/"

    for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename))

    #  if img is not None:
        # Use Flip code 0 to flip vertically
      # img = cv2.flip(img, 1)

      newImage = cv2.resize(img, (224, 224), fx = 0.75, fy = 0.75, interpolation=cv2.INTER_AREA)
      newImgPath = newResizedFolder + filename
      cv2.imwrite(newImgPath, newImage)

    print(f"Done Letter {LETTER[i]}")  
  
  


def main():
  splitdata()
  

if __name__ == "__main__":
  main()
