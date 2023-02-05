import cv2

# imread

img = cv2.imread('image.png') # reading the image
while True:
  cv2.imshow("window", img) # show the image in the window
  if cv2.waitKey(0) == 27: # if we press the escape key(ASCII -> 27) then we break the loop
    break

cv2.imwrite('copyImg.png', img); # save/write the file:  first argument is image name and second is image
cv2.destroyAllWindows() # Quit or destroy all windows