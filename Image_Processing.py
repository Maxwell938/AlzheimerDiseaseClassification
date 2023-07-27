#MAKE SURE WHEN YOU RUN THAT ONLY ONE OF THE "BLOCKS" OF CODE ARE UNCOMMENTED


#BLOCK 1

"""import cv2
def threshold_image(image, threshold):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    T, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY) #The variable threshold is the lower boundary, and everything above is set to the value 255. All other values are set to 0.

    #T is trash variable that has no use, it holds useless values

    return thresholded

# Load the grayscale image
image = cv2.imread('test.png', 0)  # 0 indicates grayscale mode

# Set the threshold value (adjust as needed)
threshold_value = 250

# Apply thresholding
thresholded_image = threshold_image(image, threshold_value)

# Display the thresholded image
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#END OF BLOCK 1





#BLOCK 2

"""import cv2

# Load the grayscale image
image = cv2.imread('test.png', 0)  # 0 indicates grayscale mode

# Apply thresholding
thresholded_image = cv2.inRange(image, 100, 150) #100 is lower bound, and 150 is upper bound. All values between 100 and 150 are set to white, and the rest are set to black.

# Display the thresholded image
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#END OF BLOCK 2




#NOW YOU KNOW HOW THRESHOLDING WORKS AND HOW TO RUN IT, NOW YOU JUST NEED TO KNOW WHAT VALUES TO THRESHOLD AT




#BLOCK 3

import cv2
import numpy as np
import math

gray_value = 0.5

def threshold_image(image, threshold):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (3*2+1, 3*2+1), 0) #THIS PERFORMS GAUSSIAN BLUR SUCH THAT THE NOISE IS SLIGHTLY REMOVED

    # Apply thresholding
    thresholded_image = cv2.inRange(blurred_image, int(threshold-26), int(threshold+26))

    return thresholded_image

def get_pixel_value(event, x, y, flags, param):
    global gray_value

    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the grayscale value of the clicked pixel
        gray_value = gray_image[y, x]
        print("Grayscale Value:", gray_value)

        # Update the threshold by calling the callback function
        on_threshold_change(0)  # Pass any value as the argument

def get_amt_pixel(value, image):
    amt = 0
    i = 0
    while (i < image.shape[0]):
        j = 0
        while (j < image.shape[1]):
            if (image[i][j] == value):
                amt += 1
            j += 1
        i += 1

    return amt


def exists(matrix, position):
  if (position[0] >= 0 and position[0] < matrix.shape[0]):
    if (position[1] >= 0 and position[1] < matrix.shape[1]):
      return True

  return False

def convert(matrix, x, y, step):

    if (step >= 900):
        return matrix
    
    if (exists(matrix, (x,y))):
        if (matrix[x][y] == 255): # its white
            matrix[x][y] = 0
            matrix = convert(matrix, x-1, y-1, step+1)
            matrix = convert(matrix, x, y-1, step+1)
            matrix = convert(matrix, x+1, y-1, step+1)
            matrix = convert(matrix, x-1, y, step+1)
            matrix = convert(matrix, x+1, y, step+1)
            matrix = convert(matrix, x-1, y+1, step+1)
            matrix = convert(matrix, x, y+1, step+1)
            matrix = convert(matrix, x+1, y+1, step+1)
            
    return matrix

def convert2(matrix, x, y, step):

    if (step >= 900):
        return matrix
    
    if (exists(matrix, (x,y))):
        if (matrix[x][y] == 0): # its white
            matrix[x][y] = 255
            matrix = convert2(matrix, x-1, y-1, step+1)
            matrix = convert2(matrix, x, y-1, step+1)
            matrix = convert2(matrix, x+1, y-1, step+1)
            matrix = convert2(matrix, x-1, y, step+1)
            matrix = convert2(matrix, x+1, y, step+1)
            matrix = convert2(matrix, x-1, y+1, step+1)
            matrix = convert2(matrix, x, y+1, step+1)
            matrix = convert2(matrix, x+1, y+1, step+1)
            
    return matrix

        
def borderdetect(matrix):
    # go through each line, 
    
    matrix = convert(matrix, 0, 0, 0)

    matrix = convert(matrix, 0, math.floor(matrix.shape[1]/2), 0)

    matrix = convert(matrix, matrix.shape[0]-1, math.floor(matrix.shape[1]/2), 0)

    matrix = convert(matrix, 0, matrix.shape[1]-1, 0)

    matrix = convert(matrix, matrix.shape[0]-1, matrix.shape[1]-1, 0)

    matrix = convert(matrix, matrix.shape[0]-1, math.floor(4*matrix.shape[1]/10), 0)


    matrix = convert2(matrix, 0, 0, 0)

    matrix = convert2(matrix, 0, math.floor(matrix.shape[1]/2), 0)

    matrix = convert2(matrix, matrix.shape[0]-1, math.floor(matrix.shape[1]/2), 0)

    matrix = convert2(matrix, 0, matrix.shape[1]-1, 0)

    matrix = convert2(matrix, matrix.shape[0]-1, matrix.shape[1]-1, 0)

    matrix = convert2(matrix, matrix.shape[0]-1, math.floor(4*matrix.shape[1]/10), 0)

    return matrix
    

def cutoffleft(matrix, istart):
    newmatrix = np.zeros([matrix.shape[0],matrix.shape[1]-istart]) # initialize it to 0s
    
    # now start from i and transfer
    
    i = 0
    while (i < newmatrix.shape[0]):
        j = 0
        while (j < newmatrix.shape[1]):
            newmatrix[i][j] = matrix[i][j+istart]
            j += 1
        i += 1
    
    return newmatrix

def cutoffright(matrix, iend):
    newmatrix = np.zeros([matrix.shape[0],iend]) # initialize it to 0s
    
    # now start from i and transfer
    
    i = 0
    while (i < newmatrix.shape[0]):
        j = 0
        while (j < newmatrix.shape[1]):
            newmatrix[i][j] = matrix[i][j]
            j += 1
        i += 1
    
    return newmatrix


def cropl(matrix):

    i = 0
    while (i < matrix.shape[1]):
        j = 0
        while (j < matrix.shape[0]):
            if (matrix[j][i] > 200):
                #matrix[j][i] = 0.5
                pass
            else:
                #print("found notblack at",j,i)
                #return matrix
                
                return cutoffleft(matrix,i-1)
                
#                 return (rec, i, j)
    
            j += 1
        i += 1
        

def cropr(matrix):

    i = matrix.shape[1]-1
    while (i >= 0):
        j = 0
        while (j < matrix.shape[0]):
            if (matrix[j][i] > 200):
                pass
                #matrix[j][i] = 0.5
            else:
                return cutoffright(matrix,i+2) # so +1 makes sure that the current line isnt cut, other +1 adds a buffer

            j += 1
        i -= 1
        
    
    return matrix

def on_threshold_change(value, image, strval):
    # Adjust the threshold using the grayscale value and the threshold error
    threshold_value = value

    # Threshold the image using the adjusted threshold value

    thresholded_image = cropl(cropr(threshold_image(image, threshold_value)))
    # print(thresholded_image)
    cv2.imwrite(strval+".png", thresholded_image)

    # blackpx = get_amt_pixel(0, thresholded_image)
    
    # thresholded_image = cutinhalfleft(thresholded_image)

    # thresholded_image = cutoffleft(thresholded_image, thresholded_image.shape[1])

    #thresholded_image_compare = borderdetect(cutinhalfleft(cropl(cropr(threshold_image(image_compare, threshold_value)))))

    # thresholded_image_compare = cutoffleft(thresholded_image_compare, thresholded_image_compare.shape[1])


    # Display the thresholded image
    # cv2.imshow('Thresholded Image', thresholded_image)
    #cv2.imshow('Thresholded Image compare', thresholded_image_compare)

    #print("Amt of whites mild demented: ", get_amt_pixel(255, thresholded_image))
    #print("Amt of whites non-demented: ", get_amt_pixel(255, thresholded_image_compare))

    # return (get_amt_pixel(255, thresholded_image), get_amt_pixel(0, thresholded_image), thresholded_image.shape[0], thresholded_image.shape[1], blackpx)
    return (0, 0, 0, 0, 0)

def on_threshold_change_display(value, image):
    # Adjust the threshold using the grayscale value and the threshold error
    threshold_value = value

    # Threshold the image using the adjusted threshold value

    # thresholded_image = borderdetect(cropl(cropr(threshold_image(image, threshold_value))))

    # blackpx = get_amt_pixel(0, thresholded_image)

    thresholded_image = borderdetect(cropl(cropr(threshold_image(image, threshold_value))))
        # thresholded_image = cutinhalfleft(thresholded_image)

    # thresholded_image = cutoffleft(thresholded_image, thresholded_image.shape[1])

    #thresholded_image_compare = borderdetect(cutinhalfleft(cropl(cropr(threshold_image(image_compare, threshold_value)))))

    # thresholded_image_compare = cutoffleft(thresholded_image_compare, thresholded_image_compare.shape[1])


    # Display the thresholded image
    cv2.imshow('Thresholded Image', thresholded_image)
    #cv2.imshow('Thresholded Image compare', thresholded_image_compare)

    #print("Amt of whites mild demented: ", get_amt_pixel(255, thresholded_image))
    #print("Amt of whites non-demented: ", get_amt_pixel(255, thresholded_image_compare))

def thresholdit(gray_value, image1):
    # Adjust the threshold using the grayscale value and the threshold error
    threshold_value = gray_value

    # Threshold the image using the adjusted threshold value

    thresholded_image = borderdetect(cutinhalfleft(cropl(cropr(threshold_image(image1, threshold_value)))))

    # thresholded_image = cutoffleft(thresholded_image, thresholded_image.shape[1])

    # thresholded_image_compare = borderdetect(cutinhalfleft(cropl(cropr(threshold_image(image_compare, threshold_value)))))

    # thresholded_image_compare = cutoffleft(thresholded_image_compare, thresholded_image_compare.shape[1])


    # Display the thresholded image
    cv2.imshow('Thresholded Image', thresholded_image)
    # cv2.imshow('Thresholded Image compare', thresholded_image_compare)

    print("Amt of whites mild demented: ", get_amt_pixel(255, thresholded_image))
    # print("Amt of whites non-demented: ", get_amt_pixel(255, thresholded_image_compare))


def cropl(matrix):

    i = 0
    while (i < matrix.shape[1]):
        j = 0
        while (j < matrix.shape[0]):
            if (matrix[j][i] > 200):
                #matrix[j][i] = 0.5
                pass
            else:
                #print("found notblack at",j,i)
                #return matrix
                
                return cutoffleft(matrix,i-1)
                
#                 return (rec, i, j)
    
            j += 1
        i += 1
        

def cropr(matrix):

    i = matrix.shape[1]-1
    while (i >= 0):
        j = 0
        while (j < matrix.shape[0]):
            if (matrix[j][i] > 200):
                pass
                #matrix[j][i] = 0.5
            else:
                return cutoffright(matrix,i+2) # so +1 makes sure that the current line isnt cut, other +1 adds a buffer

            j += 1
        i -= 1
        
    
    return matrix
        
        
def cutinhalfleft(matrix):

    newmatrix = np.zeros([matrix.shape[0], math.floor(matrix.shape[1]/2)]) # initialize it to 0s

    i = 0
    while (i < matrix.shape[0]):
        j = 0
        while (j < math.floor(matrix.shape[1]/2)):
            newmatrix[i][j] = matrix[i][j]
            j += 1
        i += 1

    return newmatrix

def borderdetectleft(matrix):

    newmatrix = np.zeros([matrix.shape[0], matrix.shape[1]]) # initialize it to 0s

    i = 0
    while (i < matrix.shape[0]):
        j = 0
        found = False
        while (j < matrix.shape[1]):
            tt = matrix[i][j]
            print(tt)
            if (tt == 255 and not found):
                newmatrix[i][j] = 0
            elif (found):
                newmatrix[i][j] = matrix[i][j]
            else:
                found = True
            j += 1
        i += 1
    
    return newmatrix

def borderdetectall(matrix):
    return borderdetectleft(matrix)

def getimagewhitemoderate(person, series, slice):
    # Load the image
    path = "Data/ModerateDementia/OAS1_"+getgoodnumber(series+1)+"_MR1_mpr-"+str(person)+"_"+str(slice)+".jpg"
    image1 = cv2.imread(path)
    # image_compare = cv2.imread('Data/NonDemented/OAS1_0001_MR1_mpr-1_160.jpg')


    # Convert the image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image_compare = cv2.cvtColor(image_compare, cv2.COLOR_BGR2GRAY)


    try:
        image1 = cv2.imread(path)
        ttrs = on_threshold_change(26, image1, "data_processed/moderate/series"+str(series)+"num"+str(person)+"slice"+str(slice))
        return ttrs
    except:
        return (0,0,0,0,0)

def getgoodnumber(n):
    if (n < 10):
        return "000"+str(n)
    if (n < 100):
        return "00"+str(n)
    if (n < 1000):
        return "0"+str(n)
    str(n)

def getimagewhitenondemented(person, series, slice):
    # Load the image
    #                        OAS1_0001_MR1_mpr-1_100
    #                        OAS1_0002_MR1_mpr-1_100
    path = "Data/NonDemented/OAS1_"+getgoodnumber(series+1)+"_MR1_mpr-"+str(person)+"_"+str(slice)+".jpg"
    
    ttrs = 0

    try:
        image1 = cv2.imread(path)
        ttrs = on_threshold_change(26, image1,  "data_processed/non/series"+str(series)+"num"+str(person)+"slice"+str(slice))
        return ttrs
    except:
        return (0,0,0,0,0)

def getimagewhitemild(person, series, slice):
    # Load the image
    #            MildDementia    2      OAS1_0028_MR1_mpr-1_100
    path = "Data/MildDementia/OAS1_"+getgoodnumber(series+1)+"_MR1_mpr-"+str(person)+"_"+str(slice)+".jpg"
    
    ttrs = 0

    try:
        image1 = cv2.imread(path)
        ttrs = on_threshold_change(26, image1, "data_processed/mild/series"+str(series)+"num"+str(person)+"slice"+str(slice))
        return ttrs
    except:
        return (0,0,0,0,0)

def getimagewhiteverymild(person, series, slice):
    # Load the image
    #                        OAS1_0003_MR1_mpr-1_100
    path = "Data/VerymildDementia/OAS1_"+getgoodnumber(series+1)+"_MR1_mpr-"+str(person)+"_"+str(slice)+".jpg"
    ttrs = 0

    try:
        image1 = cv2.imread(path)
        ttrs = on_threshold_change(26, image1, "data_processed/verymild/series"+str(series)+"num"+str(person)+"slice"+str(slice))
        return ttrs
    except:
        return (0,0,0,0,0)

import csv

def writedata(category, patient, whitevalue, blackvalue, blackvaluefull, height, width):
    dataarr = []

    with open('data.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            dataarr.append(row)

    #print(dataarr)

    dataarr.append([category, patient, whitevalue, blackvalue, blackvaluefull, height, width])

    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataarr)

def measure(category1, person, series):

    print("started")

    ctr = 100
    sum1 = 0
    sum2 = 0
    sum3 = 0
    hmax = 0
    wmax = 0

    while (ctr < 161):
        if (category1 == "Moderate"):
            s1,s2,h,w,s3 = getimagewhitemoderate(person, series, ctr)
        elif (category1 == "Mild"):
            s1,s2,h,w,s3 = getimagewhitemild(person, series, ctr)
        elif (category1 == "Non"):
            s1,s2,h,w,s3 = getimagewhitenondemented(person, series, ctr)
        elif (category1 == "Verymild"):
            s1,s2,h,w,s3 = getimagewhiteverymild(person, series, ctr)

        if (h > hmax):
            hmax = h
        if (w > wmax):
            wmax = w

        sum1 += s1
        sum2 += s2
        sum3 += s3

        #print("Layer "+str(ctr)+" had "+str(s1)+" whites")
        ctr += 1

    person = person + series*4

    if (sum1 != 0):
        writedata(category1, person, sum1, sum2, sum3, hmax, wmax)
    print("Completed "+str(category1)+" patient "+str(person))#+" had "+str(sum1)+" white volume, "+str(sum2)+" black volume, "+str(sum3)+" full black volume, "+str(hmax)+" height, "+str(wmax)+" width")

def doseries(catagory2, series1):
    air = 1
    while (air < 5):
        measure(catagory2, air, series1)
        air += 1


# getimagewhitemoderate(1, 307, 120)

# stoppped at series 222


# go through and process all of the images in a certain category
integra = 0

while (integra < 400):
    doseries("Non", integra)
    integra += 1



# ctr = 100
# sum1 = 0
# sum2 = 0
# while (ctr < 161):
#     s1 = getimagewhitemoderate(1,ctr)
#     sum1 += s1
#     s2 = getimagewhitenondemented(1,ctr)
#     sum2 += s2
#     print("Layer "+str(ctr)+" had "+str(s1)+" whites")
#     print("Layer "+str(ctr)+" had "+str(s2)+" whites")
#     ctr += 1

# print("End: Moderate had"+str(sum1)+" white volume")
# print("Layer Moderate had"+str(sum2)+" white volume")

#TODO: DO YOUR CONSTRAST AND OTHER IMAGE MANIPULATION HERE TO MAKE IT EASIER TO THRESHOLD

# Create a window for displaying the image
# cv2.namedWindow('Image')

# Create a trackbar for adjusting the threshold error
# cv2.createTrackbar('Threshold Error', 'Image', 26, 255, on_threshold_change)
# cv2.createTrackbar('Blur', 'Image', 3, 50, on_threshold_change)


# image1 = cv2.imread("Data/NonDemented/OAS1_0001_MR1_mpr-1_100.jpg")
# on_threshold_change_display(26, image1)



#image = cv2.imread('Data/ModerateDementia/OAS1_0308_MR1_mpr-1_160.jpg')

# Bind the mouse callback function
# cv2.setMouseCallback('Image', get_pixel_value)

# # Display the original image
# cv2.imshow('Image', image)


# Wait for the user to press any key to exit
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#END OF BLOCK 3


#USE BLOCK 1 AND 2 TO LEARN THRESHOLDING, THEN USE BLOCK 3 TO MAKE IT EASIER TO EXECUTE. THEORETICALLY, YOU WOULD GO THROUGH ALL THE IMAGES
#AND AND USE THIS TOOL TO SEGMENT THE PROPER PART OF THE IMAGE. AFTERWARDS, SAVE THE MASK OF THE IMAGE WITH A SIMIILAR FILE NAME SO THAT WE
#CAN POSSIBLY TRAIN A SEGMENTATION MODEL ON IT LATER. TO SEGMENT THE IMAGES EASIER, YOU SHOULD PLAY WITH THE CONTRAST OF THE IMAGES USING
#THE TECHNIQUES WE LEARNED IN CLASS. LET ME KNOW IF YOU HAVE ANY QUESTIONS :D

#AFTER YOU GET CONFIDENT YOU FOUND THE PERFECT NUMBERS, JUST RUN CODE TO THRESHOLD ALL THE IMAGES IN THE FOLDER. LET ME KNOW WHEN YOU REACH
#THIS STEP