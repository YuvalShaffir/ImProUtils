
![ImgProUtils](https://github.com/YuvalShaffir/ImProUtils/assets/34415892/592d6f76-344d-4074-b84e-511362b2d15e)

The most impactful course I've undertaken during my university journey is undoubtedly the Image Processing course. 
It wasn't the ease of the exams that made it stand out; in fact, it was one of the most challenging courses. Its significance lies in its inherent interest to me and the profound impact it had on redirecting my career path.

Motivated by my passion for the subject, I crafted this utility library to serve as a valuable resource for my upcoming projects in the realm of Image Processing. 
Beyond aiding in future endeavors, the library serves as a tool for delving deep into both the concepts I've already learned and those I aspire to explore further.


# ImProUtils.image.image
Description
ImProUtils.image.image is a module containing a set of useful functions for simple image importation. It provides methods to import images from files, folders, URLs, and even create images from 1D arrays.

Accepted File Types
The module supports importing images with the following file formats:

BLP, BMP, BUFR, CUR, DCX, DDS, DIB, EPS, FITS, FLI, FTEX, GBR, GIF, GRIB,
HDF5, ICNS, ICO, IM, IMT, IPTC, JPEG, JPEG2000, MCIDAS, MPEG, MSP, PCD, PCX,
PIXAR, PNG, PPM, PSD, QOI, SGI, SPIDER, SUN, TGA, TIFF, WEBP, WMF, XBM, XPM, XVTHUMB

Accepted Modes
The supported image modes include:

1: 1-bit pixels, black and white, stored with one pixel per byte
L: 8-bit pixels, grayscale
P: 8-bit pixels, mapped to any other mode using a color palette
RGB: 3x8-bit pixels, true color
RGBA: 4x8-bit pixels, true color with transparency mask
CMYK: 4x8-bit pixels, color separation
YCbCr: 3x8-bit pixels, color video format (JPEG)
LAB: 3x8-bit pixels, the Lab color space
HSV: 3x8-bit pixels, Hue, Saturation, Value color space
I: 32-bit signed integer pixels
F: 32-bit floating point pixels

Functions:
- image_from_file
- images_from_folder
- img_from_1d_array
- img_from_url

These functions offer convenient ways to import images from various sources and manipulate them in different modes. Please refer to the function descriptions for details on their usage.
