import matplotlib.pyplot as plt
from pydicom.data import get_testdata_files
import pydicom as dicom
import os

def load_lidc_scan(filepath, resize=None, print_details=False):
	slices = [dicom.read_file(filepath + '/' + s) for s in os.listdir(filepath)]

	if not is_scan_processable(slices):
		print("Not processable: ", filepath)
		return None

	if print_details:
		print(slices[0])
		print(len(slices))
		print(filepath)
		return

	slices.sort(key=lambda x: x.ImagePositionPatient[2])

	image = get_slices_HU(slices)
	origShape = image.shape
	if resize:
		image = get_resized_image(image, resize)

	spacing = np.array(slices[0].PixelSpacing + [slices[0].SliceThickness], dtype=np.float32)
	origin = np.array(slices[0].ImagePositionPatient)

	return image, spacing, origin, origShape 

im, sp, orig, origS = load_lidc_scan('/home/suriknik/Projects/ctc/test_dicom/RLADD01000000088399_RLSDD01000000088273/test')

# print(__doc__)

# filename = get_testdata_files('test1.dcm')[0]
# dataset = pydicom.dcmread(filename)

# # Normal mode:
# print()
# print("Filename.........:", filename)
# print("Storage type.....:", dataset.SOPClassUID)
# print()

# pat_name = dataset.PatientName
# display_name = pat_name.family_name + ", " + pat_name.given_name
# print("Patient's name...:", display_name)
# print("Patient id.......:", dataset.PatientID)
# print("Modality.........:", dataset.Modality)
# print("Study Date.......:", dataset.StudyDate)

# if 'PixelData' in dataset:
#     rows = int(dataset.Rows)
#     cols = int(dataset.Columns)
#     print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
#         rows=rows, cols=cols, size=len(dataset.PixelData)))
#     if 'PixelSpacing' in dataset:
#         print("Pixel spacing....:", dataset.PixelSpacing)

# # use .get() if not sure the item exists, and want a default value if missing
# print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

# # plot the image using matplotlib
# print(dataset.pixel_array)
# plt.imshow(dataset.PixelData, cmap=plt.cm.bone)
# plt.show()
