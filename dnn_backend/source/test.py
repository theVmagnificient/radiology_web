import pydicom

ds = pydicom.dcmread('/data/DICOM_/tmp/+AGFA000000012999_AGFA000000012967/1001B997')

print(ds.pixel_array)