import patoolib
import os
import pydicom
import shutil
from multiprocessing import Pool


# Class for conversion and files extraction
class Preprocess:
    def __init__(self, PathToArchive, PathToConverted, numWorkers=3):
        self.a_path = PathToArchive
        self.e_path = PathToArchive + "_extracted"
        self.c_path = PathToConverted

        try:
            shutil.rmtree(self.c_path)
        except Exception as e:
            print(str(e))

        try:
            shutil.rmtree(self.e_path)
        except Exception as e:
            print(str(e))

        try:
            os.makedirs(self.c_path)
        except:
            pass

        self.n_workers = numWorkers

    def _extract(self):
        try:
            if os.path.isdir(self.e_path):
                os.mkdir(self.e_path)
            patoolib.extract_archive(self.a_path, outdir=self.e_path)
        except Exception as e:
            print(f"Failed: {str(e)}")


    def _copy_files(self, pathToStudy):
        ds = pydicom.dcmread(os.path.join(pathToStudy, os.listdir(pathToStudy)[0]))
        dirName = os.path.join(self.c_path, ds.AccessionNumber + '_' + ds.StudyID)

        try:
            os.makedirs(dirName)
        except Exception as e:
            if os.path.exists(dirName):
                print(str(e))
                return
            else:
                print(str(e))

        print(f"Copying {pathToStudy}...")
        for dicom in os.listdir(pathToStudy):
            shutil.copy(os.path.join(pathToStudy, dicom), dirName)

    def _convert2format(self):
        pathsToDcm = []
        for r, d, f in os.walk(self.e_path):
            for file in f:
                if r in pathsToDcm:
                    break

                flag = True
                try:
                    ds = pydicom.dcmread(os.path.join(r, file))
                except:
                    flag = False
                if flag and len(os.listdir(r)) > 100:
                    pathsToDcm.append(r)
        print("Lenght of dataset: " + str(len(pathsToDcm)))

        p = Pool(self.n_workers)

        p.map(self._copy_files, pathsToDcm)
        #for dir in pathsToDcm:
        #    self._copy_files(dir)

    def start(self):
        self._extract()
        self._convert2format()
        print(f"Converted {len(os.listdir(self.c_path))} studies")
        print(f"Removing extracted files...")
        shutil.rmtree(self.e_path)



#prep = Preprocess('/data/DICOM_/5_test.zip', '/data/DICOM_/converted_test')
#prep.start()












