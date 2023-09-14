# from __future__ import print_function
import six
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
# from pyradiomics_tool.radiomics import featureextractor
# from pyradiomics_tool.radiomics import firstorder, glcm, glrlm, glszm, gldm, ngtdm, imageoperations, shape
from radiomics import featureextractor
from radiomics import firstorder, glcm, glrlm, glszm, gldm, ngtdm, imageoperations, shape

import warnings
warnings.filterwarnings("ignore")

def catch_features(image, mask, params_path=None, show_mode=False):

    if show_mode:
        image = sitk.ReadImage(".\pyradiomics_tool\data\\brain1_image.nrrd")
        mask = sitk.ReadImage(".\pyradiomics_tool\data\\brain1_label.nrrd")

        # Setting for the feature calculation.
        settings = {'binWidth': 25,
                    'interpolator': sitk.sitkBSpline,
                    'resampledPixelSpacing': None}

        # If enabled, resample image (resampled image is automatically cropped.)
        interpolator = settings.get('interpolator')
        resampledPixelSpacing = settings.get('resampledPixelSpacing')
        if interpolator is not None and resampledPixelSpacing is not None:
            image, mask = imageoperations.resampleImage(image, mask, **settings)

        bb, correctedMask = imageoperations.checkMask(image, mask)
        if correctedMask is not None:
            mask = correctedMask
        image, mask = imageoperations.cropToTumorMask(image, mask, bb)

        applyLBP = True
        applyWavelet = True

        # Show the first order feature calculations
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
        firstOrderFeatures.enableAllFeatures()
        # firstOrderFeatures.enableFeatureByName('Mean', True)

        print('Will calculate the following first order features: ')
        print(list(firstOrderFeatures.enabledFeatures.keys()))

        print('Calculating first order features...')
        results = firstOrderFeatures.execute()
        print('done')

        print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)

        print('Numbers of first order features:', len(results))
        print('\n')

        # Show Shape features
        shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
        shapeFeatures.enableAllFeatures()

        print('Will calculate the following Shape features: ')
        print(list(shapeFeatures.enabledFeatures.keys()))

        print('Calculating Shape features...')
        results = shapeFeatures.execute()
        print('done')

        print('Calculated Shape features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)

        print('Numbers of Shape features:', len(results))
        print('\n')

        # Show GLCM features
        glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
        glcmFeatures.enableAllFeatures()

        print('Will calculate the following GLCM features: ')
        print(list(glcmFeatures.enabledFeatures.keys()))

        print('Calculating GLCM features...')
        results = glcmFeatures.execute()
        print('done')

        print('Calculated GLCM features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)
        print('Numbers of GLCM features:', len(results))
        print('\n')

        # Show GLRLM features
        glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
        glrlmFeatures.enableAllFeatures()

        print('Will calculate the following GLRLM features: ')
        print(list(glrlmFeatures.enabledFeatures.keys()))

        print('Calculating GLRLM features...')
        results = glrlmFeatures.execute()
        print('done')

        print('Calculated GLRLM features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)
        print('Numbers of GLRLM features:', len(results))
        print('\n')

        # Show GLSZM features
        glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)

        glszmFeatures.enableAllFeatures()
        print('Will calculate the following GLSZM features: ')
        print(list(glszmFeatures.enabledFeatures.keys()))


        print('Calculating GLSZM features...')
        results = glszmFeatures.execute()
        print('done')

        print('Calculated GLSZM features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)
        print('Numbers of GLSZM features:', len(results))
        print('\n')

        # Show GLDM features
        gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
        gldmFeatures.enableAllFeatures()


        print('Will calculate the following GLDM features: ')
        print(list(gldmFeatures.enabledFeatures.keys()))

        print('Calculating GLDM features...')
        results = gldmFeatures.execute()
        print('done')

        print('Calculated GLDM features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)
        print('Numbers of GLDM features:', len(results))
        print('\n')

        # Show NGTDM features
        ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)

        ngtdmFeatures.enableAllFeatures()
        print('Will calculate the following NGTDM features: ')
        print(list(ngtdmFeatures.enabledFeatures.keys()))

        print('Calculating NGTDM features...')
        results = ngtdmFeatures.execute()
        print('done')

        print('Calculated NGTDM features: ')
        for (key, val) in six.iteritems(results):
            print('  ', key, ':', val)
        print('Numbers of NGTDM features:', len(results))
        print('\n')

        # Show FirstOrder features, calculated on a LoG filtered image
        if applyLBP:
            print('Image Type: LBP')
            for lbpImage, imageTypeName, inputKwargs in imageoperations.getLBP3DImage(image, mask):
                lbpFirstorderFeatures = firstorder.RadiomicsFirstOrder(lbpImage, mask, **inputKwargs)
                lbpFirstorderFeatures.enableAllFeatures()
                results = lbpFirstorderFeatures.execute()
                for (key, val) in six.iteritems(results):
                    lbpFeatureName = '%s_%s' % (imageTypeName, key)
                    print('  ', lbpFeatureName, ':', val)

        # Show FirstOrder features, calculated on a wavelet filtered image
        if applyWavelet:
            print('Image Type: Wavelet')
            for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
                waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(decompositionImage, mask, **inputKwargs)
                waveletFirstOrderFeaturs.enableAllFeatures()
                results = waveletFirstOrderFeaturs.execute()
                print('Calculated firstorder features with wavelet ', decompositionName)
                for (key, val) in six.iteritems(results):
                    waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                    print('  ', waveletFeatureName, ':', val)
        return

    else:
        # 参数设置：特征提取的一些常用参数
        if params_path:
            print('There are parameter files...')
            extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
            print("Extraction parameters:\n\t", extractor.settings)
            print("Enabled filters:\n\t", extractor.enabledImagetypes)
            print("Enabled features:\n\t", extractor.enabledFeatures)
        # 手动设置参数的办法
        else:
            print('Set parameters manually...')
            settings = {}
            settings['binWidth'] = 25  # 5
            settings['Interpolator'] = sitk.sitkBSpline
            settings['resampledPixelSpacing'] = [1, 1, 0]  # 3,3,3
            settings['voxelArrayShift'] = 1000  # 300
            # settings['normalize'] = True
            # settings['normalizeScale'] = 100
            settings['force2D'] = True
            settings['force2Ddimension'] = 0
            settings['label'] = 1
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            print('Extraction parameters:\n\t', extractor.settings)

            # 设置提取的特征类型
            # 默认是有original类型的关闭只需要提前关闭
            extractor.disableAllImageTypes()
            extractor.enableImageTypeByName('Original')
            extractor.enableImageTypeByName('LBP2D')
            extractor.enableImageTypeByName('Wavelet')

            # 默认提取所有的特征
            # extractor.enableAllFeatures()

            # 通过特征类型选择提取的特征
            extractor.enableFeaturesByName(firstorder=['10Percentile', '90Percentile', 'Energy', 'Entropy',
                                                       'InterquartileRange', 'Kurtosis', 'Maximum', 'StandardDeviation',
                                                       'MeanAbsoluteDeviation', 'Mean', 'Median', 'Minimum', 'Range',
                                                       'RobustMeanAbsoluteDeviation', 'RootMeanSquared', 'Skewness',
                                                       'TotalEnergy', 'Uniformity', 'Variance']
                                           )

            extractor.enableFeaturesByName(shape2D=['MeshSurface', 'PixelSurface', 'Perimeter', 'PerimeterSurfaceRatio',
                                                  'Sphericity', 'SphericalDisproportion', 'MaximumDiameter', 'MajorAxisLength',
                                                  'MinorAxisLength', 'Elongation']
                                           )

            extractor.enableFeaturesByName(GLCM=['Autocorrelation', 'ClusterProminence', 'ClusterShade',
                                                 'ClusterTendency', 'Contrast', 'Correlation', 'DifferenceAverage',
                                                 'DifferenceEntropy', 'DifferenceVariance', 'Id', 'Idm', 'Idmn', 'Idn',
                                                 'Imc1', 'Imc2', 'InverseVariance', 'JointAverage', 'JointEnergy',
                                                 'JointEntropy', 'MCC', 'MaximumProbability', 'SumAverage',
                                                 'SumEntropy', 'SumSquares']
                                           )
            extractor.enableFeaturesByName(GLRLM=['Autocorrelation', 'ClusterProminence', 'ClusterShade',
                                                  'ClusterTendency', 'Contrast', 'Correlation', 'DifferenceAverage',
                                                  'DifferenceEntropy', 'DifferenceVariance', 'Id', 'Idm', 'Idmn',
                                                  'Idn', 'Imc1', 'Imc2', 'InverseVariance', 'JointAverage',
                                                  'JointEnergy', 'JointEntropy', 'MCC', 'MaximumProbability',
                                                  'SumAverage', 'SumEntropy', 'SumSquares']
                                           )
            extractor.enableFeaturesByName(GLSZM=['GrayLevelNonUniformity', 'GrayLevelNonUniformityNormalized',
                                                  'GrayLevelVariance', 'HighGrayLevelZoneEmphasis', 'LargeAreaEmphasis',
                                                  'LargeAreaHighGrayLevelEmphasis', 'LargeAreaLowGrayLevelEmphasis',
                                                  'LowGrayLevelZoneEmphasis', 'SizeZoneNonUniformity',
                                                  'SizeZoneNonUniformityNormalized', 'SmallAreaEmphasis',
                                                  'SmallAreaHighGrayLevelEmphasis', 'SmallAreaLowGrayLevelEmphasis',
                                                  'ZoneEntropy', 'ZonePercentage', 'ZoneVariance']
                                           )

            extractor.enableFeaturesByName(GLDM=['DependenceEntropy', 'DependenceNonUniformity',
                                                 'DependenceNonUniformityNormalized', 'DependenceVariance',
                                                 'GrayLevelNonUniformity', 'GrayLevelVariance',
                                                 'HighGrayLevelEmphasis', 'LargeDependenceEmphasis',
                                                 'LargeDependenceHighGrayLevelEmphasis',
                                                 'LargeDependenceLowGrayLevelEmphasis',
                                                 'LowGrayLevelEmphasis','SmallDependenceEmphasis',
                                                 'SmallDependenceHighGrayLevelEmphasis',
                                                 'SmallDependenceLowGrayLevelEmphasis']
                                           )

            extractor.enableFeaturesByName(NGTDM=['Busyness', 'Coarseness', 'Complexity', 'Contrast', 'Strength'])

            print("Enabled filters:\n\t", extractor.enabledImagetypes)
            print("Enabled features:\n\t", extractor.enabledFeatures)

        #默认禁用的特征都手动启用，为了之后特征筛选
        feature_cur = []
        feature_name = []
        result = extractor.execute(image, mask)
        for key, value in six.iteritems(result):
            feature_name.append(key)
            feature_cur.append(value)
        name = feature_name[37:]
        name = np.array(name)

        for i in range(len(feature_cur[37:])):
            feature_cur[i+37] = float(feature_cur[i+37])
        return feature_cur[37:], name

#特征展示测试
# catch_features(None, None, params_path=None, show_mode=True)
#特征提取的CT图像路径
ufilename = "./example_data"
# ufilename = "../HCC/2010-2017"
#设置特征提取相关参数配置文件的路径
para_path = './exampleCT.yaml'
save_file = []
save_curdata = []
name = []
for filename in os.listdir(ufilename):
    case_path = os.path.join(ufilename, filename)
    print(case_path)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
    if len(dicom_names) == 0:
        continue
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    for nii_path in os.listdir(case_path):
        (niiname, extension) = os.path.splitext(nii_path)
        if extension == '.gz':
            mask = sitk.ReadImage(os.path.join(case_path, nii_path))
    try:
        save_curdata, name = catch_features(image, mask, params_path=None, show_mode=False)  # 抽取特征 模块参数输入的是文件名
    except:
        continue

    save_curdata = np.array(save_curdata)
    save_file.append(save_curdata)

save_file = np.array(save_file)
name_df = pd.DataFrame(save_file)
name_df.columns = name
name_df.to_csv('../HCC/2010-2017/Radiomics-features.csv')

# writer = pd.ExcelWriter('../HCC/2010-2017/Radiomics-features.xlsx')
# name_df.to_excel(writer)
# writer.save()
