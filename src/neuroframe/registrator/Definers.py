# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
from ..logger import logger



class Definers:
    # ================================================================
    # 1. Section: Direct Parameters Assignments
    # ================================================================
    def define_loss(self, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
        if(self.loss == 'MI'): method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.bin_size)
        elif(self.loss == 'LS'): method.SetMetricAsMeanSquares()
        else: 
            logger.warning("Loss not supported yet, alter the class to add it. Deafulted to Mates Mutual Information (MI).")
            self.loss = 'MI'
            method = self.define_loss(method)

        return method
    
    def define_optimizer(self, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
        if(self.optimizer == 'LBFGS'): method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=self.gradientConvergenceTolerance,
                                                                    numberOfIterations=self.numberOfIterations,
                                                                    maximumNumberOfCorrections=self.maximumNumberOfCorrections)
        elif(self.optimizer == 'GD'): method.SetOptimizerAsGradientDescent(learningRate=self.learning_rate,
                                                                                        numberOfIterations=self.numberOfIterations,
                                                                                        convergenceMinimumValue=self.convergenceMinimumValue,
                                                                                        convergenceWindowSize=self.convergenceWindowSize,
                                                                                        estimateLearningRate=self.estimateLearningRate,
                                                                                        maximumStepSizeInPhysicalUnits=self.maximumStepSizeInPhysicalUnits
                                                                                        )
        elif(self.optimizer == 'Exhaustive'): method.SetOptimizerAsExhaustive(numberOfSteps=self.numberOfSteps,
                                                                                stepLength=self.stepLength)
        else:
            logger.warning("Optimizer not supported yet, alter the class to add it. Deafulted to GD.")
            self.optimizer = 'GD'
            method = self.define_optimizer(method)

        return method
    
    def define_dimension_transform(self) -> sitk.ImageRegistrationMethod:
        if(self.dimension == 3): return sitk.VersorRigid3DTransform()
        elif(self.dimension == 2): return sitk.Euler2DTransform()
        else:
            logger.warning("Dimension not supported yet, alter the class to add it. Deafulted to 3D.")
            self.dimension = 3
            return self.define_dimension_transform() 

            

    # ================================================================
    # 2. Section: Interpolation Assignments
    # ================================================================
    def define_registration_interpolator(self, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
        if(self.reg_interpolator == 'linear'): method.SetInterpolator(sitk.sitkLinear)
        elif(self.reg_interpolator == 'nearest'): method.SetInterpolator(sitk.sitkNearestNeighbor)
        else: 
            logger.warning("Interpolator not supported yet, alter the class to add it. Deafulted to Linear.")
            self.reg_interpolator = 'linear'
            method = self.define_registration_interpolator(method)

        return method
    
    def define_resample_interpolator(self, method: sitk.ResampleImageFilter) -> sitk.ResampleImageFilter:
        if(self.res_interpolator == 'linear'): method.SetInterpolator(sitk.sitkLinear)
        elif(self.res_interpolator == 'nearest'): method.SetInterpolator(sitk.sitkNearestNeighbor)
        else: 
            logger.warning("Interpolator not supported yet, alter the class to add it. Deafulted to Linear.")
            self.res_interpolator = 'linear'
            method = self.define_resample_interpolator(method)

        return method



    # ================================================================
    # 3. Section: Kwargs Assignments
    # ================================================================
    def define_center_type(self) -> sitk.ImageRegistrationMethod:
        if(self.rigid_type == 'moments'): return sitk.CenteredTransformInitializerFilter.MOMENTS
        elif(self.rigid_type == 'geometric'): return sitk.CenteredTransformInitializerFilter.GEOMETRIC
        else: 
            logger.warning("Rigid type not supported yet, alter the class to add it. Deafulted to 'moments'.")
            self.rigid_type = 'moments'
            return self.define_center_type()
    
    def define_multiple_resolutions(self, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
        if self.multiple_resolutions:
            method.SetShrinkFactorsPerLevel(shrinkFactors=self.shrinkFactors)
            method.SetSmoothingSigmasPerLevel(smoothingSigmas=self.smoothingSigmas)
            method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            return method
        else:
            logger.info("Multiple resolutions not enabled, Registration could take more time") 
            return method
