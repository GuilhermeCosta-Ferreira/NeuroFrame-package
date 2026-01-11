class Dunders:
    def __repr__(self):
        return f"Mouse(id={self.id}, micro_ct={self.micro_ct.path}, mri={self.mri.path}, segmentations={self.segmentations.path})"
    def __str__(self):
        return f"Mouse {self.id} with the following data:\n    MRI at {self.mri.path}\n    μCT at {self.micro_ct.path}\n    Segmentations at{self.segmentations.path}"