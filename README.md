#  Improving Computed Tomography (CT) Reconstruction and Automatic TB Classification

**This repository contains code and data used in the paper:**

" Improving Computed Tomography (CT) Reconstruction via 3D Shape Induction" by \
Elena Sizikova (York University),  \
Xu Cao (York University),  \
Ashia Lewis (The University of Alabama),  \
Kenny Moise (Université Quisqueya), \
Megan Coffee (NYU Grossman School of Medicine)  \

Chest computed tomography (CT) imaging adds valuable insight in the diagnosis and management of pulmonary infectious diseases, like tuberculosis (TB). However, due to the cost and resource limitations, only X-Ray images may be available for initial diagnosis or follow up comparison imaging during treatment. Due to their projective nature, X-Rays may exhibit noise artifacts and be more difficult to interpret, especially for clinicians with less specialized radiology training. The lack of publicly available paired X-Ray and CT image datasets makes it challenging to train a 3D reconstruction model. In addition, Chest X-Ray radiology may rely on different device modalities with varying image quality and there may be variation in underlying population disease spectrum that creates diversity in inputs. We propose shape induction, that is, learning the shape of 3D CT from X-Ray without CT supervision, as a novel technique to incorporate realistic X-Ray distributions during training of a reconstruction model. Our experiments demonstrate that this process improves both the perceptual quality of generated CT and the accuracy of down-stream classification of pulmonary infectious diseases.\footnote{Our code will be made publicly available upon acceptance.} 

