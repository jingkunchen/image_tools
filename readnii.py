from nilearn import image
from nilearn.plotting import plot_stat_map, show
import nibabel as nib
from nilearn import datasets
haxby_dataset = datasets.fetch_haxby()
func_filename = haxby_dataset.func[0]
mean_img = image.mean_img(func_filename)
weight_img = nib.load('/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0gt/patient1_C0_manual.nii')
plot_stat_map(weight_img, mean_img, title='SVM weights')
show()