MAG-Net is an implementation of the“MAG-Net: Multi-stage Attention-Guided Learning for  Peripherally Inserted Central Catheter (PICC) segmentation in Chest X-rays. 
This respository contains the network definitions and other necessary codes. 


Installation:
	python 3.6
	numpy
	opencv-python
	simpleitk
	scikit-image
	scilit-learn
	tensorboard
	scipy
	torch 1.4

train：
	train_Global_UP_K_fold.py: train code for coarse stage.
	Train_2_return.py : train code for fine stage.
test：
	test_Global_model.py : test code for coarse stage (UNet)
	test_Global_up_model.py: test code for coarse stage (our proposed CS-Net)
	test_Local_model.py: test code for fine stage
	local_test2.py : test code for drawing ROC-curves in fine stage.

config：
	func.py: some important codes (Normlization ,...)
	test_function: some function code for test. (post-processing function, clip function and so on.)

New_Model:
	These models were constructed by conducting ablation experiments in fine stage, and we selected the best model as: UNet_SW1_PAM_D.

model:
	Unet_UP_global: our proposed  CS-Net
	other: other popular network (which are used for comparison in fine stage)

Data:

The data used in our experiments is privately owned, and you can test it with any publicly available high-resolution dataset, or you can send us an email to request access to the data.  We have given a sample so you can get a little idea of our data


Contact：
If you have any questions about how the code works, etc., you can send an email and we'll do our best to respond.
email : 2111912047@zjut.edu.cn
