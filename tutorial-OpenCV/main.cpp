#include<iostream>

#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

//For generating Pseudo Random Numbers
#include<random>

int __gfnGenRandomNumber(int iaLowerBound, int iaUpperBound) {
	std::random_device clRD;
	std::mt19937 gen(clRD());
	std::uniform_int_distribution<> clDistrib(iaLowerBound, iaUpperBound);
	return clDistrib(gen);
}

/*
* Unique Learnigns:
* Use cv::Mat_<uchar>, and clImage(iRow,iCol) instead of clImage.at<uchar>(iRow,iCol)
* Use cv::Mat_<cv::Vec3b>, and clImage(iRow,iCol) instead of clImage.at<cv::Vec3b>(iRow,iCol)
* Use uchar* uchpData = clImage.ptr<iRow> to access pixel value (for multichannel: W*3) of entire row 
* Use cv::Mat::creat(): verify if the output image has an allocated data buffer with a size and pixel type that matches the ones of the input image. This is the method to use when a matrix must be re-allocated with a new size and type. If, by chance, the matrix already has the size and type specified, then no operation is performed and the method simply returns without touching the instance.
* Use cv::Mat::isContinuous() for verifying whether the image is padded or not
* The method reshape changes the matrix dimensions without requiring any memory copy or re-allocation. The first parameter is the new number of channels and the second one is the new number of rows. The number of columns is readjusted accordingly.
* Use clImage.data to get the pointer to underlying type; use clImage.step, to go to the next row; includes padding
* Use cv::MatIterator_<cv::Vec3b> it for RGB STL style iterator; cv::Mat_<cv::Vec3b>::iterator it = clImage.begin<cv::Vec3b>()
* use cv::MatConstIterator_<cv::Vec3b> for constant iterator to normal or const cv::Mat_<cv::Vec3b>; cv::Mat_<cv::Vec3b>::const_iterator
* Multi-threading is another way to increase the efficiency of your algorithms, especially since the advent of multi-core processors. OpenMP and the Intel Threading Building Blocks (TBB) are two popular APIs used in concurrent programming to create and manage your threads.
* This is done by changing negative values to 0 and values over 255 to 255. This is exactly what the cv::saturate_cast<uchar> function is doing
* Use cv::filter2D(nclImage,clImage,nclImage.depth(),kernel) to apply any kernel, specially for larger kernels
* 
*/

namespace Chapter2 {

	/// <summary>
	/// Adds the salt to the image, by setting the value of pixel to 255, depending on channels on the image
	/// </summary>
	/// <param name="clarImage"></param>
	/// <param name="iNum"></param>
	void __gfnsalt(cv::Mat& clarImage, int iNum) {
		for (auto i{ 0 }; i < iNum; ++i) {

			//Column & Row to modify
			auto iCol = __gfnGenRandomNumber(0, clarImage.cols-1);
			auto iRow = __gfnGenRandomNumber(0, clarImage.rows-1);

			if (clarImage.channels() == 1) {
				clarImage.at<uchar>(iRow, iCol) = 255;
			}
			else if (clarImage.channels() == 3) {
				clarImage.at<cv::Vec3b>(iRow,iCol) = cv::Vec3b{255,255,255};
			}
		}
	}

	/// <summary>
	/// This function returns the reduced color image
	/// </summary>
	/// <param name="clarInputImage"> Input Image</param>
	/// <param name="claOutputImage"> use the Input Image for In-place Transform</param>
	/// <param name="niDiv"> Factor Reduction</param>
	void __gfnReduceColor(const cv::Mat& nclarInputImage, cv::Mat& clarOutputImage, int niDiv = 64) {
		
		/*
		* In a color image, the first 3 bytes of the image data buffer gives the 3 color channel values of 
		* the upper left pixel, the next 3 bytes are the values of the second pixel of the first row, and so 
		* on (note that OpenCV uses, by default, BGR channel order, so blue is usually the first channel). 
		* An image of width W and height H would then require a memory block of WxHx3 uchars. 
		* However, for efficiency reasons, the length of a row can be padded with few extra pixels. This 
		* is because some multimedia processor chips (for example, the Intel MMX architecture) can 
		* process images more efficiently when their rows are multiples of 4 or 8. Obviously, these 
		* extra pixels are not displayed or saved, their exact values are ignored. OpenCV designates 
		* the length of a padded row as the keyword. Obviously, if the image has not been padded with 
		* extra pixels, the effective width will be equal to the real image width. The data attribute cols
		* gives you the image width (that is the number of columns), and the attribute rows gives you 
		* the image height while the step data attribute gives you the effective width in number of 
		* bytes. Even if your image is of a type other than uchar, step will still give you the number of 
		* bytes in a row. The size of a pixel element is given by method elemSize (for example, for a 
		* 3-channel short integer matrix (CV_16SC3), elemSize will return 6). The number of channels 
		* in the image is given by the nchannels method (which will be 1 for a gray-level image and 
		* 3 for a color image). Finally, method total returns the total number of pixels (that is matrix 
		* entries) in the matrix.
		*/

		/*
		* create() always create a continous image, that is an image with no padding.
		* The memory block allocated has a size of total()*elemSize()
		*/

		clarOutputImage.create(nclarInputImage.rows, nclarInputImage.cols, nclarInputImage.type());
		int iRows = nclarInputImage.rows, iColsChannels = nclarInputImage.cols * nclarInputImage.channels();


		//if No padding is present
		if (clarOutputImage.isContinuous()) {

			//Option-1
			{
				iColsChannels *= iRows;
				iRows = 1;
			}

			//Option-2
			//{	
			//	clarOutputImage.reshape(1, nclarInputImage.rows * nclarInputImage.cols);
			//	iRows = clarOutputImage.rows;
			//	iColsChannels = clarOutputImage.cols * clarOutputImage.channels();
			//}
		}
		
		for (auto iRow{ 0 }; iRow < iRows; iRow++) {
			const uchar* nuchpInputData = nclarInputImage.ptr<uchar>(iRow);
			uchar* uchpOutputData = clarOutputImage.ptr<uchar>(iRow);
			for (auto iChannelCol{ 0 }; iChannelCol < iColsChannels; iChannelCol++) {
				
				//Option - 1
				//uchpOutputData[iChannelCol] = nuchpInputData[iChannelCol] / niDiv * niDiv + niDiv / 2;
			
				//Option - 2, slower; reading the pixel value twice
				//uchpOutputData[iChannelCol] = nuchpInputData[iChannelCol] - nuchpInputData[iChannelCol] % niDiv + niDiv / 2;

				//Option -3, Masking, but needs to ensure that niDiv is power of 2
				uchpOutputData[iChannelCol] = (nuchpInputData[iChannelCol] & niDiv) + niDiv / 2;
			}
		}

	}

	/// <summary>
	/// Measures the time taken to run the function using # clock cycles and cycle frequency
	/// This duration calls should be averaged over several calls of the function
	/// </summary>
	/// <returns> in ms. total time elapsed </returns>
	double __gfnMeasureDuration() {

		double duration = static_cast<double>(cv::getTickCount());

		//Function to measure

		duration = static_cast<double>(cv::getTickCount()) - duration;
		return duration /= cv::getTickFrequency();
	}

	/// <summary>
	/// Only for Grayscale
	/// Applies Sharpenning to image. If Laplacian of an image is subtracted from image, edges are amplified.
	/// sharpened_pixel = 5 * current -left -right -up -down;
	/// </summary>
	/// <param name="clarInputImage"> Input Image</param>
	/// <param name="clarOutputImage"> Output Image</param>
	void __gfnSharpenImage(const cv::Mat& nclarInputImage, cv::Mat& clarOutputImage) {
	
		clarOutputImage.create(nclarInputImage.rows, nclarInputImage.cols, nclarInputImage.type());

		for (auto iRow{ 1 }; iRow < nclarInputImage.rows-1; iRow++) {
			const uchar* nchpPreviousRow = nclarInputImage.ptr<uchar>(iRow-1);
			const uchar* nchpCurrentRow = nclarInputImage.ptr<uchar>(iRow);
			const uchar* nchpNextRow = nclarInputImage.ptr<uchar>(iRow+1);
			
			uchar* uchpOutputRow = clarOutputImage.ptr<uchar>(iRow);

			for (auto iCol{ 1 }; iCol < nclarInputImage.cols - 1; iCol++) {
				*uchpOutputRow++ = cv::saturate_cast<uchar>(
					5 * nchpCurrentRow[iCol] - nchpCurrentRow[iCol-1] - nchpCurrentRow[iCol+1]
					- nchpPreviousRow[iCol] - nchpNextRow[iCol]
				);
			}
		}

		clarOutputImage.row(0).setTo(cv::Scalar(0));
		clarOutputImage.row(clarOutputImage.rows-1).setTo(cv::Scalar(0));
		clarOutputImage.col(0).setTo(cv::Scalar(0));
		clarOutputImage.col(clarOutputImage.cols - 1).setTo(cv::Scalar(0));
	}

	/// <summary>
	/// Applies Sharpening to image using kernel. For larger kernel, use of cv::filter2D is encouraged as it is optimized for it
	/// </summary>
	/// <param name="nclarInputImage"></param>
	/// <param name="clarOutputImage"></param>
	void __gfnSharpenImage2(const cv::Mat& nclarInputImage, cv::Mat& clarOutputImage) {
		const auto nclLapLacianKernel = cv::Mat{
			{3,3},
			{
				.0,-1.,0.,
				-1.,5.,-1.,
				.0,-1.,.0
			}
		};
		cv::filter2D(nclarInputImage, clarOutputImage, nclarInputImage.depth(), nclLapLacianKernel);
	}

	void __gfnImageArithmetic(const cv::Mat& nclarInputImage1, const cv::Mat& nclarInputImage2, cv::Mat& clarOutputImage) {
		cv::addWeighted(nclarInputImage1, 0.9, nclarInputImage2, 0.7, 0.0, clarOutputImage);
	}

	/*
	*	// create vector of 3 images
	*	std::vector<cv::Mat> planes;
	*	// split 1 3-channel image into 3 1-channel images
	*	cv::split(image1,planes);
	*	// add to blue channel
	*	planes[0]+= image2;
	*	// merge the 3 1-channel images into 1 3-channel image
	*	cv::merge(planes,result);
	* 
	*/
}

int main() {

	cv::Mat clInputImage = cv::imread("input/ranikhet.jpg", cv::IMREAD_COLOR);
	cv::Mat clOutputImage;

	//Use GrayScale
	//{
	//	cv::cvtColor(clInputImage, clInputImage, cv::COLOR_BGR2GRAY);
	//}

	//Chapter2::__gfnsalt(clOutputImage, 5000);
	//Chapter2::__gfnReduceColor(clInputImage,clOutputImage,128);
	//GrayScale the image
	//Chapter2::__gfnSharpenImage(clInputImage, clOutputImage);
	//Chapter2::__gfnSharpenImage2(clInputImage, clOutputImage);
	
	//cv::imshow("Normal RaniKhet", clInputImage);
	cv::Mat clInputImage2 = cv::imread("input/ranikhet2.jpg", cv::IMREAD_COLOR);
	Chapter2::__gfnImageArithmetic(clInputImage, clInputImage2, clOutputImage);
	cv::imshow("Output", clOutputImage);
	cv::waitKey(0);
	
	
	return 0;
}