#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

bool inFrame(const Mat src,int y, int x){
  return (x>=0 && y>=0 && x<src.cols && y<src.rows);
}

//MORPHOLOGICAL OPERATIONS USING STRUCTURING ELEMENTS

void dilatation(const Mat src, Mat &dst, const Mat elt){

  //variables de translation pour centrer l'elt en p
  int transx=elt.cols/2, transy=elt.rows/2;
  int tx, ty, max;

  //pour chaque point p de l'image
  for(int py=0;py<src.rows;py++)
    for(int px=0;px<src.cols;px++){
      max=0;
      //on parcourt l'elt struct
      for(int ej=0;ej<elt.rows;ej++)
	for(int ei=0;ei<elt.cols;ei++){
	  //on centre l'elt en p
	  tx=px-transx+ei;
	  ty=py-transy+ej;
	  //on prend le max des val dans l'elt struct centré en p
	  if(inFrame(src,ty,tx) && elt.at<uchar>(ej,ei)==1 && src.at<uchar>(ty,tx)>max)
	    max=src.at<uchar>(ty,tx);
	}
    
      //p prend la valeur de max
      dst.at<uchar>(py,px)=max;
    }

}

void erosion(const Mat src, Mat &dst, const Mat elt){
  
  //variables de translation pour centrer l'elt en p
  int transx=elt.cols/2, transy=elt.rows/2;
  int tx, ty, min;
  //pour chaque point p de l'image
  for(int py=0;py<src.rows;py++)
    for(int px=0;px<src.cols;px++){
      min=255;
      //on parcourt l'elt struct
      for(int ej=0;ej<elt.rows;ej++)
	for(int ei=0;ei<elt.cols;ei++){
	  //on centre l'elt en p
	  tx=px-transx+ei;
	  ty=py-transy+ej;
	  //on prend le max des val dans l'elt struct centré en p
	  if(inFrame(src,ty,tx) && elt.at<uchar>(ej,ei)==1 && src.at<uchar>(ty,tx)<min)
	    min=src.at<uchar>(ty,tx);
	}
    
      //p prend la valeur de max
      dst.at<uchar>(py,px)=min;
    }
  
}

// opening = dilatation + erosion
void ouverture(const Mat src, Mat &dst, const Mat elt) {
  dilatation(src, dst,elt);
  erosion(dst.clone(),dst,elt);
}

// closing = erosion + dilatation
void fermeture(const Mat src, Mat &dst, const Mat elt) {
  erosion(src, dst,elt);
  dilatation(dst.clone(),dst,elt);
}

//DENOISING

// denoising = closing + opening
void debruitage(const Mat src, Mat &dst, const Mat elt){
  fermeture(src,dst,elt);
  ouverture(dst.clone(),dst,elt);
}


//GRADIENTS

// internal gradient = difference( erosion , identity )
void gradientInterne(const Mat src, Mat &dst, const Mat elt){ // 
  erosion(src,dst,elt);
  dst=src-dst;
}

// external gradient = difference( dilatation , identity )
void gradientExterne(const Mat src, Mat &dst, const Mat elt){
  dilatation(src,dst,elt);
  dst = dst-src;
}
  
// morphological gradient = difference( erosion , dilatation)
void gradientMorphologique(const Mat src, Mat &dst, const Mat elt){
  Mat tmp(src.size(),CV_8UC1,Scalar(0));
  erosion(src,tmp,elt);
  dilatation(src,dst,elt);
  dst = dst-tmp;
}


//REGIONAL MINIMA

Mat minima(const Mat gradient){

  Mat gradThr(gradient.size(), CV_8UC1, Scalar(0));
  threshold(gradient, gradThr, 0, 255, THRESH_BINARY | THRESH_OTSU);

  vector< vector<Point> > contours;
  findContours(gradThr, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
  
  RNG rng(12345);
  Mat res(gradient.size(), CV_8UC1, Scalar(0)) ;
  for( int i = 0; i< contours.size(); i++ ) {
    Scalar color = Scalar( rng.uniform(0, 255) );
    drawContours(res, contours, static_cast<int>(i), color, -1);
  }
  
  return res;
}


//WATERSHED CUTS ALGORITHM : Minimal spanning tree

void partageDesEaux(const Mat grad, Mat &dst){

  Mat markers =  minima(grad);

  Mat gradBGR(grad.size(),grad.type(),Scalar(0));
  cvtColor(grad, gradBGR, COLOR_GRAY2BGR);

  markers.convertTo(markers, CV_32S);
 
  watershed(gradBGR, markers);
  
  markers.convertTo(markers, dst.type());

  markers.copyTo(dst);

}

void save(Mat I, string name){
  
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);

    try {
      imwrite(name, I, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
	return;
    }
}
  
//EXECUTION & DISPLAY

int main( int argc, char** argv )
{
    string imageName("easterEgg.png"); 					// Image by default
    int choice;
    if( argc == 3)
    {
        imageName = argv[1];
        choice = atoi(argv[2]);
    }else{
        if(argc != 1)
          cerr<<endl<<"Bad usage.";

        cerr<<endl<<endl
        <<argv[0]<<"  <path to image>  <operation>"<<endl<<endl
        << "Arg_2 is the number corresponding to the algorithm that will process the image."<<endl
        << "Check the correlation table below."<<endl<<endl
        <<"(1)   Dilatation"<<endl
        <<"(2)   Erosion"<<endl
        <<"(3)   Opening"<<endl
        <<"(4)   Closing"<<endl
        <<"(5)   Denoising"<<endl
        <<"(6)   Internal gradient"<<endl
        <<"(7)   External gradient"<<endl
        <<"(8)   Morphological gradient"<<endl
        <<"(9)   Regional minima segmentation"<<endl
        <<"(10)  Watershed algorithm"<<endl;
        return -1;
    }

    Mat image;
    image = imread(imageName.c_str(), IMREAD_GRAYSCALE); // Read the file
    int h = image.size().height;
    int w = image.size().width;
    if( image.empty() )                      		 // Check for invalid input
    {
        cerr <<endl<< "The path to the image isn't valid or the image cannot be opened." << endl ;
        return -1;
    }
    
    //initialisation de l'image resultat
    Mat res(image.size(), image.type(), Scalar(0));

    //initialisation de l'element structurant
    Mat eltStruct(3,3,CV_8UC1,Scalar(1));
    eltStruct.at<uchar>(0,0)=0;
    eltStruct.at<uchar>(2,0)=0;
    eltStruct.at<uchar>(0,2)=0;
    eltStruct.at<uchar>(2,2)=0;

    string oper_adj = "unknown";

    switch (choice)
    {
    case 1: //DILATATION
      {
        oper_adj = "dilatated";
        cout<<"DILATATION"<<endl;
        dilatation(image, res, eltStruct);
        break;
      }
    
    case 2: //EROSION
      {
        oper_adj = "eroded";
        cout<<"EROSION"<<endl;
        erosion(image, res, eltStruct);
        break;
      }

    case 3: //OUVERTURE
      {
        oper_adj = "openening";
        cout<<"OPENING"<<endl;
        ouverture(image, res, eltStruct);
        break;
      }
    
    case 4: //FERMETURE
      {
        oper_adj = "closing";
        cout<<"CLOSING"<<endl;
        fermeture(image,res, eltStruct);
        break;
      }
    
    case 5: //DEBRUITAGE
      {
        oper_adj = "denoised";
        cout<<"DENOISING"<<endl;
        debruitage(image, res, eltStruct);
        break;
      }

    case 6: //GRADIENT INTERNE
      {
        oper_adj = "internal gradient";
        cout<<"INTERNAL GRADIENT"<<endl;
        gradientInterne(image,res,eltStruct);
        break;
      }
    
    case 7: //GRADIENT EXTERNE
      {
        oper_adj = "external gradient";
        cout<<"EXTERNAL GRADIENT"<<endl;
        gradientExterne(image,res,eltStruct);
        break;
      }

    case 8: //GRADIENT MORPHOLOGIQUE
      {
        oper_adj = "morphological gradient";
        cout<<"MORPHOLOGICAL GRADIENT"<<endl;
        gradientMorphologique(image,res,eltStruct);
        break;
      }

    case 9: //MINIMA
      {
        oper_adj = "regional minima";
        cout<<"REGIONAL MINIMA"<<endl;
        vector<vector<Point>> markers;
        gradientMorphologique(image,res,eltStruct);
        res = minima(res);
        break;
      }

    case 10: //PARTAGE DES EAUX
      {
        oper_adj = "watershed cuts";
        cout<<"WATERSHED CUTS ALGORITHM"<<endl;
        Mat grad(image.size(),CV_8UC1,Scalar(0));
        gradientMorphologique(image,grad,eltStruct);
        partageDesEaux(grad, res);
        break;
      }

    default:
      {
        cerr<<endl<<"Arg_2 isn't valid. Try one in the list below."<<endl<<endl
        <<"(1)   Dilatation"<<endl
        <<"(2)   Erosion"<<endl
        <<"(3)   Opening"<<endl
        <<"(4)   Closing"<<endl
        <<"(5)   Denoising"<<endl
        <<"(6)   Internal gradient"<<endl
        <<"(7)   External gradient"<<endl
        <<"(8)   Morphological gradient"<<endl
        <<"(9)   Regional minima segmentation"<<endl
        <<"(10)  Watershed algorithm"<<endl;
        return -1;
      }

    }

    //Display the original image
    string window_name = imageName;
    namedWindow( window_name, WINDOW_AUTOSIZE ); 	// Create a window for display.
    imshow( window_name, image );                	// Show our image inside it.

    //Display the processed one
    string res_name = oper_adj+" image";
    namedWindow( res_name, WINDOW_AUTOSIZE );
    imshow(res_name, res);

    //Proceed...
    cout<<endl<<"Press S to SAVE the processed image in the current directory..."<<endl<<"or"<<endl<<"Press any other key to EXIT..."<<endl;
    char k = waitKey(0); 									// Wait for the image to be displayed
    if (k=='s' || k=='S'){
      save(res, "res_img.png");
      cout<<endl<<"Saved as \"res_img.png\" in the current directory."<<endl<<endl;
    }else{
      cout<<endl<<"Exit without saving."<<endl<<endl;
    }

    destroyAllWindows();

    return 0;
}
