#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace tensorflow;


void CVMat_to_Tensor(Mat img, Tensor* converted_tensor, int image_height, int image_width){
  float *p = converted_tensor->flat<float>().data();
  cv::Mat tempMat(input_rows, input_cols, CV_32FC1, p);
  img.convertTo(tempMat, CV_32FC1);
}
void preporcess_image(Mat img){
  return;
}
int main(int argc, char* argv[]){
Session* session;
Status status = NewSession(SessionOptions(), &session);
GraphDef graph_def;
status = ReadBinaryProto(Env::Default(), "/Users/tef-itm/Documents/H5_CPP/", &graph_def);
status = session->Create(graph_def);
string input_node_name = "check the name";
string output_node_name = "check the name";

img = imread('/Users/tef-itm/Documents/EGT_DNOX/compressed_data_96/Backflow_line/0_0.jpg', 0);
input_image_height = img.size().height;
input_image_width = img.size().width;
input_image_channels = img.channels();

preprocess_image(img);

#height, width, channels should meet the requirement of the network.
Tensor input_data(DT_FLOAT, TensorShape({1, input_image_height, input_image_width, input_image_channels}));
CVMat_to_Tensor(img, &input_data, input_image_height, input_image_width);

std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{input_node_name, input_data}};
std::vector<tensorflow::Tensor> outputs;
status = session->Run(inputs, {output_node_name}, {}, &outputs);
Tensor output_data = outputs[0];
#for classification problems, the output_data is a tensor of shape [batch_size, class_num]
auto tp = output_data.tensor<float, 2>();

int class_num = output_data.shape().dim_size(1);
int output_class_id = -1;
double output_prob = 0.0;

for(int j = 0; j < output_dim; j++){
  if(tp(0, j) >= output_prob){
    output_class_id = j;
    output_prob = tp(0, j);
  }
}
cout << "Class index is: " << output_class_id << ", with prob " << output_prob << std::endl;
session->Close();
return 0;
}
