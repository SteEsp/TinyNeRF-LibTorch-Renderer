// From: https://pytorch.org/tutorials/advanced/cpp_export.html
// Run with: .\build\RelWithDebInfo\import_trained_model.exe traced_resnet_model.pt 

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc,  char* argv[]) {
  
  if (argc != 2) {
    std::cout << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch ( c10::Error& e) {
    std::cout << "error loading the model\n";
    return -1;
  }
  std::cout << "Model loaded!\n";

  try {
    module.to(torch::kCUDA);
  }
  catch ( c10::Error& e) {
    std::cout << "error loading the model in GPU\n";
    return -1;
  }
  std::cout << "Model loaded in GPU!\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 90}).to(torch::kCUDA));
  std::cout << "Created input tensor and moved to GPU!\n";

  at::Tensor output;
  try {
    // Execute the model and turn its output into a tensor.
    output = module.forward(inputs).toTensor();
  }
  catch ( c10::Error& e) {
    std::cout << "error querying the model\n";
    return -1;
  }
  std::cout << "Query operated successfully!\n";
  
  try {
    output = output.cpu();
  }
  catch ( c10::Error& e) {
    std::cout << "error moving the result back to CPU\n";
    return -1;
  }
  std::cout << output << '\n';

}
