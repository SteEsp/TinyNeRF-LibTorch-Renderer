// Pytorch C++ conversion of https://github.com/krrish94/nerf-pytorch renderer
// Many thanks to: https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-2-tracing-your-pytorch-model/

// Run with: .\build\RelWithDebInfo\renderer.exe traced_models/traced_tiny_nerf.pt

#include <torch/script.h> // One-stop header.

/*
#include <glm/vec3.hpp> // vec3
#include <glm/vec4.hpp> // vec4
#include <glm/mat3x3.hpp> // mat3
#include <glm/mat4x4.hpp> // mat4
#include <glm/ext/matrix_transform.hpp> // translate, rotate, scale
#include <glm/ext/matrix_clip_space.hpp> // perspective
#include <glm/ext/scalar_constants.hpp> // pi
*/

#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>

using std::string;
using namespace glm;
using namespace torch::indexing;
// namespace F = torch::nn::functional;

bool load_module_to_gpu(torch::jit::script::Module &model, string path) {

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(path);
    model.to(torch::kCUDA);
  }
  catch ( c10::Error& e) {
    return false;
  }

  return true;
}

/** Mimick np.meshgrid(..., indexing="xy") in libtorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch::Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch::Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
*/
std::vector<torch::Tensor> meshgrid_xy(
    torch::Tensor tensor1, torch::Tensor tensor2
) {
    
    // TODO: test
    std::vector<torch::Tensor> tensors = {tensor1, tensor2};
    auto meshgrid = torch::meshgrid(torch::TensorList(tensors));
    auto ii = meshgrid[0];
    auto jj = meshgrid[1];

    return {ii.transpose(-1, -2), jj.transpose(-1, -2)};
}


/**
 * Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

  Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
  
  Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
*/
torch::Tensor cumprod_exclusive(torch::Tensor tensor) {

    // TODO: test

    // Only works for the last dimension (dim=-1)
    auto dim = -1;
    // Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    auto cumprod = torch::cumprod(tensor, dim);
    // "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch::roll(cumprod, 1, dim);
    // Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod.index({"...", 0}) = 1.;
  
    return cumprod;

}


/**
 * Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

  Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch::Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.
  
  Returns:
    ray_origins (torch::Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch::Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
*/
std::vector<torch::Tensor> get_ray_bundle(
    int height, int width, float focal_length, 
    torch::Tensor tform_cam2world
) {

    torch::Tensor ray_origins;

    torch::Tensor ray_directions;

    return {ray_origins, ray_directions};

}

/**
 * Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
  variables indicate the bounds within which 3D points are to be sampled.

  Args:
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
      coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
      coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
      randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
      By default, this is set to `True`. If disabled (by setting to `False`), we sample
      uniformly spaced points along each ray in the "bundle".
  
  Returns:
    query_points (torch.Tensor): Query points along each ray
      (shape: :math:`(width, height, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
*/
std::vector<torch::Tensor> compute_query_points_from_rays(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    float near_thresh,
    float far_thresh,
    int num_samples,
    bool randomize=true
) {

    // TODO: test

    // shape: (num_samples)
    torch::Tensor depth_values = torch::linspace(near_thresh, far_thresh, num_samples).to(ray_origins);

    if (randomize) {
        // ray_origins: (width, height, 3)
        // noise_shape = (width, height, num_samples)
        // TODO: continue
    } 

    // (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    // query_points:  (width, height, num_samples, 3)
    torch::Tensor query_points = ray_origins.index({"...", None, Slice()}) + ray_directions.index({"...", None, Slice()}) * depth_values.index({"...", Slice(), None});
    
    // TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.

    return {query_points, depth_values};
}


/**
 * Differentiably renders a radiance field, given the origin of each ray in the
  "bundle", and the sampled depth values along them.

  Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
  
  Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
*/
std::vector<torch::Tensor> render_volume_density(
    torch::Tensor radiance_field,
    torch::Tensor ray_origins,
    torch::Tensor depth_values
) {

    // TODO: test

    // auto sigma_a = F::relu(radiance_field.index({"...", 3}));
    auto sigma_a = torch::relu(radiance_field.index({"...", 3}));
    auto rgb = torch::sigmoid(radiance_field.index({"...", Slice(None, None, 3)}));
    auto one_e_10 = torch::tensor({1e10}, 
        torch::TensorOptions().dtype(ray_origins.dtype()).device(ray_origins.device())
    );
    auto dists = at::cat((depth_values.index({"...", Slice(1, None, None)}) - depth_values.index({"...", Slice(None, None, -1)}),
                  one_e_10.expand(depth_values.index({"...", Slice(None, None, 1)}).sizes())), -1); // dim=-1
    auto alpha = 1. - torch::exp(-sigma_a * dists);
    auto weights = alpha * cumprod_exclusive(1. - alpha + 1e-10);

    auto rgb_map = (weights.index({"...", None}) * rgb).sum(-2); // dim=-2
    auto depth_map = (weights * depth_values).sum(-1); // dim=-1
    auto acc_map = weights.sum(-1);

    return {rgb_map, depth_map, acc_map};

}


/**
 * Apply positional encoding to the input.

  Args:
    tensor (torch.Tensor): Input tensor to be positionally encoded.
    num_encoding_functions (optional, int): Number of encoding functions used to
        compute a positional encoding (default: 6).
    include_input (optional, bool): Whether or not to include the input in the
        computed positional encoding (default: True).
    log_sampling (optional, bool): Sample logarithmically in frequency space, as
        opposed to linearly (default: True).
  
  Returns:
    (torch.Tensor): Positional encoding of the input tensor.
*/
torch::Tensor positional_encoding(torch::Tensor tensor, int num_encoding_functions=6,
  bool include_input=true, bool log_sampling=true) {

    // TODO: test

    std::vector<torch::Tensor> encoding;

    //Trivially, the input tensor is added to the positional encoding.
    if (include_input)
      encoding.push_back(tensor);

    // Now, encode the input using a set of high-frequency functions and append the
    // resulting values to the encoding.
    torch::Tensor frequency_bands;
    if (log_sampling) {
        frequency_bands = pow(2.0, torch::linspace(
              0.0,
              num_encoding_functions - 1,
              num_encoding_functions,
              torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device())
        ));
    } else {
        frequency_bands = torch::linspace(
          pow(2.0, 0.0),
          pow(2.0, (num_encoding_functions - 1)),
          num_encoding_functions,
          torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device())
        );
    }

    // Iterate over each frequency band
    float* freq_ptr = (float*) frequency_bands.data_ptr(); 
    for (int freq = 0; freq < frequency_bands.sizes()[0]; ++freq) {
        encoding.push_back(torch::sin(tensor * *freq_ptr++));
        encoding.push_back(torch::cos(tensor * *freq_ptr++));
    }

    // Special case, for no positional encoding
    if (encoding.size() == 1) 
        return encoding[0];
    else
        return at::cat(encoding, -1); // dim=-1

}


/**
 * Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
*/
std::vector<torch::jit::IValue> get_minibatches(torch::Tensor inputs, int chunksize=1024*8) {
    
    std::vector<torch::jit::IValue> minibatches;

    // TODO: test 
    for (int i = 0; i < inputs.sizes()[0]; i+=chunksize) {
        auto batch = inputs.index({Slice(i, None, i + chunksize)}); // [i:i + chunksize]
        minibatches.push_back(batch);
    }

    return minibatches;

} 


/**
 * One iteration of TinyNeRF (forward pass) 
*/
torch::Tensor render(const torch::jit::script::Module model, const int height, const int width, const float focal_length, 
            const torch::Tensor tform_cam2world, const float near_thresh, 
            const float far_thresh, const int depth_samples_per_ray,
            const int num_encoding_functions, const int chunksize
            ) {

    // TODO: test

    std::vector<torch::Tensor> res;

    // Get the "bundle" of rays through all image pixels.
    res = get_ray_bundle(height, width, focal_length, tform_cam2world);
    auto ray_origins = res[0];
    auto ray_directions = res[1];
    
    // Sample query points along each ray
    res = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    );
    auto query_points = res[0];
    auto depth_values = res[1]; 

    // "Flatten" the query points.
    auto flattened_query_points = query_points.reshape({-1, 3});

    // Encode the query points (default: positional encoding).
    auto encoded_points = positional_encoding(flattened_query_points, num_encoding_functions);

    // Split the encoded points into "chunks", run the model on all chunks, and
    //  concatenate the results (to avoid out-of-memory issues).
    auto batches = get_minibatches(encoded_points, chunksize);

    std::vector<torch::Tensor> predictions;
    for(auto &batch: batches) {
        // model.forward returns an IValue which we convert back to a Tensor
        auto output = model.forward(batch).toTensor();
        predictions.push_back(output);
    }
    auto radiance_field_flattened = at::cat(predictions, 0); // dim=0

    // "Unflatten" to obtain the radiance field.
    auto radiance_field = torch::reshape(radiance_field_flattened, (query_points.sizes()[0], query_points.sizes()[1], 4));

    // Perform differentiable volume rendering to re-synthesize the RGB image.
    res = render_volume_density(radiance_field, ray_origins, depth_values);
    auto rgb_predicted = res[0];

    return rgb_predicted;

}


int main(int argc,  char* argv[]) {
  
    if (argc != 2) {
        std::cout << "usage: renderer <path-to-traced-model>\n";
        return -1;
    }

    torch::jit::script::Module model;

    load_module_to_gpu(model, string(argv[1]));

    // Set to `eval` model (just like Python)
    model.eval();     
    // Within this scope/thread, don't use gradients (again, like in Python)     
    torch::NoGradGuard no_grad_;
    
    auto tform_cam2world = torch::tensor({
        { 6.8935e-01,  5.3373e-01, -4.8982e-01, -1.9745e+00},
        {-7.2443e-01,  5.0789e-01, -4.6611e-01, -1.8789e+00},
        { 1.4901e-08,  6.7615e-01,  7.3676e-01,  2.9700e+00},
        { 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00}}).to(torch::kCUDA);

    // std::cout << tform_cam2world;

    /* 
    mat4 mat4_cam2world(
        6.8935e-01,  5.3373e-01, -4.8982e-01, -1.9745e+00,
        -7.2443e-01,  5.0789e-01, -4.6611e-01, -1.8789e+00,
        1.4901e-08,  6.7615e-01,  7.3676e-01,  2.9700e+00,
        0.0,  0.00,  0.0,  1.0
    );

    auto target_tform_cam2world = torch::zeros({4, 4});
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            target_tform_cam2world.index_put_({i, j}, mat4_cam2world[i][j]);
    target_tform_cam2world.to(torch::kCUDA);

    std::cout << target_tform_cam2world;    
    */

    /* Parameters */

    // Image height and width
    int image_width = 100;
    int image_height = 100;

    // Near and far clipping thresholds for depth values
    float near_thresh = 2.;
    float far_thresh = 6.;

    // Focal lenght
    float focal_length = 0.0;

    int depth_samples_per_ray = 6;
    
    // Number of functions used in the positional encoding (Be sure to update the 
    // model if this number changes).
    int num_encoding_functions = 6;

    // Chunksize (Note: this isn't batchsize in the conventional sense. This only
    // specifies the number of rays to be queried in one go. Backprop still happens
    // only after all rays from the current "bundle" are queried and rendered).
    int chunksize = 16384; // Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.

    render(model, image_height, image_width, focal_length, tform_cam2world, 
           near_thresh, far_thresh, depth_samples_per_ray,
           num_encoding_functions, chunksize);
    /*
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 90}).to(torch::kCUDA));
    std::cout << "Created input tensor and moved to GPU!\n";

    torch::Tensor output;
    try {
        // Execute the model and turn its output into a tensor.
        output = model.forward(inputs).toTensor();
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
    */
}
