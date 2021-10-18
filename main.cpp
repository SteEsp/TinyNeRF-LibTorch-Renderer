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
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <string>

using std::string;
using std::vector;
// using namespace glm;
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

/**
 * inputs is a list of tensors jit::IValue, on the same device
 * as the model are. Output will also be on the same device the model is.
 */
bool query_model(
    torch::jit::script::Module model, 
    torch::Tensor input_batch, 
    torch::Tensor &output
) {

    // std::cout<< "[DEBUG] input_batch.sizes() = " << input_batch.sizes() << "\n";
    try {
        // Execute the model; model.forward returns an IValue 
        // which we convert back to a Tensor
        output = model.forward({input_batch}).toTensor();
        // std::cout<< "[DEBUG] output.sizes() = " << output.sizes() << "\n";
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
vector<torch::Tensor> meshgrid_xy(
    torch::Tensor tensor1, torch::Tensor tensor2
) {
    
    vector<torch::Tensor> tensors = {tensor1, tensor2};
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
vector<torch::Tensor> get_ray_bundle(
    int height, int width, float focal_length, 
    torch::Tensor tform_cam2world
) {

    // std::cout<< "[DEBUG} tform_cam2world shape: " << tform_cam2world.sizes() << "\n Data:" << tform_cam2world << "\n";

    auto res = meshgrid_xy(
      torch::arange(width).to(tform_cam2world),
      torch::arange(height).to(tform_cam2world)
    );
    torch::Tensor ii = res[0];
    torch::Tensor jj = res[1]; 

    // std::cout<< "[DEBUG} meshgrid_xy: " << ii << "\n" << jj << "\n";

    auto directions = torch::stack(
      {(ii - width * .5) / focal_length, 
      -(jj - height * .5) / focal_length,
      -torch::ones_like(ii)}, 
      -1); // dim=-1

    // std::cout<< "[DEBUG} directions shape: " << directions.sizes() << "\n"; // Data:" << directions << "\n";

    // ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    auto index_directions = directions.index({"...", None, Slice()});
    auto multiplier = tform_cam2world.index({Slice(None, 3), Slice(None, 3)});

    // std::cout<< "[DEBUG} indexed directions shape: " << index_directions.sizes() << "\n";
    // std::cout<< "[DEBUG} directions multiplier shape: " << multiplier.sizes() << "\n  Data:" << multiplier << "\n";

    auto op1 = index_directions * multiplier;
    auto op2 = torch::sum(op1, -1); // dim=-1
    torch::Tensor ray_directions = op2;
    // ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    torch::Tensor ray_origins = tform_cam2world.index({Slice(None, 3), -1}).expand(ray_directions.sizes());

    // std::cout<< "[DEBUG}ray_origins shape: " << ray_origins.sizes() << "\n  Data:" << ray_origins << "\n";

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
vector<torch::Tensor> compute_query_points_from_rays(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    float near_thresh,
    float far_thresh,
    int num_samples
    // TODO: bool randomize=true
) {

    // TODO: test

    // shape: (num_samples)
    torch::Tensor depth_values = torch::linspace(near_thresh, far_thresh, num_samples).to(ray_origins);
    // std::cout<< "[DEBUG] depth_values shape: " << depth_values.sizes() << "\n  Data:" << depth_values << "\n";
    /*
    if (randomize) {
        // ray_origins: (width, height, 3)
        // noise_shape = (width, height, num_samples)
        // noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        auto noise_shape = (ray_origins.sizes()[0], ray_origins.sizes()[1], num_samples);
        // depth_values: (num_samples)
        auto depth_values = depth_values \
            + torch::rand(noise_shape).to(ray_origins) * (far_thresh
                - near_thresh) / num_samples;
    } 
    */

    // (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    // query_points:  (width, height, num_samples, 3)
    // query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    torch::Tensor query_points = ray_origins.index({"...", None, Slice()}) + ray_directions.index({"...", None, Slice()}) * depth_values.index({"...", Slice(), None});
    // std::cout<< "[DEBUG] query_points shape: " << query_points.sizes() << "\n  Data:" << ray_origins[0] << "\n";

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
vector<torch::Tensor> render_volume_density(
    torch::Tensor radiance_field,
    torch::Tensor ray_origins,
    torch::Tensor depth_values
) {

    // TODO: test

    // auto sigma_a = F::relu(radiance_field.index({"...", 3}));
    auto sigma_a = torch::relu(radiance_field.index({"...", 3}));
    std::cout<< "[DEBUG] sigma_a shape: " << sigma_a.sizes() << "\n";
    // rgb = torch.sigmoid(radiance_field[..., :3])
    auto rgb = torch::sigmoid(radiance_field.index({"...", Slice(None, 3)}));
    std::cout<< "[DEBUG] rgb shape: " << rgb.sizes() << "\n";
    // one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    auto one_e_10 = torch::tensor({1e10}, 
        torch::TensorOptions().dtype(ray_origins.dtype()).device(ray_origins.device())
    );
    std::cout<< "[DEBUG] one_e_10 shape: " << one_e_10.sizes() << "\n";
    //dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
    //              one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
    auto dists = at::cat((depth_values.index({"...", Slice(1, None)}) - depth_values.index({"...", Slice(None, -1)}),
                  one_e_10.expand(depth_values.index({"...", Slice(None, 1)}).sizes())), -1); // dim=-1
    std::cout<< "[DEBUG] dists shape: " << dists.sizes() << "\n";
    // alpha = 1. - torch.exp(-sigma_a * dists)
    auto alpha = 1. - torch::exp(-sigma_a * dists);
    std::cout<< "[DEBUG] alpha shape: " << alpha.sizes() << "\n";
    // weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    auto weights = alpha * cumprod_exclusive(1. - alpha + 1e-10);
    std::cout<< "[DEBUG] weights shape: " << weights.sizes() << "\n";

    // rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    auto rgb_map = (weights.index({"...", None}) * rgb).sum(-2); // dim=-2
    std::cout<< "[DEBUG] rgb_map shape: " << rgb_map.sizes() << "\n";
    // depth_map = (weights * depth_values).sum(dim=-1)
    auto depth_map = (weights * depth_values).sum(-1); // dim=-1
    std::cout<< "[DEBUG] depth_map shape: " << depth_map.sizes() << "\n";
    // acc_map = weights.sum(-1)
    auto acc_map = weights.sum(-1);
    std::cout<< "[DEBUG] acc_map shape: " << acc_map.sizes() << "\n";

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

    vector<torch::Tensor> encoding;

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
              torch::TensorOptions().dtype(tensor.dtype())
              // torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device())
        ));
    } else {
        frequency_bands = torch::linspace(
          pow(2.0, 0.0),
          pow(2.0, (num_encoding_functions - 1)),
          num_encoding_functions,
          torch::TensorOptions().dtype(tensor.dtype())
          // torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device())
        );
    }
    frequency_bands = frequency_bands.contiguous();

    // std::cout<< "[DEBUG] frequency_bands shape: " << frequency_bands.sizes() << "\n  Data:" << frequency_bands << "\n";

    // Iterate over each frequency band
    std::vector<float> frequency_bands_vector(frequency_bands.data_ptr<float>(), frequency_bands.data_ptr<float>()+frequency_bands.numel());
    /*
    float* freq_ptr = (float*) frequency_bands.data_ptr(); 
    for (int freq = 0; freq < frequency_bands.sizes()[0]; ++freq) {
        encoding.push_back(torch::sin(tensor * *freq_ptr));
        encoding.push_back(torch::cos(tensor * *freq_ptr));
        freq_ptr++;
    }*/
    for(const auto& frequency_band: frequency_bands_vector) {
        encoding.push_back(torch::sin(tensor * frequency_band));
        encoding.push_back(torch::cos(tensor * frequency_band));
    }

    // std::cout<< "[DEBUG] encoding.size() = " << encoding.size() << "\n";

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
vector<torch::Tensor> get_minibatches(torch::Tensor inputs, int chunksize=1024*8) {
    
    // TODO: test 
    vector<torch::Tensor> minibatches;

    for (int i = 0; i < inputs.sizes()[0]; i+=chunksize) {
        auto batch = inputs.index({Slice(i, i + chunksize)}); // [i:i + chunksize]
        minibatches.push_back(batch);
    }

    /*
    vector<vector<torch::jit::IValue>> minibatches;
    inputs = inputs.contiguous();

    for (int i = 0; i < inputs.sizes()[0]; i+=chunksize) {

        vector<torch::jit::IValue> batch;
        for (int j = 0; j < chunksize; j++) {
            if (i + j >= inputs.sizes()[0]) break;
            batch.push_back(inputs.index({i + j})); // [i:i + chunksize])
        } 
        //torch::jit::IValue batch = inputs.index({Slice(i, i + chunksize)}); // [i:i + chunksize]
        std::cout<< "[DEBUG] batch.size() = " << batch.size() << "\n";
        minibatches.push_back(batch);
    
    }
    */

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

    vector<torch::Tensor> res;

    // Get the "bundle" of rays through all image pixels.
    res = get_ray_bundle(height, width, focal_length, tform_cam2world);
    auto ray_origins = res[0];
    auto ray_directions = res[1];
    
    // Sample query points along each ray
    res = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    );
    auto query_points = res[0];
    std::cout<< "[DEBUG] query_points.sizes() = " << query_points.sizes() << "\n";
    auto depth_values = res[1]; 
    std::cout<< "[DEBUG] depth_values.sizes() = " << depth_values.sizes() << "\n";

    // "Flatten" the query points.
    auto flattened_query_points = query_points.reshape({-1, 3});

    // Encode the query points (default: positional encoding).
    auto encoded_points = positional_encoding(flattened_query_points, num_encoding_functions);

    // Split the encoded points into "chunks", run the model on all chunks, and
    //  concatenate the results (to avoid out-of-memory issues).
    auto batches = get_minibatches(encoded_points, chunksize);

    vector<torch::Tensor> predictions;
    for(auto &batch: batches) {

        // std::cout<< "[DEBUG] batch.sizes() = " << batch.sizes() << "\n";
        // batch is of type torch::jit::IValue
        torch::Tensor output; 
        if (!query_model(model, batch, output)) {
            std::cout << "[FATAL ERROR] Something went wrong during model query. \n";
            // TODO: error propagation
        } else {
            predictions.push_back(output);
        }

    }
    std::cout<< "[DEBUG] predictions.size() = " << predictions.size() << "\n";

    auto radiance_field_flattened = at::cat(predictions, 0); // dim=0
    std::cout<< "[DEBUG] radiance_field_flattened.sizes() = " << radiance_field_flattened.sizes() << "\n";

    // "Unflatten" to obtain the radiance field.
    std::cout<< "[DEBUG] query_points.sizes() = " << query_points.sizes() << "\n";
    // TODO: improve
    auto unflattened_shape = torch::zeros({query_points.sizes()[0], query_points.sizes()[1],  query_points.sizes()[2], 4}).sizes();
    std::cout<< "[DEBUG] unflattened_shape = " << unflattened_shape << "\n";
    auto radiance_field = torch::reshape(radiance_field_flattened, unflattened_shape);
    std::cout<< "[DEBUG] radiance_field.sizes() = " << radiance_field.sizes() << "\n";

    // Perform differentiable volume rendering to re-synthesize the RGB image.
    res = render_volume_density(radiance_field, ray_origins, depth_values);
    auto rgb_predicted = res[0];

    return rgb_predicted;

}


int main(int argc,  char* argv[]) {
  
    /* TODO: remove comment
    if (argc != 2) {
        std::cout << "usage: renderer <path-to-traced-model>\n";
        return -1;
    }
    */

    torch::jit::script::Module model;

    string path = "C:\\Users\\StefanoEsposito\\Desktop\\GitHub\\libtorch_NeRF_renderer\\traced_models\\traced_tiny_nerf.pt";

    // load_module_to_gpu(model, string(argv[1]));
    load_module_to_gpu(model, path);

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
    int image_width = 150;
    int image_height = 150;

    // Near and far clipping thresholds for depth values
    float near_thresh = 2.;
    float far_thresh = 6.;

    // Focal lenght
    float focal_length = 138.8889;

    int depth_samples_per_ray = 32;
    
    // Number of functions used in the positional encoding (Be sure to update the 
    // model if this number changes).
    int num_encoding_functions = 6;

    // Chunksize (Note: this isn't batchsize in the conventional sense. This only
    // specifies the number of rays to be queried in one go. Backprop still happens
    // only after all rays from the current "bundle" are queried and rendered).
    int chunksize = 16384; // Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.


    // torch::jit::IValue ivalue;
    // torch::Tensor tensor;
    // ivalue = tensor;

    auto tensor_image = render(model, image_height, image_width, focal_length, tform_cam2world, 
           near_thresh, far_thresh, depth_samples_per_ray,
           num_encoding_functions, chunksize);
    std::cout<< "[DEBUG] image.sizes() = " << tensor_image.sizes() << "\n";
    std::cout<< "[DEBUG] image (first colors) = " << tensor_image.index({Slice(1,3), Slice(1,3)}) << "\n";

    // Thanks to: https://stackoverflow.com/questions/59512310/convert-pytorch-tensor-to-opencv-mat-and-vice-versa-in-c
    tensor_image = tensor_image.detach();
    // float to 255 range
    tensor_image = tensor_image.mul(255).clamp(0, 255).to(torch::kU8);
    // GPU to CPU?, may not needed
    tensor_image = tensor_image.to(torch::kCPU);
    // shape of tensor
    int64_t height = tensor_image.size(0);
    int64_t width = tensor_image.size(1);
    
    // Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
    // so we must reshape tensor, otherwise we get a 3x3 grid
    tensor_image = tensor_image.reshape({width * height * 3});
    // CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
    cv::Mat cv_image(cv::Size(width, height), CV_8UC3, tensor_image.data_ptr());

    string windowName = "Output Image"; //Name of the window
    cv::namedWindow(windowName); // Create a window
    cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
    cv::imshow(windowName, cv_image); // Show our image inside the created window.
    cv::waitKey(0); // Wait for any keystroke in the window
    cv::destroyWindow(windowName); //destroy the created window
    /*
    // Create a vector of inputs.
    vector<torch::jit::IValue> inputs;
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

   return 0;

}
