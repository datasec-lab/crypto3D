use ::network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

type InputDims = (usize, usize, usize, usize, usize);

use super::*;

//(3D) input_dims(): (bath_size, in_channels, in_depth, in_height, in_width)
//(3D) kernel.dim(): (num, k_channels, k_depth, k_height, k_width)

pub fn construct_networks<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    rng: &mut R,
) -> Vec<(InputDims, NeuralNetwork<TenBitAS, TenBitExpFP>)> {
    let mut networks = Vec::new();
    let input_dims = [
        (batch_size, 3, 16, 112, 112), //1
        // (batch_size, 64, 16, 112, 112), //1
        (batch_size, 64, 16, 56, 56), //2
        (batch_size, 128, 8, 28, 28), //3a
        (batch_size, 256, 8, 28, 28), //3b
        (batch_size, 256, 4, 14, 14), //4a
        (batch_size, 512, 4, 14, 14), //4b
        (batch_size, 512, 2, 7, 7), //5a
        (batch_size, 512, 2, 7, 7), //5b
    ];
    let kernel_dims = [
        (64, 3, 3, 3, 3),
        (64, 64, 3, 3, 3),
        (128, 64, 3, 3, 3),
        (256, 128, 3, 3, 3),
        (256, 256, 3, 3, 3),
        (256, 256, 3, 3, 3),
        (512, 256, 3, 3, 3),
        (512, 512, 3, 3, 3)
    ];

    for i in 0..7 {
        let input_dims = input_dims[i];
        let kernel_dims = kernel_dims[i];
        let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Same, rng).0;
        let network = match &vs {
            Some(vs) => NeuralNetwork {
                layers:      vec![Layer::LL(conv)],
                eval_method: ::network::EvalMethod::TorchDevice(vs.device()),
            },
            None => NeuralNetwork {
                layers: vec![Layer::LL(conv)],
                ..Default::default()
            },
        };
        networks.push((input_dims, network));
    }
    networks
}
