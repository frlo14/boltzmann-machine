#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include "maths/matrix.hpp"
#include "maths/activations.hpp"
#include "maths/loss.hpp"
#include "machine/rbm.hpp"
#include "data/mnist.hpp"
#include "visualise.hpp"

int main() {
    // try catch block for debugging
    try {
        // loads training images
        MNISTloader mnist(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte"
        );
        // for now using a tenth of full dataset for speed 
        const int n_images = std::min(6000, mnist.numImages());
        std::cout << "loaded " << n_images << " training images\n";

        // loads test images
        MNISTloader test_mnist(
            "data/t10k-images.idx3-ubyte",
            "data/t10k-labels.idx1-ubyte"
        );
        const int n_test = std::min(1000, test_mnist.numImages());
        std::cout << "loaded " << n_test << " test images\n";

        // hyperparameters
        const int visible_neurons = 784;
        const int hidden_neurons = 128;
        const int batch = 32;
        const double lr = 0.01;
        const int k = 1;    

        RBM rbm(visible_neurons, hidden_neurons);

        // shuffles images 
        std::vector<int> image_indicies(n_images);
        std::iota(image_indicies.begin(), image_indicies.end(), 0);
        std::mt19937 rng(26);

        // actual training loop
        for (int i = 0; i < 10; i++) {
            std::shuffle(image_indicies.begin(), image_indicies.end(), rng);

            double loss = 0.0;
            int batches = 0;

            for (int j = 0; j < n_images - batch; j += batch) {
                for (int l = 0; l < batch; l++) {
                    int index = image_indicies[j + l];
                    rbm.contrastiveDivergence(mnist.images[index], lr, k);
                }

                // measures reconstruction loss
                int index = image_indicies[j];
                std::vector<double> reconstruction = rbm.reconstruct(mnist.images[index]);
                loss += binaryCrossEntropy(reconstruction, mnist.images[index]);
                batches++;
            }

            double val_loss = 0.0;

            for (int j = 0; j < n_test; j++) {
                std::vector<double> reconstruction = rbm.reconstruct(test_mnist.images[j]);
                val_loss += binaryCrossEntropy(reconstruction, test_mnist.images[j]);
            }

            std::cout << "epoch " << i + 1 << " / " << 10 << "  train loss: " << loss / batches << "  val loss: " << val_loss / n_test << "\n";
        }

        std::vector<std::vector<double>> vis_originals;
        std::vector<std::vector<double>> vis_reconstructions;
        std::vector<bool> seen(10, false);
 
        // adds the first of each digit tp the list of those to be visualised
        for (int i = 0; i < n_test && (int)vis_originals.size() < 10; i++) {
            int label = test_mnist.labels[i];
            if (label < 10 && !seen[label]) {
                seen[label] = true;
                vis_originals.push_back(test_mnist.images[i]);
                vis_reconstructions.push_back(rbm.reconstruct(test_mnist.images[i]));
            }
        }
 
        saveReconstructionGrid(vis_originals, vis_reconstructions, "reconstructions.pgm");
        std::cout << "saved reconstructions.pgm";

        return 0;
    }
    catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}