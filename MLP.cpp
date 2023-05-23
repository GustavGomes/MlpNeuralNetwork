#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

class NeuralNetwork {
 public:
  struct Neuron {
	std::vector<float> weights;
	float bias;
	float bias_weight;

	Neuron() : bias(1), weights(1) {
	  for (float& weight : this->weights) {
		weight = -1;
	  }
	  this->bias_weight = -1;
	}

	Neuron(size_t input_num): bias(1), weights(input_num) {
	  std::random_device rd;
	  std::mt19937 generator(rd());
	  std::uniform_real_distribution<float> distribution(-0.3, 0.3);
	  for (float& weight : this->weights) {
		weight = distribution(generator);
	  }
	  this->bias_weight = distribution(generator);
	}

	float SumInput(std::vector<float> input) {
	  float sum = this->bias * this->bias_weight;
	  for (size_t i = 0; i < input.size(); ++i) {
		sum += input[i] * weights[i];
	  }
	  return sum;
	}

	void UpdateWeights(float learning_rate, float error, std::vector<float> input) {
		for (int i = 0; i < weights.size(); i++) {
		  weights[i] += learning_rate * error * input[i];
		}
	}
  };
  std::vector<std::vector<Neuron>> neurons_per_layer;
  std::vector<std::vector<float>> neurons_per_layer_sum_output;

  explicit NeuralNetwork(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels,
						 const std::vector<size_t>& units_per_layer, const size_t& epochs, const float& learning_rate) {
	neurons_per_layer.resize(units_per_layer.size());
	neurons_per_layer_sum_output.resize(units_per_layer.size());

	for (size_t i = 0; i < units_per_layer.size(); i++) {
	  neurons_per_layer[i] = std::vector<Neuron>(units_per_layer[i]);
	  neurons_per_layer_sum_output[i] = std::vector<float>(units_per_layer[i]);
	}

	for (size_t i = 0; i < neurons_per_layer.size(); i ++) {
	  for (size_t j = 0; j < neurons_per_layer[i].size(); j++) {
		if (i == 0) {
		  neurons_per_layer[i][j] = Neuron(training_data[0].size());
		} else {
		  neurons_per_layer[i][j] = Neuron(neurons_per_layer[i - 1].size());
		}
	  }
	}

	Train(training_data, labels, epochs, learning_rate);
  }

  float relu(float x) {
	return std::max(0.0f, x);
  }

  float reluDerivative(float x) {
	return x > 0 ? 1.0f : 0.0f;
  }

  float tanh(float x) {
	return std::tanh(x);
  }

  float tanhDerivative(float x) {
	float tanhX = std::tanh(x);
	return 1 - tanhX * tanhX;
  }


  float sigmoid(float x) {
	return 1 / (1 + std::exp(-x));
  }

  float sigmoidDerivative(float x) {
	float sigmoidX = sigmoid(x);
	return sigmoidX * (1 - sigmoidX);
  }

  float Predict(const std::vector<float>& input) {
	for (size_t i = 0; i < neurons_per_layer.size(); i ++) {
	  for (size_t j = 0; j < neurons_per_layer[i].size(); j++) {
		if (i == 0) {
		  neurons_per_layer_sum_output[i][j] = tanh(neurons_per_layer[i][j].SumInput(input));
		} else {
		  neurons_per_layer_sum_output[i][j] = tanh(neurons_per_layer[i][j].SumInput(neurons_per_layer_sum_output[i - 1]));
		}
	  }
	}

	const std::vector<float>& output_layer_output = neurons_per_layer_sum_output.back();

	// Apply softmax to convert raw outputs into probabilities
	std::vector<float> probabilities(output_layer_output.size());
	float sum_exp = 0.0;
	for (size_t i = 0; i < output_layer_output.size(); i++) {
	  probabilities[i] = std::exp(output_layer_output[i]);
	  sum_exp += probabilities[i];
	}

	// Normalize the probabilities
	for (size_t i = 0; i < output_layer_output.size(); i++) {
	  probabilities[i] /= sum_exp;
	}

	// Find the index of the maximum probability
	size_t max_index = 0;
	float max_probability = probabilities[0];
	for (size_t i = 1; i < probabilities.size(); i++) {
	  if (probabilities[i] > max_probability) {
		max_index = static_cast<size_t>(i);
		max_probability = probabilities[i];
	  }
	}
	//std::cout << neurons_per_layer_sum_output.back()[0];

	return probabilities[max_index] > 0.5 ? 1 : 0;
  }

  void Train(const std::vector<std::vector<float>>& training_data, const std::vector<float>& labels, size_t num_epochs, float learning_rate) {
	std::vector<std::vector<float>> error_per_neuron;
	error_per_neuron.resize(neurons_per_layer.size());
	for (size_t i = 0; i < error_per_neuron.size(); i++) {
	  error_per_neuron[i] = std::vector<float>(neurons_per_layer[i].size());
	}

	// Perform training for the specified number of epochs
	for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
	  // Iterate over the training data and corresponding labels
	  for (size_t sample = 0; sample < training_data.size(); ++sample) {
		// Get the input and expected label for the current sample
		float output_error = labels[sample] - Predict(training_data[sample]);
		for (size_t i = neurons_per_layer.size() - 1; i >= 0; i--) {
		  if (i == neurons_per_layer.size() - 1) {
			for (size_t j = 0; j < neurons_per_layer[i].size(); j++) {
			  error_per_neuron[i][j] = tanhDerivative(neurons_per_layer_sum_output[i][j]) * output_error;
			  neurons_per_layer[i][j]
				  .UpdateWeights(learning_rate, error_per_neuron[i][j], neurons_per_layer_sum_output[i - 1]);
			}
		  } else {
			for (size_t j = 0; j < neurons_per_layer[i].size(); j++) {
			  float neuron_error = 0;
			  for (size_t k = 0; k < neurons_per_layer[i + 1].size(); k++) {
				neuron_error += neurons_per_layer[i + 1][k].weights[j] * error_per_neuron[i + 1][k];
			  }
			  error_per_neuron[i][j] = tanhDerivative(neurons_per_layer_sum_output[i][j]) * neuron_error;
			  if (i != 0) {
				neurons_per_layer[i][j]
					.UpdateWeights(learning_rate, error_per_neuron[i][j], neurons_per_layer_sum_output[i - 1]);
			  } else {
				neurons_per_layer[i][j].UpdateWeights(learning_rate, error_per_neuron[i][j], training_data[sample]);
			  }
			}
		  }
		  if (i == 0) {
			break;
		  }
		}
	  }
	}
  }
};

int main() {
  std::vector<std::vector<float>> training_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<float> labels = {0, 1, 1, 1};
  NeuralNetwork neural_network(training_data, labels, {2, 8, 2}, 100000, 0.01);
  std::cout << neural_network.Predict({0, 0}) << " " << neural_network.Predict({0, 1}) << " "
			<< neural_network.Predict({1, 0}) << " " << neural_network.Predict({1, 1}) << " " << std::endl;
  return 0;
}
