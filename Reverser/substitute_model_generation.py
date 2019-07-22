import torch
import torchvision

from .predictions_similarity_estimator import PredictionsSimilarityEstimator
from .input_generation import WhiteboxModelExtractor

class SubstituteModelGenerator():

    def __init__(self):
        self.verifier = PredictionsSimilarityEstimator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_initial_training_set(self):
        initial_training_set = None
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=200, shuffle=True)
        for i, (inputs, labels) in enumerate(test_loader):
            initial_training_set = inputs.to(self.device)
            break 
        return initial_training_set

    def generate_target_black_box(self):
        self.whitebox_extractor = WhiteboxModelExtractor()
        weights_dataset = self.whitebox_extractor.extract_whitebox_model_weights(1)
        outputs_dataset = self.whitebox_extractor.extract_whitebox_model_outputs(1)
        predictions_dataset = self.whitebox_extractor.extract_whitebox_model_predictions(1)
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        weights_loader = torch.utils.data.DataLoader(dataset=weights_dataset, batch_size=1)
        outputs_loader = torch.utils.data.DataLoader(dataset=outputs_dataset, batch_size=1)
        predictions_loader = torch.utils.data.DataLoader(dataset=predictions_dataset, batch_size=1)


        self.verifier = PredictionsSimilarityEstimator()
        with torch.no_grad():
            for i, (weights, outputs, predictions) in enumerate(zip(self.test_weights_loader, self.test_outputs_loader, self.test_predictions_loader)):

                weights = weights.to(self.device)
                weights_of_black_box = weights[0]
                self.verifier.verify_predictions_diff(weights, predictions)
                self.verifier.set_on_a_black_box_model(weights_of_black_box)
                break

        return self.verifier

    def generate_substitute_model(self):
        num_of_epochs_for_reversing = 10  # ρ

        # Setup initial training set S0 (PENDING)
        # make it float32
        S = self.generate_initial_training_set() 
        print('S', S)

        # Setup a target black-box model f with architecture F
        f = self.generate_target_black_box()

        # Setup a model f' to approxiate f with architecture F
        # PENDING

        for _ in range(num_of_epochs_for_reversing):
            # Label the substitute training set S_i and get the relation D_i
            # PENDING

            # Train f' with D_i
            # PENDING

            # Jacobian-based dataset augmentation to get S_i+1
            # PENDING

           # Obtain f' & Evaluate f'
            # PENDING
            pass 
