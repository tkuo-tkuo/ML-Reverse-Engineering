
class SubstituteModelGenerator():

    def __init__(self):
        self.verifier = PredictionsSimilarityEstimator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_initial_training_set(self):
        initial_training_set = None
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)
        for i, (inputs, labels) in enumerate(test_loader):
            initial_training_set = inputs.to(device)
            break 
        return initial_training_set

    def generate_substitute_model(self):
        num_of_epochs_for_reversing = 10  # ρ
        S = self.generate_initial_training_set()  # initial training set S0 (PENDING)
        print('S', S)
        # Setup a target black-box model f with architecture F
        self.verifier = PredictionsSimilarityEstimator()
        with torch.no_grad():
            for i, (weights, outputs, predictions) in enumerate(zip(self.test_weights_loader, self.test_outputs_loader, self.test_predictions_loader)):

                weights = weights.to(self.device)
                weights_of_black_box = weights[0]
                self.verifier.set_on_a_black_box_model(weights_of_black_box)
                break

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