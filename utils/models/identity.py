class IdentityPredictor:
    def __init__(self, delay=0):
        pass

    def predict(self, current_input_sample, current_output_sample):
        return current_output_sample

if __name__ == '__main__':
    from utils.performance import sin_noise_test
    sin_noise_test(IdentityPredictor(), plot=True, delay=10)