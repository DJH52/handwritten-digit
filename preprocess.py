class Preprocess:
    def __init__ (self, data):
        self.data = data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def split_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.data.load_data()
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def normalise_images(self):
        self.x_train, self.x_test, self.y_train, self.y_test = self.x_train / 255.0, self.x_test / 255.0, self.y_train / 255.0, self.y_test / 255.0
        return self.x_train, self.x_test, self.y_train, self.y_test
    
