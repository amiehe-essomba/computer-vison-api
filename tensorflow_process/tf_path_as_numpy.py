from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


class ModelWrapper:
    ## Insérez votre code ici
    batch_size          : int    =32
    img_size            : tuple  =(224, 224)
    random_state        : int    =1234
    rotation_range      : float  =20.
    width_shift_range   : float  =0.2   
    height_shift_range  : float  =0.2   
    shear_range         : float  =0.2   
    zoom_range          : float  =0.2

    def __init__(self) -> None:
        pass


class dataGenerator(ModelWrapper):
    def __init__(self) -> None:
        super().__init__()

    def preprocess_image(self, image : tf):
        # redimensionner les images 
        self.image = tf.image.resize(image, self.img_size)
        self.image = self.image / 255.
        
        return image
    
    def data_gen(self) -> ImageDataGenerator:
        self.datagen = ImageDataGenerator(
            preprocessing_function  =self.preprocess_image,  
            rotation_range          =self.rotation_range,   
            width_shift_range       =self.width_shift_range,   
            height_shift_range      =self.height_shift_range,   
            shear_range             =self.shear_range,   
            zoom_range              =self.zoom_range,   
            horizontal_flip         =True,   
            fill_mode               ='nearest'   
        )
         
        return  self.datagen
    
    def load_and_preprocess_image(self, image_path : str, label : list) -> tuple:
        self.datagen = self.data_gen()

        self.image = tf.io.read_file(image_path)
        self.image = tf.image.decode_jpeg(self.image, channels=3) 
        self.image = self.datagen.preprocessing_function(self.image)
        self.image = tf.image.resize(self.image, self.img_size)
        
        return self.image, label
 
    def generator(self, **kwargs):
        X_train_path, y_train = kwargs['X_train_path'], kwargs['y_train']
        X_test_path, y_test = kwargs['X_test_path'], kwargs['y_test']
        
        # train
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train_path, y_train))
        dataset_train = dataset_train.map(self.load_and_preprocess_image)
        dataset_train = dataset_train.shuffle(buffer_size=len(X_train_path)).batch(self.batch_size)

        ## test
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test_path, y_test))
        dataset_test = dataset_test.map(self.load_and_preprocess_image)
        dataset_test = dataset_test.shuffle(buffer_size=len(X_test_path)).batch(self.batch_size) 

        return dataset_train, dataset_test
    

class config:
    decoder    = tf.keras.applications.mobilenet_v3
    l2         = tf.keras.regularizers.l2(1e-3)
    mod        = None 
    base_model = None

class base(config):
    def __init__(self) -> None:
         super().__init__()

    def build(self, shape : tuple = (224, 224,3)):
        self.base_model = tf.keras.applications.MobileNetV3Large(input_shape=shape, include_top=False, weights="imagenet")
        self.base_model.trainable = False 

        self.inputs = tf.keras.layers.Input(shape=shape)

        self.x = self.decoder.preprocess_input(self.inputs)
        self.x = self.base_model(self.x, training=False) 

        # use global avg pooling to summarize the info in each channel
        self.x = tf.keras.layers.GlobalAveragePooling2D()(self.x)
        
        # include dropout with probability of 0.2 to avoid overfitting
        self.x = tf.keras.layers.Dropout(rate=0.5)(self.x)

        # dense layer of 4096 units
        self.x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_regularizer=self.l2)(self.x)

        # dropout 
        
        # dense layer of 4096 units
        self.x = tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=self.l2)(self.x)

        # dropout 
        self.x = tf.keras.layers.Dropout(rate=0.2)(self.x)

        # use a prediction layer with one neuron (as a binary classifier only needs one)
        self.x = tf.keras.layers.Dense(units=4, activation='linear', kernel_regularizer=self.l2)(self.x)

        # ouput layer 
        self.outputs = self.x

        self.mod = tf.keras.models.Model(inputs=self.inputs, outputs=self.outputs)

        return self.mod

    def compile(self, loss_function, optimizer, metrics):
        self.mod.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=metrics
        )
    
    def callbacks(self) -> list:
        
        path_best_models = f"./data/best-model.h5"
        callbacks_best_models = tf.keras.callbacks.\
            ModelCheckpoint(filepath=path_best_models, 
                monitor="val_accuracy", verbose=1, save_best_only=True)
        

        earlystopping = tf.keras.callbacks.\
                    EarlyStopping(
                        monitor='val_accuracy',
                        patience=6,
                        min_delta=0.01,
                        verbose=1,
                        restore_best_weights=True
                    )
        
        improve_learnin_rate = tf.keras.callbacks. \
            ReduceLROnPlateau(
                        monitor="val_accuracy",
                        factor=0.1,
                        patience=5,
                        cooldown=3,
                        min_delta=0.01,
                        verbose=1
                        )
    
        return [callbacks_best_models, earlystopping, improve_learnin_rate]

    def fit(self, epochs : int = 10, **kwargs):
        self.mod.fit(kwargs['dataset_train'], 
                    epochs=epochs, 
                    validation_data=kwargs['dataset_test'],
                    callbacks=self.callbacks()
                    )
    
    def fine_tuning(self, loss_function, optimizer, metrics, epochs : int = 10, **kwargs):
        self.base_model.trainable=True 

        # Fine-tune from this layer onwards
        self.fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[ : self.fine_tune_at]:
            layer.trainable = False

        self.mod.compile(loss=loss_function, optimizer = optimizer, metrics=metrics)
        self.history             = self.mod.history
        self.fine_tune_epochs    = 20
        self.total_epochs        = epochs + self.fine_tune_epochs
        self.history_fine        = self.mod.fit(kwargs['dataset_train'],
                                                epochs=self.total_epochs,
                                                initial_epoch=self.history.epoch[-1],
                                                validation_data=kwargs['dataset_test'],
                                                callbacks=self.callbacks()
                                                )
        return self.history
    
    def evaluation(self, **kwargs):
        self.loss_test, self.score_test = self.mod.evaluate(kwargs['dataset_test'])
        self.loss_train, self.score_train = self.mod.evaluate(kwargs['dataset_train'])

        return dict(
                    train_loss=self.loss_train, 
                    score_train=self.score_train, 
                    loss_test=self.loss_test, 
                    score_test=self.score_test
                    )
    
    def prediction(self, **kwargs):
        self.y_pred_test = self.mod.predict(kwargs['dataset_test'])
        self.y_pred_train = self.mod.predict(kwargs['dataset_train'])

        return dict( y_pred_test= self.y_pred_test , y_pred_train=self.y_pred_train)