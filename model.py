from keras.models import load_model

def load_emotion_model():
    model = load_model('emotion_model.h5')
    return model
